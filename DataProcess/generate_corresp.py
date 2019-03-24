"""
Use the generated disp_mat to get pro & cam view real depth mat.
Shadow effect is considered.

Input values:
- para_M, para_D
- mask_mat, disp_mat
Output values:
- disp_cam.npy
- cor_xc.npy
- cor_yc.npy
- mask_cam.png
- mask_pro.png

"""

import sys

sys.path.append('../')

from flow_data_set import FlowDataSet
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as func
import numpy as np
from matplotlib import pyplot as plt
import visdom


class Parameters:
    def __init__(self, root_dir):
        self.H = 1024
        self.W = 1280
        self.Hp = 768
        self.Wp = 1024
        self.M = torch.from_numpy(np.load(root_dir + 'para_M.npy')).float()
        self.D = torch.from_numpy(np.load(root_dir + 'para_D.npy')).float()
        self.f_tvec_mul = 1185.92488
        self.y_bias = 512.0
        self.disp_range = (1010, 1640)
        # self.scale = 10.0
        # self.Hps = self.Hp * self.scale
        # self.Wps = self.Wp * self.scale


def fill_holes(image, mask_raw, ker_size=5, avg_thred=0.55):
    avg_mask = func.avg_pool2d(mask_raw.float(), kernel_size=ker_size, stride=1, padding=ker_size // 2)
    avg_image = func.avg_pool2d(image, kernel_size=ker_size, stride=1, padding=ker_size // 2)
    fill_image = avg_image / avg_mask
    hole_mask = (mask_raw == 0) * (avg_mask > avg_thred)
    out_image = image.clone()
    out_image[hole_mask] = fill_image[hole_mask]
    out_mask = (mask_raw == 1) + hole_mask
    return out_image, out_mask


def my_imshow(image):
    npimg = image.numpy()
    plt.imshow(npimg)
    plt.show()


def generate_flow_gt(jump_k, stride):
    # Step 1: Set data_loader
    batch_size = 1
    workers = 0
    root_path = '../SLDataSet/20190209/'
    para = Parameters(root_dir=root_path)
    opt_header = ('mask_mat', 'disp_mat')  # out: mask_flow, flow_mat
    camera_dataset = FlowDataSet(root_path, 'TotalDataList3', jump_k=jump_k,
                                 opts=dict(header=opt_header, disp_range=para.disp_range))
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    print('Step 0: DataLoader size: %d.' % len(data_loader))

    # vis_env = 'FlowGene'
    # vis = visdom.Visdom(env=vis_env)
    print('Step 1: Initialize finished.')

    # Step 2: Process all data by looping
    report_period = 40
    x_cam_ori_mat = torch.arange(0, para.W).view((1, -1)).repeat((para.H, 1))
    y_cam_ori_mat = torch.arange(0, para.H).view((-1, 1)).repeat((1, para.W))
    cam_ori_mat = torch.stack((x_cam_ori_mat, y_cam_ori_mat), dim=0).reshape((1, 2, para.H, para.W))  # batch_size = 1
    for i, data in enumerate(data_loader, 0):
        # Get data:
        data_idx = data['idx']
        mask_mat = data['mask_mat']
        disp_mat = data['disp_mat']
        disp_mat = disp_mat * (para.disp_range[1] - para.disp_range[0]) + para.disp_range[0]

        # Disp -> idx_pro_mat
        tmp_mat = disp_mat * para.D[2] + para.M[2, :, :] * para.f_tvec_mul
        x_pro_mat = (disp_mat * para.D[0] + para.M[0, :, :] * para.f_tvec_mul) / tmp_mat
        y_pro_mat = (disp_mat * para.D[1] + para.M[1, :, :] * para.f_tvec_mul) / tmp_mat + para.y_bias

        # Fill pro_mat
        mask_cam = torch.zeros_like(mask_mat)
        mask_pro_tmp = torch.zeros(para.Hp, para.Wp).byte()
        cor_xc = torch.zeros(para.Hp, para.Wp)
        cor_yc = torch.zeros(para.Hp, para.Wp)
        # max_hp, min_hp = 0, para.Hp
        # max_wp, min_wp = 0, para.Wp
        for h in range(0, para.H):
            for w in range(0, para.W):
                # if h == 700 and w == 600:
                #     print('(700, 600):')
                if not mask_mat[0, 0, h, w]:
                    continue
                hp = int(y_pro_mat[0, 0, h, w])
                wp = int(x_pro_mat[0, 0, h, w])
                # if h == 700 and w == 600:
                #     print('(hp, wp):', hp, wp)
                # max_hp, min_hp = max(hp, max_hp), min(hp, min_hp)
                # max_wp, min_wp = max(wp, max_wp), min(wp, min_wp)
                if not (0 <= hp < para.Hp and 0 <= wp < para.Wp):
                    continue
                if mask_pro_tmp[hp, wp]:  # Already have value here
                    w_old, h_old = cor_xc[hp, wp].long(), cor_yc[hp, wp].long()
                    if disp_mat[0, 0, h, w] < disp_mat[0, 0, h_old, w_old]:  # Discard: jump to next pixel
                        continue
                    else:  # Update: Invalid mask_cam(h_old, w_old)
                        mask_cam[0, 0, h_old, w_old] = 0
                mask_cam[0, 0, h, w] = 1
                cor_xc[hp, wp] = w
                cor_yc[hp, wp] = h
                mask_pro_tmp[hp, wp] = 1
        mask_pro_tmp = mask_pro_tmp.reshape(1, 1, para.Hp, para.Wp).cuda()
        cor_xc = cor_xc.reshape(1, 1, para.Hp, para.Wp).float().cuda()
        cor_yc = cor_yc.reshape(1, 1, para.Hp, para.Wp).float().cuda()

        # Fill holes
        cor_xc, mask_pro = fill_holes(cor_xc, mask_pro_tmp)
        cor_yc, mask_pro = fill_holes(cor_yc, mask_pro_tmp)

        # Generate disp_cam
        disp_cam = disp_mat.clone().cuda()
        disp_cam[mask_cam == 0] = 0
        disp_cam, mask_cam = fill_holes(disp_cam, mask_cam.byte().cuda())

        # Show filled pro_mat
        # my_imshow(disp_cam.squeeze())
        # my_imshow(cor_yc.squeeze())

        # h, w = 700, 600
        # print('x_pro_mat:', x_pro_mat[0, 0, h, w])
        # print('y_pro_mat:', y_pro_mat[0, 0, h, w])

        # return

        # Save part:
        assert camera_dataset.save_item(item=disp_cam.cpu(), name='disp_cam', idx=data_idx)
        assert camera_dataset.save_item(item=cor_xc, name='cor_xc', idx=data_idx)
        assert camera_dataset.save_item(item=cor_yc, name='cor_yc', idx=data_idx)
        assert camera_dataset.save_item(item=mask_cam, name='mask_cam', idx=data_idx)
        assert camera_dataset.save_item(item=mask_pro, name='mask_pro', idx=data_idx)

        # grid to [-1, 1]
        # x_pro_mat = 2.0 * x_pro_mat.clone() / max(para.Wp - 1, 1) - 1.0
        # y_pro_mat = 2.0 * y_pro_mat.clone() / max(para.Hp - 1, 1) - 1.0
        # x_pro_mat[mask_mat == 0] = -1.5
        # y_pro_mat[mask_mat == 0] = -1.5
        # grid_pro_mat = torch.cat((x_pro_mat, y_pro_mat), dim=1)
        #
        # grid_pro_mat = grid_pro_mat.permute(0, 2, 3, 1)
        # cam_dest_mat = torch.nn.functional.grid_sample(input=cor_pro2, grid=grid_pro_mat)
        # mask_pro = (cor_pro2 > 0).float()
        # mask_flow = torch.nn.functional.grid_sample(input=mask_pro, grid=grid_pro_mat)
        # mask_flow[mask_flow < 0.9999] = 0
        # mask_flow[mask_flow > 0] = 1
        #
        # flow_mat = cam_dest_mat - cam_ori_mat
        #
        # # Save flow_mat
        # flow_mat_name = camera_dataset.get_path_by_name(name='flow_mat', idx=data_idx)
        # flow_mat = flow_mat.cpu().squeeze().numpy()  # [2, H, W]
        # np.save(flow_mat_name, flow_mat)  # disp_c: [H_c, W_c]
        #
        # # Save mask_flow
        # mask_name = camera_dataset.get_path_by_name(name='mask_flow', idx=data_idx)
        # mask_flow = mask_flow.cpu().squeeze().numpy()
        # plt.imsave(mask_name, mask_flow, cmap='Greys_r')

        # Visualization and report
        train_num = '.'
        if i % report_period == report_period - 1:
            # vis.image(flow_mat[:, 0, :, :], win='Flow_X', env=vis_env)
            report_info = '[%4d/%d]' % (i + 1, len(data_loader))
            print(train_num, report_info)
        else:
            print(train_num, end='', flush=True)
    print('Step 3: Finish evaluating.')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    generate_flow_gt(jump_k=int(sys.argv[1]), stride=5)
