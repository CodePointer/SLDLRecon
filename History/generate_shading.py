import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import visdom
import cv2

sys.path.append('../')
from History.data_set import CameraDataSet


def generate_shade_mat(root_path):
    # Step 1: Set data_loader
    f_tvec_mul = 1185.92488
    alpha = 10 * 63
    beta = 1010
    H = 1024
    W = 1280
    cam_mat = np.loadtxt('./sys_para/cam_mat.txt')
    batch_size = 1
    workers = 0
    opt_header = ('mask_mat', 'disp_mat')
    camera_dataset = CameraDataSet(root_path, 'TestDataList' + '3', opts=dict(header=opt_header))
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    print('Step 0: DataLoader size: %d.' % len(data_loader))

    vis_env = 'Shade_Generation'
    vis = visdom.Visdom(env=vis_env)
    w_range = torch.Tensor(range(0, W))
    h_range = torch.Tensor(range(0, H))
    x_tmp = (torch.stack([w_range] * H, dim=0) - cam_mat[0, 2]) / cam_mat[0, 0]
    y_tmp = (torch.stack([h_range] * W, dim=1) - cam_mat[1, 2]) / cam_mat[1, 1]
    z_tmp = torch.ones(H, W)
    p_tmp = torch.stack([x_tmp, y_tmp, z_tmp], dim=0).cuda()  # [3, H, W]
    print('Step 1: Initialize finished.')

    # Step 2: Process all data by looping
    report_period = 40
    disp_min = 2.0
    disp_max = -1.0
    for i, data in enumerate(data_loader, 0):
        # Get data:
        data_idx = data['idx']
        dense_mask = data['mask_mat']
        dense_disp = data['disp_mat']
        dense_mask = dense_mask.cuda()
        dense_disp = dense_disp.cuda()

        # Set disp max & min
        tmp_min = torch.min(dense_disp.masked_select(dense_mask)).item()
        tmp_max = torch.max(dense_disp.masked_select(dense_mask)).item()
        disp_min = tmp_min if tmp_min < disp_min else disp_min
        disp_max = tmp_max if tmp_max > disp_max else disp_max
        dense_mask = dense_mask.float()

        # Calculate depth
        disp_out = dense_disp.squeeze() * alpha + beta  # [H, W]
        depth_mat = f_tvec_mul / disp_out  # [H, W]
        depth_mat[dense_mask.squeeze() == 0] = 0  # [H, W]

        # Blur
        blurred = cv2.GaussianBlur(depth_mat.cpu().numpy(), ksize=(9, 9), sigmaX=3.0)
        depth_mat = torch.from_numpy(blurred).cuda()

        # Calculate 2 cross vectors
        p1_mat = p_tmp * torch.stack([depth_mat] * 3, dim=0)  # [3, H, W]
        p2_mat = torch.zeros(3, H, W).cuda()
        p3_mat = torch.zeros(3, H, W).cuda()
        p4_mat = torch.zeros(3, H, W).cuda()
        p2_mat[:, :, 1:] = p1_mat[:, :, :-1]
        p3_mat[:, 1:, :] = p1_mat[:, :-1, :]
        p4_mat[:, 1:, 1:] = p1_mat[:, :-1, :-1]

        # Cross multiply and norm
        u_mat = p4_mat - p1_mat
        v_mat = p2_mat - p3_mat
        s_mat = torch.cross(u_mat, v_mat, dim=0)  # [3, H, W]
        s_mat_norm = torch.norm(s_mat, dim=0, keepdim=True)

        shade_mat = torch.abs(-s_mat[2, :, :]) / s_mat_norm
        shade_mat[dense_mask.squeeze(0) == 0] = 0

        # Save render_mat
        shade_name = camera_dataset.get_path_by_name(name='shade_mat', idx=data_idx)
        shade_mat = shade_mat.cpu().squeeze().numpy()
        plt.imsave(shade_name, shade_mat, cmap='Greys_r')

        # Visualization and report
        train_num = '.'
        vis.image(shade_mat, win='Shade', env=vis_env)
        if i % report_period == report_period - 1:
            report_info = '[%4d/%d]' % (i + 1, len(data_loader))
            print(train_num, report_info)
        else:
            print(train_num, end='', flush=True)
    print('Step 2: Finish iteration.')
    print('[%.2f, %.2f]' % (disp_min, disp_max))


def main(argv):
    # Input parameters
    assert len(argv) >= 1

    generate_shade_mat(root_path='../SLDataSet/20190209/')


if __name__ == '__main__':
    main(sys.argv)
