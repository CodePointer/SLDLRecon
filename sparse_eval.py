import sys
from data_set import CameraDataSet
from torch.utils.data import DataLoader
from sparse_net import SparseNet
import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import os
import visdom


def eval_sparse_net(root_path):

    # Step 1: Set data_loader, create net, visual
    batch_size = 1
    down_k = 4
    opts = {'vol': False}
    camera_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k) + '.csv', down_k=down_k, opts=opts)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    print('Step 0: DataLoader size: %d.' % len(data_loader))

    vis_env = 'K' + str(down_k) + '_Network_Disp_Eval'
    vis = visdom.Visdom(env=vis_env)
    win_hist = 'loss_hist'
    criterion = torch.nn.MSELoss()
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    network = SparseNet(root_path=root_path, batch_size=batch_size, down_k=down_k, opts=opts)
    network.load_state_dict(torch.load('./model_volume.pt'), strict=True)
    network.eval()
    for param in network.parameters():
        param.requires_grad = False

    pattern = camera_dataset.get_pattern()
    pattern = torch.stack([pattern] * batch_size, 0)  # BatchSize
    H_p = pattern.shape[2]
    W_p = pattern.shape[3]
    pattern_search = pattern.reshape(pattern.shape[0], pattern.shape[1], H_p * W_p)

    raw_m = torch.from_numpy(np.fromfile(root_path + 'para_M.bin', dtype='<f4'))
    para_m = raw_m.reshape((3, 1280, 1024)).transpose(1, 2)  # [3, H, W]
    raw_d = torch.from_numpy(np.fromfile(root_path + 'para_D.bin', dtype='<f4'))
    para_d = raw_d.reshape((3, 1))
    f_tvec_mul = 1185.92488
    alpha = 16.0 * 63
    beta = 716
    print('Step 2: Model loading finished.')

    # Step 3: Training
    criterion = criterion.cuda()
    pattern = pattern.cuda()
    pattern_search = pattern_search.cuda()
    network = network.cuda()
    para_m = para_m.cuda()
    para_d = para_d.cuda()
    report_period = 40

    loss_list = []
    running_loss = 0
    for i, data in enumerate(data_loader, 0):
        # Get data
        data_idx = data['idx']
        image = data['image']
        dense_mask = data['mask']
        coarse_mask = data['mask_c']
        coarse_disp = data['disp_c']
        dense_disp = data['disp']

        image = image.cuda()  # [N=1, 3, H_c, W_c]
        dense_mask = dense_mask.cuda()
        coarse_disp = coarse_disp.cuda()
        coarse_mask = coarse_mask.cuda()
        dense_disp = dense_disp.cuda()

        # Train
        sparse_disp = network((image, pattern))
        loss_coarse = criterion(sparse_disp.masked_select(coarse_mask), coarse_disp.masked_select(coarse_mask))

        loss_add = loss_coarse.item()
        loss_list.append(loss_add)
        running_loss += loss_add

        # Save disp_out
        disp_out_name = camera_dataset.get_disp_out_path(data_idx)
        disp_out_mat = torch.nn.functional.interpolate(input=sparse_disp, scale_factor=math.pow(2, down_k),
                                                       mode='bilinear', align_corners=False)
        np_disp_out = disp_out_mat.cpu().squeeze().numpy()
        np.save(disp_out_name, np_disp_out)  # disp_out: [H, W]

        # Get est_img_name and save
        est_img_name = camera_dataset.get_img_est_path(data_idx)
        disp_out_mat = disp_out_mat * alpha + beta
        tmp_mat = (disp_out_mat * para_d[2] + para_m[2, :, :] * f_tvec_mul)
        x_pro_mat = (disp_out_mat * para_d[0] + para_m[0, :, :] * f_tvec_mul) / tmp_mat
        x_pro_mat = torch.remainder(torch.round(x_pro_mat), W_p).type(torch.cuda.LongTensor)
        y_pro_mat = (disp_out_mat * para_d[1] + para_m[1, :, :] * f_tvec_mul) / tmp_mat
        y_pro_mat = torch.remainder(torch.round(y_pro_mat), H_p).type(torch.cuda.LongTensor)
        idx_pro_mat = y_pro_mat * W_p + x_pro_mat
        idx_pro_mat[dense_mask == 0] = 0
        idx_pro_vec = idx_pro_mat.reshape(image.shape[2] * image.shape[3])
        assert torch.max(idx_pro_vec).item() <= H_p * W_p - 1 and torch.min(idx_pro_vec).item() >= 0
        est_img_vec = torch.index_select(input=pattern_search, dim=2, index=idx_pro_vec)
        est_img_mat = est_img_vec.reshape(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        est_img_mat.masked_fill_(dense_mask == 0, -1)
        np_est_img = est_img_mat.cpu().squeeze().numpy()  # [3, H, W], [-1, 1]
        np_est_img = (np_est_img.transpose((1, 2, 0)) + 1) / 2
        plt.imsave(est_img_name, np_est_img)

        # Visualization and report
        train_num = '.'
        if i % report_period == report_period - 1:
            average = running_loss / report_period
            running_loss = 0.0
            report_info = '[%4d/%d]: %.2e' % (i + 1, len(data_loader), average)
            print(train_num, report_info)
            vis.histogram(X=np.array(loss_list), opts=dict(numbins=32), win=win_hist)
        else:
            print(train_num, end='', flush=True)
    print('Step 3: Finish evaluating.')


def main(argv):
    # Input parameters
    assert len(argv) >= 1

    # Get start epoch num
    start_epoch = 0
    if len(argv) >= 2:
        start_epoch = int(argv[1])

    eval_sparse_net(root_path='./SLDataSet/20181204/')


if __name__ == '__main__':
    main(sys.argv)
