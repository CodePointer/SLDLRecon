# -*- coding: utf-8 -*-

"""
Used for predictor training.

TODO: Some further notes are needed here.

This file is created by QIAO RUKUN on April.18, 2019.

"""

# System operation package
import sys
import configparser
import os

# PyTorch & NumPy
from torch.utils.data import DataLoader
import torch
import numpy as np
import visdom

# My package
import Module.visual_module as vm
from Module.depth_net import DepthNet
from Module.data_set_loader import FlowDataSet, SampleSet


def my_save(save_tensor, file_name):
    save_mat = save_tensor.cpu().detach().squeeze().numpy()
    np.save(file_name, save_mat)
    return


def check_nan(network, save_path=None):

    flag = True
    param_list = list(network.parameters())
    for idx in range(0, len(param_list)):
        param = param_list[idx]
        if torch.isnan(param).any().item():
            flag = False
            break
        if param.grad is not None and torch.isnan(param.grad).any().item():
            flag = False
            break
    try:
        assert flag
    except AssertionError as inst:
        if save_path:
            torch.save(network.state_dict(), save_path)
        print(inst)
        raise
    return True


def test():

    # Step 0: Set data_loader, visual
    # --------------------------------------------
    # DataLoader and parameters
    depth_range = [float(x) for x in config.get('DataLoader', 'depth_range').split(',')]
    opts_test = {'header': config.get('DataLoader', 'opt_header').split(','),
                 'stride': config.getint('DataLoader', 'test_stride'),
                 'bias': config.getint('DataLoader', 'test_bias')}
    test_dataset = FlowDataSet(root_dir=config.get('FilePath', 'root_path'),
                               list_name=config.get('DataLoader', 'test_list'),
                               opts=opts_test)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=0)

    print('Step 0: DataSet initialize finished.')
    print('    DataLoader size: %d' % len(test_loader))

    # Step 1: Create network model, Optimizers
    # -----------------------------------------------

    # Network
    depth_net = DepthNet(alpha_range=depth_range)
    flag_set = [False]
    if os.path.exists('model/depth_model7.pt'):
        flag_set[0] = True
        depth_net.load_state_dict(torch.load('model/depth_model7.pt'), strict=False)
        depth_net.eval()
        assert check_nan(depth_net)
    if cuda:
        depth_net = depth_net.cuda()
    print('Step 1: Network finished. Load model: ', flag_set)

    # Step 2: Main loop. Including training/testing/visualization/storage
    # ----------------------------------------------------------------------
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    iter_times = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            data = SampleSet(data)
            if cuda:
                data.to_cuda()
            else:
                data.to_cpu()
            idx_vec = data['idx']
            depth_cam = data['depth_cam']
            mask_cam = data['mask_cam']
            flow1_cv = data['flow1_cv']
            mask_flow1 = data['mask_flow1']

            depth_est = depth_net(flow1_cv)
            depth_est[mask_flow1 == 0] = 0

            # Write
            file_name = 'result/m11f%03d_depth_est.npy' % idx_vec
            my_save(depth_est, file_name)
            file_name = 'result/m11f%03d_depth_gt.npy' % idx_vec
            my_save(depth_cam, file_name)
            print('Frame %03d finished.' % idx_vec)

    print('Step 3: Finish training.')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    config_path = sys.argv[1]

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)

    cuda = True if torch.cuda.is_available() else False

    test()
