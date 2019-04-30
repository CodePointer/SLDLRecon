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


def lr_change(epoch):
    epoch = epoch // config.getint('NetworkPara', 'lr_period')
    return config.getfloat('NetworkPara', 'lr_base') ** epoch


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


def spatial_train():

    # Step 0: Set data_loader, visual
    # --------------------------------------------
    # DataLoader and parameters
    depth_range = [float(x) for x in config.get('DataLoader', 'depth_range').split(',')]
    opts_train = {'header': config.get('DataLoader', 'opt_header').split(','),
                  'stride': config.getint('DataLoader', 'train_stride'),
                  'bias': config.getint('DataLoader', 'train_bias')}
    train_dataset = FlowDataSet(root_dir=config.get('FilePath', 'root_path'),
                                list_name=config.get('DataLoader', 'train_list'),
                                opts=opts_train)
    train_loader = DataLoader(train_dataset, batch_size=config.getint('Paras', 'batch_size'),
                              shuffle=True, num_workers=0)
    # opts_test = {'header': config.get('DataLoader', 'opt_header').split(','),
    #              'stride': config.getint('DataLoader', 'test_stride'),
    #              'bias': config.getint('DataLoader', 'test_bias')}
    # test_dataset = FlowDataSet(root_dir=config.get('FilePath', 'root_path'),
    #                            list_name=config.get('DataLoader', 'test_list'),
    #                            opts=opts_test)
    # test_loader = DataLoader(test_dataset, batch_size=config.getint('Paras', 'batch_size'),
    #                          shuffle=True, num_workers=0)

    # Visdom setting
    vis_env = config.get('Paras', 'vis_env')
    vis = visdom.Visdom(env=vis_env)

    print('Step 0: DataSet initialize finished.')
    # print('    DataLoader size (tr/te): (%d/%d).' % (len(train_loader), len(test_loader)))
    print('    DataLoader size: %d' % len(train_loader))

    # Step 1: Create network model, Optimizers
    # -----------------------------------------------

    # Loss function
    rigid_loss = torch.nn.L1Loss()

    # Network
    depth_net = DepthNet(alpha_range=depth_range)
    flag_set = [False]
    if os.path.exists(config.get('FilePath', 'depth_model') + '.pt'):
        flag_set[0] = True
        depth_net.load_state_dict(torch.load(config.get('FilePath', 'depth_model') + '.pt'), strict=False)
        depth_net.train()
        assert check_nan(depth_net)
    if cuda:
        depth_net = depth_net.cuda()
        rigid_loss = rigid_loss.cuda()
    print('Step 1: Network finished. Load model: ', flag_set)

    # Optimizers
    g_lr = config.getfloat('NetworkPara', 'predictor_lr')
    optimizer_g = torch.optim.RMSprop(params=depth_net.parameters(), lr=g_lr)
    schedular_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lr_change)
    print('Step 2: Optimizers setting finished.')

    # Step 2: Main loop. Including training/testing/visualization/storage
    # ----------------------------------------------------------------------
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    report_period = config.getint('Paras', 'report_period')
    save_period = config.getint('Paras', 'save_period')
    iter_times = 0
    for epoch in range(config.getint('Paras', 'start_epoch'), config.getint('Paras', 'total_epoch')):
        # ................. #
        # 2.1 Train part    #
        # ................. #
        g_loss_running = 0.0
        g_loss_epoch = 0.0
        schedular_g.step()
        for i, data in enumerate(train_loader, 0):
            # Get data
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

            # Train Easy Version: Use depth map for error calculation.
            optimizer_g.zero_grad()
            depth_est = depth_net(flow1_cv)
            g_loss = rigid_loss(depth_est.masked_select(mask_flow1), depth_cam.masked_select(mask_flow1))
            g_loss.backward()
            optimizer_g.step()
            g_loss_running += g_loss.item()
            g_loss_epoch += g_loss.item()
            iter_times += 1

            # Report: draw depth map and loss line.
            now_lr = optimizer_g.param_groups[0]['lr']
            report_info = vm.iter_report(vis=vis, win_set=config['WinSet'], input_set=(
                (iter_times, g_loss_running, now_lr),))
            g_loss_running = 0
            print(report_info)

            # Visualization:
            if (i + 1) % report_period == 0:
                vm.show_report(vis=vis, win_set=config['WinSet'], input_set=((depth_est, depth_cam, mask_cam),))
                g_loss_running = 0
                check_nan(depth_net, save_path=config.get('FilePath', 'depth_model') + '_error.pt')

        # 2.2 Epoch visualization:
        print('Epoch[%d] finished.' % epoch)
        epoch_loss = g_loss_epoch / len(train_loader)
        vm.epoch_report(vis, config['WinSet'], input_set=((epoch, epoch_loss),))

        # Check Parameter nan number
        check_nan(depth_net, save_path=config.get('FilePath', 'depth_model') + '_error.pt')

        # Save
        if epoch % save_period == save_period - 1:
            torch.save(depth_net.state_dict(),
                       ''.join([config.get('FilePath', 'save_model'),
                                config.get('FilePath', 'depth_model'),
                                str(epoch),
                                '.pt']))
            print('    Save model at epoch %d.' % epoch)
        torch.save(depth_net.state_dict(), config.get('FilePath', 'depth_model') + '.pt')

        # ------------
        # Test part:
        # ------------
        # with torch.no_grad():
        #     g_loss_test = 0
        #     for i, data in enumerate(test_loader, 0):
        #         # 1. Get data
        #         data = SampleSet(data)
        #         if cuda:
        #             data.to_cuda()
        #         else:
        #             data.to_cpu()
        #         mask_cam = data['mask_cam']
        #         disp_cam = data['disp_cam']
        #         cor_xc = data['cor_xc']
        #         cor_xc_t = data['cor_xc_t']
        #         cor_yc = data['cor_yc']
        #         cor_yc_t = data['cor_yc_t']
        #         mask_pro = data['mask_pro']
        #         flow_mat, mask_flow = flow_estimator.get_flow_value(disp_cam, mask_cam, cor_xc_t, cor_yc_t, mask_pro,
        #                                                             cor_xc, cor_yc)
        #         op_center = flow_estimator.get_op_center(flow_mat, mask_flow)
        #         flow_estimator.set_cam_info(op_center)
        #
        #         # Test Easy Version:
        #         alpha_mat = depth_net(flow_mat)
        #         disp_fake, disp_fake_t, mask_flow_t = flow_estimator.alpha2disps(alpha_mat, flow_mat, mask_flow)
        #         g_loss = rigid_loss(disp_fake.masked_select(mask_flow.byte()), disp_cam.masked_select(mask_flow.byte()))
        #         g_loss_test += g_loss.item()
        #         print('.', end='', flush=True)
        #         # Save
        #         np.save('show_output/disp_real_test%d.npy' % (i + 1), disp_cam.cpu().numpy())
        #         np.save('show_output/disp_fake_test%d.npy' % (i + 1), disp_fake.cpu().numpy())
        #         np.save('show_output/mask_flow_test%d.npy' % (i + 1), mask_flow.cpu().numpy())
        #
        #     report_info = vm.iter_visual_test(vis=vis, win_set=config['WinSet'], input_set=(
        #         (epoch, len(test_loader)),
        #         (g_loss_test, g_loss_test)))
        #     print(report_info)

    print('Step 3: Finish training.')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    config_path = sys.argv[1]

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)

    cuda = True if torch.cuda.is_available() else False

    spatial_train()
