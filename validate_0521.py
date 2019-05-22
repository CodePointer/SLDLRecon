# -*- coding: utf-8 -*-

"""
Used for predictor for test. Result will be saved for visualization.

All the related config is made in config.ini.
Usage:
    >> python3 validate_0521.py config0521.ini

This file is created by QIAO RUKUN on Feb.22, 2019.
"""

# System operation package
import sys
import configparser
import os

# PyTorch & NumPy
from torch.utils.data import DataLoader
import torch
import numpy as np
from tensorboardX import SummaryWriter

# My package
from Module.depth_net import DepthNet
from Module.data_set_loader import FlowDataSet, SampleSet
from Module.util import check_nan, AverageMeter


n_iter = 0
cuda = True if torch.cuda.is_available() else False


def lr_change(epoch):
    epoch = epoch // config.getint('NetworkPara', 'lr_period')
    return config.getfloat('NetworkPara', 'lr_base') ** epoch


def create_dataloader(flag, cfg):
    """
    Create dataloader for training or testing.

    Parameters are written in config.ini file.
    Only set set_name to 'train' or 'test' for control.
    The input depth are normalized.

    Param:
        :param flag: 'train' or 'test'

    Return:
        :return: dataloader for train set or test set.

    Raise:
        AssertionError: input flag is not 'train' or 'test'.
    """
    assert flag in ['train', 'test']
    opts = {'header': cfg.get('DataLoader', 'opt_header').split(','),
            'stride': cfg.getint('DataLoader', '%s_stride' % flag),
            'bias': cfg.getint('DataLoader', '%s_bias' % flag),
            'depth_range': [float(x) for x in cfg.get('DataLoader', 'depth_range').split(',')]}
    data_set = FlowDataSet(root_dir=cfg.get('FilePath', 'data_path'),
                           header_name=cfg.get('DataLoader', 'header_name'),
                           list_name=cfg.get('DataLoader', '%s_name' % flag),
                           batch_size=cfg.getint('Paras', 'batch_size'),
                           opts=opts)
    data_loader = DataLoader(data_set, batch_size=cfg.getint('Paras', 'batch_size'),
                             shuffle=cfg.getboolean('DataLoader', '%s_shuffle' % flag),
                             num_workers=cfg.getint('Paras', 'workers'))
    return data_loader


def evaluate(test_loader, model, loss_fun):
    """Evaluate result by using test_loader for every data.

    Param:
        :param test_loader: Data loader for test_data.
        :param model:       The Network used for Training.
        :param loss_fun:    The loss function for loss calculation.

    Return:
        :return: test loss result. Average.

    Raise:
        None
    """
    losses = AverageMeter()
    model = model.eval()
    # Set output writer's number
    step = len(test_loader) // config.getint('Paras', 'writer_num')
    check_pos = {}
    for i in range(0, config.getint('Paras', 'writer_num')):
        check_pos[int((i + 0.5) * step)] = i

    for i, data in enumerate(test_loader):
        # Get data
        data = SampleSet(data)
        data.to_cuda() if cuda else data.to_cpu()
        depth_cam = data['depth_cam']
        mask_cam = data['mask_cam']
        flow1_cv = data['flow1_cv_est']
        mask_flow1 = data['mask_flow1']

        # Compute output
        depth_est = model(flow1_cv)
        loss = loss_fun(depth_est.masked_select(mask_flow1), depth_cam.masked_select(mask_flow1))
        losses.update(loss, depth_cam.size(0))

        # Write result
        depth_est[mask_cam == 0] = 0
        for n in range(depth_est.size(0)):
            depth_est_np = depth_est[n].cpu().squeeze().numpy()
            idx = data['idx'][n].item()
            np.save('%sdepth_%d.npy' % (config.get('FilePath', 'save_path'), idx), depth_est_np)

        # Print report
        if (i + 1) % config.getint('Paras', 'report_iter') == 0:
            report_str = 'Test, Iter[%03d/%03d]: loss=%.4e' % (i + 1, len(test_loader), losses.avg)
            print(report_str)

    return losses.avg


def main():
    """
    Main function for validation.
    :return:
    """

    # Step 0: Set data_loader, visual
    # --------------------------------------------
    print('Step 0: DataSet initializing...', end='', flush=True)
    test_loader = create_dataloader(flag='test', cfg=config)
    image_writers = []
    for i in range(config.getint('Paras', 'writer_num')):
        image_writers.append(SummaryWriter(config.get('FilePath', 'save_path') + 'test' + str(i)))
    print('Finished.')
    print('  --cuda status:', cuda)
    print('  --test  data load: %d' % len(test_loader))
    print('  --All info will be saved at', config.get('FilePath', 'save_path'))

    # Step 1: Create network model and load
    # -----------------------------------------------
    print('Step 1: Network Creating...', end='', flush=True)
    depth_net = DepthNet()
    assert os.path.exists(config.get('FilePath', 'predictor_name') + '.pt')
    depth_net.load_state_dict(torch.load(config.get('FilePath', 'predictor_name') + '.pt'), strict=False)
    depth_net.eval()
    check_nan(depth_net)
    depth_net = depth_net.cuda() if cuda else depth_net
    print('Finished.')
    print('  --Model path: ', config.get('FilePath', 'predictor_name') + '.pt')
    print('  --Load predictor model finished.')

    # Step 2: Create loss function, optimizer
    # --------------------------------------------------------------
    print('Step 2: Setting optimizer...', end='', flush=True)
    match_loss = torch.nn.L1Loss()
    print('Step 2: Finished.')

    # Step 3: Main loop. Including training/testing/visualization/storage
    # ----------------------------------------------------------------------
    print('Step 3: Start evaluating.')
    with torch.no_grad():
        test_loss = evaluate(test_loader=test_loader, model=depth_net,
                             loss_fun=match_loss)
    print('  --Test loss: %.4e' % test_loss)

    print('All program finished.')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    config_path = sys.argv[1]

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)

    main()
