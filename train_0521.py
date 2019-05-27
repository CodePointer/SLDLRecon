# -*- coding: utf-8 -*-

"""
Used for predictor training. Combine with test validation.

All the related config is made in config.ini.
Usage:
    >> python3 train_0521.py config0521.ini

This file is created by QIAO RUKUN on Feb.21, 2019.
"""

# System operation package
import sys
import configparser
import os

# PyTorch & NumPy
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter

# My package
from Module.depth_net import DepthNet
from Module.data_set_loader import FlowDataSet, SampleSet
from Module.util import check_nan, AverageMeter, depth2rgb, flow2rgb


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


def train(train_loader, model, optimizer, loss_fun, epoch, train_writer):
    """Train model using train_loader for one epoch.

    Param:
        :param train_loader: Data loader for training data.
        :param model: The Network need to be trained.
        :param optimizer: Optimize network.
        :param loss_fun: Used for loss calculation.
        :param epoch: Used for visualization. Needed in train_writer.
        :param train_writer: Tensorboard summarywriter. Used for loss scalar visualization.

    Return:
        :return: train_loss. The average loss for one epoch.

    Raise:
        None.
    """
    global n_iter
    losses = AverageMeter()
    size_epoch = config.getint('Paras', 'size_epoch')
    model = model.train()
    for i, data in enumerate(train_loader):
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
        train_writer.add_scalar('train_loss', loss.item(), n_iter)

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % config.getint('Paras', 'report_iter') == 0:
            report_str = 'Epoch[%03d], Iter[%03d/%03d]: loss=%.4e' % (epoch, i + 1, size_epoch, losses.avg)
            print(report_str)

        n_iter += 1
        if i >= size_epoch:
            break
    return losses.avg


def test(test_loader, model, loss_fun, epoch, image_writers):
    """Test model using test_loader for every epoch.

    Param:
        :param test_loader: Data loader for test_data.
        :param model:       The Network used for Training.
        :param loss_fun:    The loss function for loss calculation.
        :param epoch:       Epoch number used for visualization.
        :param image_writers: The Tensorboardx SummarayWritter. Output image.

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

        # Show image result
        if i in check_pos:
            idx = check_pos[i]
            if epoch == 0:  # TODO: Check visualization part.
                image_writers[idx].add_image('Input', flow2rgb(flow1_cv[0], config.getfloat('Paras', 'flow_thred')), 0)
                image_writers[idx].add_image('Depth GroundTruth', depth2rgb(depth_cam[0], mask_cam[0]), 0)
            image_writers[idx].add_image('Depth Estimate', depth2rgb(depth_est[0], mask_cam[0]), epoch)

        if (i + 1) % config.getint('Paras', 'report_iter') == 0:
            report_str = 'Test[%03d], Iter[%03d/%03d]: loss=%.4e' % (epoch, i + 1, len(test_loader), losses.avg)
            print(report_str)

    return losses.avg


def main(main_path):
    """
    Main function for training.
    :return:
    """

    # Step 0: Set data_loader, visual
    # --------------------------------------------
    print('Step 0: DataSet initializing...', end='', flush=True)
    if not os.path.exists(os.path.join(main_path, config.get('FilePath', 'save_path'))):
        os.mkdir(os.path.join(main_path, config.get('FilePath', 'save_path')))
    train_loader = create_dataloader(flag='train', cfg=config)
    test_loader = create_dataloader(flag='test', cfg=config)
    train_writer = SummaryWriter(os.path.join(main_path, config.get('FilePath', 'save_path'), 'train'))
    test_writer = SummaryWriter(os.path.join(main_path, config.get('FilePath', 'save_path'), 'test'))
    image_writers = []
    for i in range(config.getint('Paras', 'writer_num')):
        image_writers.append(
            SummaryWriter(os.path.join(main_path, config.get('FilePath', 'save_path'), 'test%d' % i)))
    print('Finished.')
    print('  --cuda status:', cuda)
    print('  --train data load: %d' % len(train_loader))
    print('  --test  data load: %d' % len(test_loader))
    print('  --All info will be saved at', config.get('FilePath', 'save_path'))

    # Step 1: Create network model and load
    # -----------------------------------------------
    print('Step 1: Network Creating...', end='', flush=True)
    depth_net = DepthNet()
    flag_set = [False]
    tmp_path = os.path.join(main_path, config.get('FilePath', 'predictor_name') + '.pt')
    if os.path.exists(tmp_path):
        flag_set[0] = True
        depth_net.load_state_dict(torch.load(tmp_path), strict=False)
        depth_net.train()
        check_nan(depth_net)
    depth_net = depth_net.cuda() if cuda else depth_net
    print('Finished.')
    print('  --Model path: ', tmp_path)
    print('  --Load predictor model: ', flag_set[0])

    # Step 2: Create loss function, optimizer
    # --------------------------------------------------------------
    print('Step 2: Setting optimizer...', end='', flush=True)
    match_loss = torch.nn.L1Loss()
    p_lr = config.getfloat('NetworkPara', 'predictor_lr')
    optimizer_p = torch.optim.RMSprop(params=depth_net.parameters(), lr=p_lr)
    schedular_p = torch.optim.lr_scheduler.LambdaLR(optimizer_p, lr_lambda=lr_change)
    print('Step 2: Finished.')
    print('  --Predictor learning rate: %e' % config.getfloat('NetworkPara', 'predictor_lr'))

    # Step 3: Main loop. Including training/testing/visualization/storage
    # ----------------------------------------------------------------------
    print('Step 3: Start training.')
    for epoch in range(config.getint('Paras', 'start_epoch'), config.getint('Paras', 'total_epoch')):
        schedular_p.step()

        # Train for one epoch
        train_loss = train(train_loader=train_loader, model=depth_net,
                           optimizer=optimizer_p, loss_fun=match_loss,
                           epoch=epoch, train_writer=train_writer)
        train_writer.add_scalar('mean loss', train_loss, epoch)

        # Evaluation on test set
        with torch.no_grad():
            test_loss = test(test_loader=test_loader, model=depth_net,
                             loss_fun=match_loss, epoch=epoch, image_writers=image_writers)
        test_writer.add_scalar('mean loss', test_loss, epoch)

        # Save
        torch.save(depth_net.state_dict(),
                   os.path.join(main_path, config.get('FilePath', 'save_path'),
                                '%s.pt' % config.get('FilePath', 'predictor_name')))

        # Save for history model
        if (epoch + 1) % config.getint('Paras', 'save_period') == 0:
            torch.save(depth_net.state_dict(),
                       os.path.join(main_path,
                                    config.get('FilePath', 'save_path'),
                                    '%s%d.pt' % (config.get('FilePath', 'predictor_name'), epoch)))
    print('All program finished.')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    config_path = os.path.join(sys.argv[1], '000config.ini')

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)

    main(main_path=sys.argv[1])
