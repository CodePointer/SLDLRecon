import sys
from data_set import CameraDataSet
from torch.utils.data import DataLoader
from sparse_net import SparseNet
import torch
import torchvision
import math
import numpy as np
import os
import visdom
import visual_module as vm


def lr_change(epoch):
    epoch = epoch // 1
    return 1.00 ** epoch


def train_sparse_net(root_path, lr_n, start_epoch=1):

    # Step 1: Set data_loader, create net, visual
    batch_size = 4
    down_k = 3
    opts = {'vol': False, 'disp_c': True, 'stride': 5}

    camera_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k) + '.csv', down_k=down_k, opts=opts)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Step 0: DataLoader size: %d.' % len(data_loader))
    network = SparseNet(root_path=root_path, batch_size=batch_size, down_k=down_k, opts=opts)
    if opts['vol']:
        vis_env = 'K' + str(down_k) + '_Network_Volume'
    else:
        vis_env = 'K' + str(down_k) + '_Network_Disp'
    vis = visdom.Visdom(env=vis_env)
    win_loss = 'training_loss'
    win_image = 'image_set'
    win_figure = 'vec_prob'
    win_lr = 'lr_change'
    learning_rate = math.pow(0.1, lr_n)
    print('learning_rate: %.1e' % learning_rate)
    criterion = torch.nn.MSELoss()
    print('learning_rate: %.1e' % learning_rate)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    report_period = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_change)
    pattern = camera_dataset.get_pattern()
    pattern = torch.stack([pattern] * batch_size, 0)  # BatchSize
    if os.path.exists('./model_volume.pt'):
        print('Model found. Import parameters.')
        network.load_state_dict(torch.load('./model_volume.pt'), strict=False)
        network.train()  # For BN
    print('Step 2: Model loading finished.')

    # Step 3: Training
    criterion = criterion.cuda()
    pattern = pattern.cuda()
    network = network.cuda()
    for epoch in range(start_epoch - 1, 500):
        scheduler.step()
        param_group = optimizer.param_groups[0]
        now_lr = param_group['lr']
        print('learning_rate: %.1e' % now_lr)
        vis.line(X=torch.FloatTensor([epoch + 0.5]), Y=torch.FloatTensor([now_lr]), win=win_lr, update='append')
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # Get data
            image = data['image']
            coarse_mask = data['mask_c']
            if opts['vol']:
                coarse_disp = data['disp_v']
            else:
                coarse_disp = data['disp_c']
            # Check input valid
            # img_have_nan = np.any(np.isnan(image))
            # disp_have_nan = np.any(np.isnan(coarse_disp))
            # mask_have_nan = np.any(np.isnan(coarse_mask))
            # if img_have_nan or disp_have_nan or mask_have_nan:
            #     print('Nan detect: img, disp, mask -> ', img_have_nan, disp_have_nan, mask_have_nan)
            image = image.cuda()
            coarse_disp = coarse_disp.cuda()
            coarse_mask = coarse_mask.cuda()

            # Train
            optimizer.zero_grad()
            sparse_disp = network((image, pattern))
            # print('sparse_disp: ', sparse_disp.shape)
            # print('coarse_mask: ', coarse_mask.shape)
            # print('coarse_disp: ', coarse_disp.shape)
            loss_coarse = criterion(sparse_disp.masked_select(coarse_mask), coarse_disp.masked_select(coarse_mask))
            # para_list = list(network.dn_convs[0].parameters())
            # print(para_list[1])
            loss_coarse.backward()
            optimizer.step()
            # print(para_list[1])

            # Optimize
            loss_add = loss_coarse.item()
            running_loss += loss_add
            epoch_loss += loss_add
            if np.isnan(loss_add):
                print('Error: Nan detected at set [%d].' % i)
                return

            # Visualization and report
            train_num = '.'
            if i % report_period == report_period - 1:
                average = running_loss / report_period
                running_loss = 0.0
                report_info = '[%d, %3d/%d]: %.2e' % (epoch + 1, i + 1, len(data_loader), average)
                print(train_num, report_info)
                # Draw:
                vis.line(X=torch.FloatTensor([epoch + i / len(data_loader)]), Y=torch.FloatTensor([average]),
                         win=win_loss, update='append', name='report')
                # Visualize:
                if opts['vol']:
                    vm.volume_visual(gt_v=coarse_disp, res_v=sparse_disp, mask=coarse_mask, vis=vis, win_imgs=win_image,
                                     win_fig=win_figure)
                else:
                    vm.disp_visual(gt_c=coarse_disp, res_c=sparse_disp, mask_c=coarse_mask, vis=vis,
                                   win_imgs=win_image, nrow=2)
            else:
                print(train_num, end='', flush=True)

        # Draw:
        epoch_average = epoch_loss / len(data_loader)
        vis.line(X=torch.FloatTensor([epoch + 0.5]), Y=torch.FloatTensor([epoch_average]), win=win_loss,
                 update='append', name='epoch')
        print('Average loss for epoch %d: %.2e' % (epoch + 1, epoch_average))

        # Check Parameter nan number
        param_list = list(network.parameters())
        for idx in range(0, len(param_list)):
            param = param_list[idx]
            if torch.isnan(param).any().item():
                print('Found NaN number. Epoch %d, on param[%d].' % (epoch + 1, idx))
                return
            if param.grad is not None and torch.isnan(param.grad).any().item():
                print('Found NaN grad. Epoch %d, on param[%d].' % (epoch + 1, idx))
                return

        # Save
        torch.save(network.state_dict(), './model/model_volume' + str(epoch) + '.pt')
        print('Save model at epoch %d.' % (epoch + 1))
    print('Step 3: Finish training.')

    # Step 3: Save the model
    torch.save(network.state_dict(), './model_volume.pt')
    print('Step 4: Finish saving.')


def main(argv):
    # Input parameters
    lr_n = int(argv[1])

    # Get start epoch num
    start_epoch = 1
    if len(argv) >= 3:
        start_epoch = int(argv[2])

    train_sparse_net(root_path='./SLDataSet/20181204/', start_epoch=start_epoch, lr_n=lr_n)


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    main(sys.argv)
