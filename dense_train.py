import sys
from data_set import CameraDataSet
from torch.utils.data import DataLoader
from dense_net import DenseNet
import torch
import torchvision
import math
import numpy as np
import os
import visdom
import visual_module as vm


def lr_change(epoch):
    epoch = epoch // 6
    return (2 ** epoch)


def train_dense_net(root_path, lr_n, start_epoch=0):
    # Step 1: Set data_loader, create net, visual
    batch_size = 4
    down_k = 5
    opts = {'dense': True}

    camera_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k) + '.csv', down_k=down_k, opts=opts)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print('Step 0: DataLoader size: %d.' % len(data_loader))
    vis_env = 'K' + str(down_k) + '_Network_Dense2'
    vis = visdom.Visdom(env=vis_env)
    win_loss = 'training_loss'
    win_lr = 'lr_change'
    win_image = 'image_set'
    win_disp = 'disp_set'

    dense_network = DenseNet()
    criterion = torch.nn.MSELoss()
    learning_rate = math.pow(0.1, lr_n)
    print('learning_rate: %.1e' % learning_rate)
    optimizer = torch.optim.SGD(dense_network.parameters(), lr=learning_rate, momentum=0.9)
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    report_period = 10
    schedular = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_change)
    print('Step 2: Model loading finished.')

    # Step 3: Training
    criterion = criterion.cuda()
    dense_network = dense_network.cuda()
    for epoch in range(start_epoch - 1, 600):
        schedular.step()
        param_group = optimizer.param_groups[0]
        now_lr = param_group['lr']
        vis.line(X=torch.FloatTensor([epoch + 0.5]), Y=torch.FloatTensor([now_lr]), win=win_lr, update='append')
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # Get data
            image_obs = data['image']
            dense_mask = data['mask']
            dense_disp_gt = data['disp']
            disp_input = data['disp_out']
            image_est = data['img_est']
            image_obs = image_obs.cuda()
            dense_mask = dense_mask.cuda()
            dense_disp_gt = dense_disp_gt.cuda()
            disp_input = disp_input.cuda()
            image_est = image_est.cuda()

            # Train
            optimizer.zero_grad()
            dense_disp_res = dense_network((image_obs, image_est, disp_input))
            loss_dense = criterion(dense_disp_res.masked_select(dense_mask), dense_disp_gt.masked_select(dense_mask))
            loss_dense.backward()

            # Optimize
            loss_add = loss_dense.item()
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
                # Visualize
                vm.dense_visual(input_set=(image_obs, image_est, disp_input, dense_mask),
                                output_set=(dense_disp_gt, dense_disp_res),
                                vis=vis, win_img=win_image, win_disp=win_disp)
            else:
                print(train_num, end='', flush=True)

        # Draw:
        epoch_average = epoch_loss / len(data_loader)
        vis.line(X=torch.FloatTensor([epoch + 0.5]), Y=torch.FloatTensor([epoch_average]), win=win_loss,
                 update='append',
                 name='epoch', opts={'markers': True})
        print('Average loss for epoch %d: %.2e' % (epoch + 1, epoch_average))
        vis.text('Average loss for epoch %d: %f<br>' % (epoch + 1, epoch_average), win='log', append=True)
        # Save Model
        torch.save(dense_network.state_dict(), './model/model_up' + str(epoch) + '.pt')
        print('Save model at epoch %d.' % (epoch + 1))

    print('Step 3: Finish training.')

    # Step 3: Save the model
    torch.save(dense_network.state_dict(), './model_up.pt')
    print('Step 4: Finish saving.')


def main(argv):
    lr_n = int(argv[1])

    # Get start epoch num
    start_epoch = 1
    if len(argv) >= 3:
        start_epoch = int(argv[2])

    train_dense_net(root_path='./SLDataSet/20181204/', start_epoch=start_epoch, lr_n=lr_n)


if __name__ == '__main__':
    assert len(sys.argv) >= 2

    main(sys.argv)
