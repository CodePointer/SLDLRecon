import sys
from data_set import CameraDataSet
from torch.utils.data import DataLoader
from up_net import UpNet
from sparse_net import SparseNet
import torch
import torchvision
import math
import numpy as np
import os
import visdom
import visual_module as vm


def train_dense_net(root_path, sparse_path, start_epoch=0):

    # Step 1: Set data_loader, create net, visual
    batch_size = 2
    down_k = 5
    opts = {'vol': False}

    camera_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k) + 'v.csv', down_k=down_k, opts=opts)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print('Step 0: DataLoader size: %d.' % len(data_loader))
    vis_env = 'K' + str(down_k) + '_Network_Dense'
    vis = visdom.Visdom(env=vis_env)
    win_epoch = 'epoch_loss'
    win_report = 'report_loss'
    win_image = 'image_set'
    win_figure = 'vec_prob'

    sparse_network = SparseNet(root_path=root_path, batch_size=batch_size, down_k=down_k, opts=opts)
    dense_network = UpNet(down_k=down_k)
    learning_rate = 1e-4
    criterion = torch.nn.MSELoss()
    print('learning_rate: %.1e', learning_rate)
    optimizer = torch.optim.SGD(dense_network.parameters(), lr=learning_rate, momentum=0.9)
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    report_period = 40
    save_period = 1000
    pattern = camera_dataset.get_pattern()
    pattern = torch.stack([pattern] * batch_size, 0)  # BatchSize
    sparse_network.load_state_dict(torch.load(sparse_path), strict=True)
    sparse_network.train()
    for param in sparse_network.parameters():
        param.requires_grad = False
    print('Step 2: Model loading finished.')

    # Step 3: Training
    criterion = criterion.cuda()
    pattern = pattern.cuda()
    sparse_network = sparse_network.cuda()
    dense_network = dense_network.cuda()
    for epoch in range(start_epoch, 50):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # Get data
            image = data['image']
            dense_mask = data['mask']
            dense_disp = data['disp']
            # coarse_mask = data['mask_c']
            # coarse_disp = data['disp_c']
            # Check input valid
            # img_have_nan = np.any(np.isnan(image))
            # disp_have_nan = np.any(np.isnan(dense_disp))
            # mask_have_nan = np.any(np.isnan(dense_mask))
            # if img_have_nan or disp_have_nan or mask_have_nan:
            #     print('Nan detect: img, disp, mask -> ', img_have_nan, disp_have_nan, mask_have_nan)
            image = image.cuda()
            dense_mask = dense_mask.cuda()
            dense_disp = dense_disp.cuda()
            # coarse_mask = coarse_mask.cuda()

            # Train
            optimizer.zero_grad()
            down_sparse = sparse_network((image, pattern))
            up_dense, up_sparse = dense_network((image, down_sparse))
            loss_dense = criterion(up_dense.masked_select(dense_mask), dense_disp.masked_select(dense_mask))
            loss_dense.backward()

            # Check parameters validation
            param_list = list(dense_network.parameters())
            for idx in range(0, len(param_list)):
                param = param_list[idx]
                if torch.isnan(param).any().item():
                    print('Found nan parameter.', i, '->', idx)
                    print(param.shape)
                    return
                if param.grad is not None and torch.isnan(param.grad).any().item():
                    print('Found nan grad.', i, '->', idx)
                    print(param.grad)
                    return

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
                         win=win_report, update='append')
                # Visualize
                vm.dense_visual(gt_d=dense_disp, res_d=up_dense, inter_d=up_sparse, mask_d=dense_mask, vis=vis,
                                win_imgs=win_image)
            else:
                print(train_num, end='', flush=True)

            # Save model
            if i % save_period == save_period - 1:
                torch.save(dense_network.state_dict(), './model/model_up' + str(epoch) + '.pt')
                print('Save model at epoch %d after %3d dataset.' % (epoch + 1, i + 1))

        epoch_average = epoch_loss / len(data_loader)
        # Draw:
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([epoch_average]), win=win_epoch, update='append')
        print('Average loss for epoch %d: %.2e' % (epoch + 1, epoch_average))
        vis.text('Average loss for epoch %d: %f<br>' % (epoch + 1, epoch_average), win='log', append=True)
    print('Step 3: Finish training.')

    # Step 3: Save the model
    torch.save(dense_network.state_dict(), './model_up.pt')
    print('Step 4: Finish saving.')


def main(argv):
    # Get start epoch num
    start_epoch = 0
    if len(argv) >= 2:
        start_epoch = int(argv[2])

    train_dense_net(root_path='./SLDataSet/20181204/', sparse_path='./model_volume.pt', start_epoch=start_epoch)


if __name__ == '__main__':
    main(sys.argv)
