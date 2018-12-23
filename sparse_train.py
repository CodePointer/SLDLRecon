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


def train_sparse_net(opts, root_path, start_epoch=0):

    # Step 1: Set data_loader, create net, visual
    batch_size = 16
    down_k = 5
    camera_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k) + 'v.csv', down_K=down_k, opts=opts)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Step 0: DataLoader size: %d.' % len(data_loader))
    network = SparseNet(root_path=root_path, batch_size=batch_size, down_K=down_k, opts=opts)
    if opts['vol']:
        vis_env = 'K' + str(down_k) + '_Network_Volume'
    else:
        vis_env = 'K' + str(down_k) + '_Network_Disp'
    vis = visdom.Visdom(env=vis_env)
    win_epoch = 'epoch_loss'
    win_report = 'report_loss'
    win_image = 'image_set'
    win_figure = 'vec_prob'
    learning_rate = 1e-7
    if opts['vol']:
        learning_rate = 1e-2
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    report_period = 10
    save_period = 125
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
    for epoch in range(start_epoch, 300):
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
            loss_coarse = criterion(sparse_disp.masked_select(coarse_mask), coarse_disp.masked_select(coarse_mask))
            loss_coarse.backward()

            # Check parameters validation
            param_list = list(network.parameters())
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
            loss_add = loss_coarse.item()
            running_loss += loss_add
            epoch_loss += loss_add
            if np.isnan(loss_add):
                print('Error: Nan detected at set [%d].' % i)
                return

            # Visualization and report
            train_num = '[%d, %4d/%d](%.2e)' % (epoch + 1, i + 1, len(data_loader), loss_add)
            if i % report_period == report_period - 1:
                average = running_loss / report_period
                running_loss = 0.0
                report_info = ', MSE-loss=%.2e' % average
                print(train_num, report_info)
                # Draw:
                vis.line(X=torch.FloatTensor([epoch + i / len(data_loader)]), Y=torch.FloatTensor([average]),
                         win=win_report, update='append')
                # Visualize:
                if opts['vol']:
                    vm.volume_visual(gt_v=coarse_disp, res_v=sparse_disp, mask=coarse_mask, vis=vis, win_imgs=win_image,
                                     win_fig=win_figure)
                else:
                    vm.disp_visual(gt_c=coarse_disp, res_c=sparse_disp, mask_c=coarse_mask, vis=vis,
                                   win_imgs=win_image, nrow=4)
            else:
                print(train_num)

            # Save model
            if i % save_period == save_period - 1:
                torch.save(network.state_dict(), './model/model_volume' + str(epoch) + '.pt')
                print('Save model at epoch %d after %3d dataset.' % (epoch + 1, i + 1))

        epoch_average = epoch_loss / len(data_loader)
        # Draw:
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([epoch_average]), win=win_epoch, update='append')
        print('Average loss for epoch %d: %.2e' % (epoch + 1, epoch_average))
        # vis.text('Average loss for epoch %d: %f<br>' % (epoch + 1, epoch_average), win='log', append=True)
    print('Step 3: Finish training.')

    # Step 3: Save the model
    torch.save(network.state_dict(), './model_volume.pt')
    print('Step 4: Finish saving.')


def main(argv):
    # Input parameters
    assert len(argv) >= 2

    opts = {}
    if argv[1] == 'Vol':
        opts['vol'] = True
    elif argv[1] == 'Disp':
        opts['vol'] = False
    else:
        return

    # Get start epoch num
    start_epoch = 0
    if len(argv) >= 3:
        start_epoch = int(argv[2])

    train_sparse_net(opts=opts, root_path='./SLDataSet/20181204/', start_epoch=start_epoch)


if __name__ == '__main__':
    main(sys.argv)
