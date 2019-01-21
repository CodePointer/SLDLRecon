import sys
from data_set import CameraDataSet
from torch.utils.data import DataLoader
from Module.generator_net import GeneratorNet
import torch
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import visdom
import visual_module as vm


def lr_change(epoch):
    epoch = epoch // 300
    return 0.5 ** epoch


def train_together(root_path, lr_n, start_epoch=1):

    # Step 1: Set data_loader, create net, visual
    batch_size = 1
    pattern_network = GeneratorNet(root_path=root_path, batch_size=batch_size)

    vis_env = 'K' + str(3) + '_Network_PatTest'
    vis = visdom.Visdom(env=vis_env)
    win_loss = 'training_loss'
    win_images = 'image_set'
    win_pattern = 'pattern'

    learning_rate = math.pow(0.1, lr_n)
    print('learning_rate: %.1e' % learning_rate)

    criterion = torch.nn.MSELoss()
    pattern_opt = torch.optim.SGD(pattern_network.parameters(), lr=learning_rate, momentum=0.9)
    # sparse_opt = torch.optim.SGD(sparse_network.parameters(), lr=learning_rate, momentum=0.9)
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    pattern_schedular = torch.optim.lr_scheduler.LambdaLR(pattern_opt, lr_lambda=lr_change)
    if os.path.exists('./model_pattern.pt'):
        print('Pattern Model found. Import parameters.')
        pattern_network.load_state_dict(torch.load('./model_pattern.pt'), strict=False)
        pattern_network.train()
    print('Step 2: Model loading finished.')

    # Step 3: Training
    report_period = 100
    save_period = 100
    criterion = criterion.cuda()
    pattern_network = pattern_network.cuda()
    pattern_seed = torch.from_numpy(np.load('random_seed.npy')).float().cuda()
    pattern_seed = (pattern_seed - 0.5) * 2
    pattern_seed = torch.stack([pattern_seed] * batch_size, dim=0).unsqueeze(1)  # [N, 1, X]

    pattern_gt = plt.imread(root_path + 'pattern_part0.png')
    pattern_gt = torch.from_numpy((pattern_gt.transpose(2, 0, 1) - 0.5) * 2)
    pattern_gt = pattern_gt.unsqueeze(0).cuda()  # [N, 3, H, W]

    cam_height = 1024
    cam_width = 1280
    pro_height = 128
    pro_width = 1024
    for epoch in range(start_epoch - 1, 5000):
        epoch_loss = 0.0
        pattern_schedular.step()
        param_group = pattern_opt.param_groups[0]
        # print(param_group)
        now_lr = param_group['lr']
        print('learning_rate: %.1e' % now_lr)

        for i in range(0, report_period):
            pattern_opt.zero_grad()

            # Generate pattern
            sparse_pattern = pattern_network(pattern_seed)
            dense_pattern = torch.nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
                                                            align_corners=False)
            pattern_mat = dense_pattern
            # print(torch.max(pattern_mat), torch.min(pattern_mat))
            # vis.image(pattern_mat[0, :, :, :] / 2 + 0.5, win=win_pattern)

            # Train
            loss_pat = criterion(pattern_mat, pattern_gt)
            loss_pat.backward()
            pattern_opt.step()

            # Optimize
            loss_add = loss_pat.item()
            epoch_loss += loss_add
            if np.isnan(loss_add):
                print('Error: Nan detected at set [%d].' % i)
                return

            # Visualization and report
            # print('.', end='', flush=True)

        average = epoch_loss / report_period
        report_info = '[%d]: %.2e' % (epoch + 1, average)
        print(report_info)
        # Draw:
        vis.line(X=torch.FloatTensor([epoch + 0.5]), Y=torch.FloatTensor([average]), win=win_loss,
                 update='append', name='epoch', opts=dict(showlegend=True))
        pattern_show = dense_pattern[0, :, :, :]
        pattern_gt_show = pattern_gt[0, :, :, :]
        pattern_box = sparse_pattern[0, :, :, :].reshape(sparse_pattern.shape[1], sparse_pattern.shape[2] *
                                                         sparse_pattern.shape[3]).transpose(1, 0)
        vis.image((pattern_show / 2) + 0.5, win=win_pattern)
        vis.image((pattern_gt_show / 2) + 0.5, win='Input pattern')
        vis.boxplot(X=pattern_box, opts=dict(legend=['R', 'G', 'B']), win='Pattern_box')
        # Check Parameter nan number
        param_list = list(pattern_network.parameters())
        for idx in range(0, len(param_list)):
            param = param_list[idx]
            if torch.isnan(param).any().item():
                print('Found NaN number. Epoch %d, on param[%d].' % (epoch + 1, idx))
                return
            if param.grad is not None and torch.isnan(param.grad).any().item():
                print('Found NaN grad. Epoch %d, on param[%d].' % (epoch + 1, idx))
                return

        # Save
        pattern = pattern_mat[0, :, :, :].detach().cpu().numpy()
        if epoch % save_period == save_period - 1:
            torch.save(pattern_network.state_dict(), './model_pair/model_pattern' + str(epoch) + '.pt')
            np.save('./model_pair/res_pattern' + str(epoch) + '.npy', pattern)
            print('Save model at epoch %d.' % (epoch + 1))
        torch.save(pattern_network.state_dict(), './model_pattern.pt')
        np.save('./res_pattern.npy', pattern)
    print('Step 3: Finish training.')


def main(argv):
    # Input parameters
    lr_n = int(argv[1])

    # Get start epoch num
    start_epoch = 1
    if len(argv) >= 3:
        start_epoch = int(argv[2])

    train_together(root_path='./SLDataSet/20181204/', start_epoch=start_epoch, lr_n=lr_n)


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    main(sys.argv)
