import sys
from data_set import CameraDataSet
from torch.utils.data import DataLoader
from Module.generator_net import GeneratorNet
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


def train_together(root_path, lr_n, start_epoch=1):

    # Step 1: Set data_loader, create net, visual
    batch_size = 4
    down_k = 3
    opts = {'vol': False, 'disp_c': True, 'stride': 10, 'shade': True, 'idx_vec': True}

    camera_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k), down_k=down_k, opts=opts)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Step 0: DataLoader size: %d.' % len(data_loader))
    pattern_network = GeneratorNet(root_path=root_path, batch_size=batch_size)
    sparse_network = SparseNet(root_path=root_path, batch_size=batch_size, down_k=down_k)

    vis_env = 'K' + str(down_k) + '_Network_Gene'
    vis = visdom.Visdom(env=vis_env)
    win_loss = 'training_loss'
    win_images = 'image_set'
    win_pattern = 'pattern'

    learning_rate = math.pow(0.1, lr_n)
    print('learning_rate: %.1e' % learning_rate)

    criterion = torch.nn.MSELoss()
    pattern_opt = torch.optim.SGD(pattern_network.parameters(), lr=learning_rate, momentum=0.9)
    sparse_opt = torch.optim.SGD(sparse_network.parameters(), lr=learning_rate, momentum=0.9)
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    pattern_schedular = torch.optim.lr_scheduler.LambdaLR(pattern_opt, lr_lambda=lr_change)
    sparse_scheduler = torch.optim.lr_scheduler.LambdaLR(sparse_opt, lr_lambda=lr_change)
    if os.path.exists('./model_sparse.pt'):
        print('Sparse Model found. Import parameters.')
        sparse_network.load_state_dict(torch.load('./model_sparse.pt'), strict=False)
        sparse_network.train()  # For BN
    if os.path.exists('./model_pattern.pt'):
        print('Pattern Model found. Import parameters.')
        pattern_network.load_state_dict(torch.load('./model_pattern.pt'), strict=False)
        pattern_network.train()
    print('Step 2: Model loading finished.')

    # Step 3: Training
    report_period = 20
    save_period = 10
    criterion = criterion.cuda()
    sparse_network = sparse_network.cuda()
    pattern_network = pattern_network.cuda()
    pattern_seed = torch.from_numpy(np.load('random_seed.npy')).float().cuda()
    pattern_seed = (pattern_seed - 0.5) * 2
    pattern_seed = torch.stack([pattern_seed] * batch_size, dim=0).unsqueeze(1)
    cam_height = 1024
    cam_width = 1280
    pro_height = 128
    pro_width = 1024
    for epoch in range(start_epoch - 1, 5000):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # Get data
            disp_mat = data['disp_mat'].cuda()
            mask_mat = data['mask_mat'].cuda()
            disp_c = data['disp_c'].cuda()
            mask_c = data['mask_c'].cuda()
            shade_mat = data['shade_mat'].cuda()
            idx_vec = data['idx_vec'].cuda()
            pattern_opt.zero_grad()
            sparse_opt.zero_grad()

            # Generate pattern
            pattern_mat = pattern_network(pattern_seed)  # [N, C=3, Hp, Wp]
            # print(torch.max(pattern_mat), torch.min(pattern_mat))
            # vis.image(pattern_mat[0, :, :, :] / 2 + 0.5, win=win_pattern)

            # Get Image
            # print('pattern_mat: ', pattern_mat.shape)
            # print('idx_vec: ', idx_vec.shape)
            pattern_rearrange = pattern_mat.transpose(1, 0)
            pattern_search = pattern_rearrange.reshape(3, batch_size * pro_height * pro_width)
            idx_vec_plain = idx_vec.reshape(batch_size * cam_height * cam_width)
            est_img_vec = torch.index_select(input=pattern_search, dim=1, index=idx_vec_plain)
            # print('est_img_vec:', est_img_vec.shape)
            image_mat = est_img_vec.reshape(3, batch_size, disp_mat.shape[2], disp_mat.shape[3]).transpose(1, 0)
            # print('image_mat:', image_mat.shape)
            image_mat.masked_fill_(mask_mat == 0, -1)
            # print('shade_mat:', shade_mat.shape)
            image_mat = ((image_mat / 2 + 0.5) * shade_mat - 0.5) * 2
            # vis.image(shade_mat[0, :, :, :], win="shade")
            # vis.image(image_mat[0, :, :, :] / 2 + 0.5, win="Est_Image")

            # Get Disp
            sparse_disp = sparse_network((image_mat, pattern_mat))

            # Train
            loss_coarse = criterion(sparse_disp.masked_select(mask_c), disp_c.masked_select(mask_c))
            loss_coarse.backward()
            sparse_opt.step()
            pattern_opt.step()

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
                report_info = '[%d, %4d/%d]: %.2e' % (epoch + 1, i + 1, len(data_loader), average)
                print(train_num, report_info)
                # Draw:
                vis.line(X=torch.FloatTensor([epoch + i / len(data_loader)]), Y=torch.FloatTensor([average]),
                         win=win_loss, update='append', name='report')
                # Visualize:
                vm.pattern_visual((disp_c, sparse_disp, mask_c, pattern_mat, image_mat),
                                  vis=vis, win_imgs=win_images, win_cam="Est_Image", win_pattern=win_pattern)
            else:
                print(train_num, end='', flush=True)

        # Draw:
        epoch_average = epoch_loss / len(data_loader)
        vis.line(X=torch.FloatTensor([epoch + 0.5]), Y=torch.FloatTensor([epoch_average]), win=win_loss,
                 update='append', name='epoch')
        print('Average loss for epoch %d: %.2e' % (epoch + 1, epoch_average))

        # Check Parameter nan number
        param_list = list(sparse_network.parameters()) + list(pattern_network.parameters())
        for idx in range(0, len(param_list)):
            param = param_list[idx]
            if torch.isnan(param).any().item():
                print('Found NaN number. Epoch %d, on param[%d].' % (epoch + 1, idx))
                return
            if param.grad is not None and torch.isnan(param.grad).any().item():
                print('Found NaN grad. Epoch %d, on param[%d].' % (epoch + 1, idx))
                return

        # Save
        if epoch % save_period == save_period - 1:
            torch.save(pattern_network.state_dict(), './model_pair/model_pattern' + str(epoch) + '.pt')
            torch.save(sparse_network.state_dict(), './model_pair/model_sparse' + str(epoch) + '.pt')
            print('Save model at epoch %d.' % (epoch + 1))
        torch.save(pattern_network.state_dict(), './model_pattern.pt')
        torch.save(sparse_network.state_dict(), './model_sparse.pt')
    print('Step 3: Finish training.')

    # Step 3: Save the model
    print('Step 4: Finish saving.')


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
