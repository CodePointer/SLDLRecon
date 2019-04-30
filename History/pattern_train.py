import sys
from History.data_set import CameraDataSet
from torch.utils.data import DataLoader
from Module.generator_net import GeneratorNet
# from Module.generator_net_1d import GeneratorNet
from sparse_net import SparseNet
import torch
import math
import numpy as np
import os
import visdom
from History import visual_module_old as vm


def lr_change(epoch):
    epoch = epoch // 1
    return 1 ** epoch


def render_image(pattern, idx_vec, mask_mat):
    ###############
    # parameters: #
    ###############
    p_noise_rad = 0.1
    i_noise_rad = 0.1
    batch_size = pattern.shape[0]
    pattern_channel = pattern.shape[1]
    pro_height = pattern.shape[2]
    pro_width = pattern.shape[3]
    cam_height = 1024
    cam_width = 1280
    #################
    # Render part:  #
    #################
    '''
    Render part:
        Use idx_vec to fill image.
        include pattern noise and image noise.
    '''
    pattern_noise = torch.randn(pattern.shape).cuda() / 3 * p_noise_rad
    # pattern_rearrange = pattern_mat.transpose(1, 0)
    pattern_rearrange = (pattern + pattern_noise).transpose(1, 0)
    pattern_search = pattern_rearrange.reshape(pattern_channel, batch_size * pro_height * pro_width)
    pattern_search = torch.clamp(pattern_search, min=-1, max=1)
    idx_vec_plain = idx_vec.reshape(batch_size * cam_height * cam_width)
    est_img_vec = torch.index_select(input=pattern_search, dim=1, index=idx_vec_plain)
    image_mat = est_img_vec.reshape(3, batch_size, cam_height, cam_width).transpose(1, 0)
    image_noise = torch.randn(image_mat.shape).cuda() / 3 * i_noise_rad
    image_mat = image_mat + image_noise
    image_mat = torch.clamp(image_mat, min=-1, max=1)
    image_mat.masked_fill_(mask_mat == 0, -1)
    return image_mat


def select_prob(sparse_prob, disp_c, mask_c):
    disp_c = torch.clamp(disp_c, min=1/64, max=1.0)
    disp_idx = 63 - torch.round((disp_c - 1/64) * 64)
    disp_idx[mask_c == 0] = 0
    # print(torch.max(disp_idx), torch.min(disp_idx))
    try:
        assert torch.max(disp_idx).item() <= 63 and torch.min(disp_idx).item() >= 0
    except AssertionError:
        print(torch.max(disp_c), torch.min(disp_c))
        print(torch.max(disp_idx), torch.min(disp_idx))
        exit()
    selected_prob = torch.gather(input=sparse_prob, dim=1, index=disp_idx.long())
    return selected_prob


def check_nan_param(network):
    param_list = list(network.parameters())
    for idx in range(0, len(param_list)):
        param = param_list[idx]
        if torch.isnan(param).any().item():
            return False
        if param.grad is not None and torch.isnan(param.grad).any().item():
            return False
    return True


def train_together(root_path, lr_n, start_epoch=1):
    # Step 1: Set data_loader, create net, visual
    batch_size = 4
    down_k = 3
    opt_header = ('mask_mat', 'disp_c', 'mask_c', 'disp_mat', 'idx_vec')
    opts_train = {'header': opt_header, 'stride': 10}
    opts_test = {'header': opt_header, 'stride': 50, 'bias': 25}
    train_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k), down_k=down_k, opts=opts_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k), down_k=down_k, opts=opts_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Step 0: DataLoader size (tr/te): (%d/%d).' % (len(train_loader), len(test_loader)))

    vis_env = 'K' + str(down_k) + '_Network_Gene'
    vis = visdom.Visdom(env=vis_env)
    win_loss = 'training_loss'
    win_images = 'image_set'
    win_pattern = 'pattern'
    learning_rate = math.pow(0.1, lr_n)
    print('learning_rate: %.1e' % learning_rate)

    pattern_network = GeneratorNet(root_path=root_path, batch_size=batch_size)
    sparse_network = SparseNet(root_path=root_path, batch_size=batch_size, down_k=down_k)
    # criterion = torch.nn.SmoothL1Loss()
    pattern_opt = torch.optim.Adam(pattern_network.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    sparse_opt = torch.optim.Adam(sparse_network.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    pattern_schedular = torch.optim.lr_scheduler.LambdaLR(pattern_opt, lr_lambda=lr_change)
    sparse_scheduler = torch.optim.lr_scheduler.LambdaLR(sparse_opt, lr_lambda=lr_change)
    if os.path.exists('./model_sparse.pt'):
        print('    Sparse Model found. Import parameters.')
        sparse_network.load_state_dict(torch.load('./model_sparse.pt'), strict=False)
        sparse_network.train()  # For BN
    if os.path.exists('./model_pattern.pt'):
        print('    Pattern Model found. Import parameters.')
        pattern_network.load_state_dict(torch.load('./model_pattern.pt'), strict=False)
        pattern_network.train()
    print('Step 2: Model loading finished.')

    # Step 3: Training
    report_period = 20
    save_period = 10
    # criterion = criterion.cuda()
    sparse_network = sparse_network.cuda()
    pattern_network = pattern_network.cuda()
    pattern_seed = torch.from_numpy(np.load('random_seed.npy')).float().cuda()
    pattern_seed = (pattern_seed - 0.5) * 2
    pattern_seed = torch.stack([pattern_seed] * batch_size, dim=0).unsqueeze(1)
    for epoch in range(start_epoch - 1, 5000):
        ##############
        # Train part #
        ##############
        running_loss = 0.0
        epoch_loss = 0.0
        pattern_schedular.step()
        param_group = pattern_opt.param_groups[0]
        # print(param_group)
        now_lr = param_group['lr']
        print('learning_rate: %.1e' % now_lr)
        for i, data in enumerate(train_loader, 0):
            # Get data
            mask_mat = data['mask_mat'].cuda()
            disp_c = data['disp_c'].cuda()
            mask_c = data['mask_c'].cuda()
            # shade_mat = data['shade_mat'].cuda()
            idx_vec = data['idx_vec'].cuda()
            pattern_opt.zero_grad()
            sparse_opt.zero_grad()

            # Generate pattern
            # pattern_mat = pattern_network(pattern_seed)  # [N, C=3, Hp, Wp]
            sparse_pattern = pattern_network(pattern_seed)
            # sparse_pattern_1d = pattern_network(pattern_seed)
            # sparse_pattern = torch.stack([sparse_pattern_1d] * 16, dim=2)
            dense_pattern = torch.nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
                                                            align_corners=False)
            pattern_mat = dense_pattern

            # Image Rendering Part
            image_mat = render_image(pattern=pattern_mat, idx_vec=idx_vec, mask_mat=mask_mat)

            # Get Disp
            # sparse_disp = sparse_network((image_mat, pattern_mat))
            sparse_prob = sparse_network((image_mat, pattern_mat))
            selected_prob = select_prob(sparse_prob, disp_c, mask_c)

            # Calculate loss
            cross_entropy = - torch.log(selected_prob.masked_select(mask_c))
            loss_entropy = cross_entropy.mean()
            loss_entropy.backward()

            # Train
            sparse_opt.step()
            pattern_opt.step()

            # Optimize
            loss_add = loss_entropy.item()
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
                report_info = '[%d, %4d/%d]: %.2e' % (epoch + 1, i + 1, len(train_loader), average)
                print(train_num, report_info)
                # Draw:
                vis.line(X=torch.FloatTensor([epoch + i / len(train_loader)]), Y=torch.FloatTensor([average]),
                         win=win_loss, update='append', name='train_report', opts=dict(showlegend=True))
                # Visualize:
                vm.pattern_visual((selected_prob, mask_c, pattern_mat, image_mat, sparse_pattern),
                                  vis=vis, win_imgs=win_images, win_cam="Est_Image", win_pattern=win_pattern)
            else:
                print(train_num, end='', flush=True)
        # Draw:
        epoch_average = epoch_loss / len(train_loader)
        vis.line(X=torch.FloatTensor([epoch + 0.5]), Y=torch.FloatTensor([epoch_average]), win=win_loss,
                 update='append', name='train_epoch', opts=dict(showlegend=True))

        ##############
        # Test part: #
        ##############
        with torch.no_grad():
            test_loss = 0
            for i, data in enumerate(test_loader, 0):
                # Get data
                mask_mat = data['mask_mat'].cuda()
                disp_c = data['disp_c'].cuda()
                mask_c = data['mask_c'].cuda()
                # shade_mat = data['shade_mat'].cuda()
                idx_vec = data['idx_vec'].cuda()

                # Generate pattern
                # pattern_mat = pattern_network(pattern_seed)  # [N, C=3, Hp, Wp]
                sparse_pattern = pattern_network(pattern_seed)
                # sparse_pattern_1d = pattern_network(pattern_seed)
                # sparse_pattern = torch.stack([sparse_pattern_1d] * 16, dim=2)
                dense_pattern = torch.nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
                                                                align_corners=False)
                pattern_mat = dense_pattern

                # Image Rendering Part
                image_mat = render_image(pattern=pattern_mat, idx_vec=idx_vec, mask_mat=mask_mat)

                # Get Disp
                # sparse_disp = sparse_network((image_mat, pattern_mat))
                sparse_prob = sparse_network((image_mat, pattern_mat))
                selected_prob = select_prob(sparse_prob, disp_c, mask_c)

                # Calculate loss
                cross_entropy = - torch.log(selected_prob.masked_select(mask_c))
                loss_entropy = cross_entropy.mean()

                # Optimize
                loss_add = loss_entropy.item()
                test_loss += loss_add
                if np.isnan(loss_add):
                    print('Error: Nan detected at set [%d].' % i)
                    return

                # Visualization and report
                print('.', end='', flush=True)
            test_average = test_loss / len(test_loader)
            vis.line(X=torch.FloatTensor([epoch + 0.5]), Y=torch.FloatTensor([test_average]), win=win_loss,
                     update='append', name='test_epoch', opts=dict(showlegend=True))

        print('Average loss for epoch[%d]: %.2e, %.2e' % (epoch + 1, epoch_average, test_average))

        # Check Parameter nan number
        assert check_nan_param(sparse_network)
        assert check_nan_param(pattern_network)

        # Save
        pattern = pattern_mat[0, :, :, :].detach().cpu().numpy()
        if epoch % save_period == save_period - 1:
            torch.save(pattern_network.state_dict(), './model_pair/model_pattern' + str(epoch) + '.pt')
            torch.save(sparse_network.state_dict(), './model_pair/model_sparse' + str(epoch) + '.pt')
            np.save('./model_pair/res_pattern' + str(epoch) + '.npy', pattern)
            print('Save model at epoch %d.' % (epoch + 1))
        torch.save(pattern_network.state_dict(), './model_pattern.pt')
        torch.save(sparse_network.state_dict(), './model_sparse.pt')
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
