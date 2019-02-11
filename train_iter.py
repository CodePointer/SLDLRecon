import sys
from data_set import CameraDataSet
from torch.utils.data import DataLoader
# from Module.generator_net import GeneratorNet
# from Module.generator_net_dot import GeneratorNet
from Module.generator_net_grey import GeneratorNet
# from Module.sparse_net import SparseNet
from Module.sparse_net_grey import SparseNet
import torch
import torchvision
import math
import numpy as np
import scipy.stats as st
import os
import visdom
import visual_module as vm


def lr_change(epoch):
    epoch = epoch // 50
    return 1 ** epoch


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.0) / kernlen
    x = np.linspace(-nsig-interval/2.0, nsig+interval/2.0, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def render_image(pattern, idx_vec, mask_mat, gkernel):
    ###############
    # parameters: #
    ###############
    p_noise_rad = 0.1
    i_noise_rad = 0.1
    batch_size = pattern.shape[0]
    pattern_channel = pattern.shape[1]
    pro_height = pattern.shape[2] * 8
    pro_width = pattern.shape[3] * 8
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

    pattern_bias = torch.randn(1).item() / 3 * 0.3
    pattern_noise = torch.randn(pattern.shape).cuda() / 3 * p_noise_rad + pattern_bias
    dense_pattern = torch.nn.functional.interpolate(input=pattern + pattern_noise, scale_factor=8, mode='bilinear',
                                                    align_corners=False)
    # pattern_rearrange = pattern_mat.transpose(1, 0)
    pattern_rearrange = dense_pattern.transpose(1, 0)
    pattern_search = pattern_rearrange.reshape(pattern_channel, batch_size * pro_height * pro_width)
    pattern_search = torch.clamp(pattern_search, min=-1, max=1)
    idx_vec_plain = idx_vec.reshape(batch_size * cam_height * cam_width)
    est_img_vec = torch.index_select(input=pattern_search, dim=1, index=idx_vec_plain)
    image_mat = est_img_vec.reshape(pattern_channel, batch_size, cam_height, cam_width).transpose(1, 0)

    # Project Gaussian Blur
    image_mat = torch.nn.functional.conv2d(image_mat, gkernel, padding=7 // 2)

    # Image noise
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


def train_iteration(root_path, lr_n, start_epoch=1):
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
    win_set = {'g_loss': 'Generator Loss',
               'd_loss': 'Discriminator Loss',
               'image': 'Rendered Image',
               'match': 'Match Result',
               'disp': 'Disparity Result',
               'pattern': 'Generated Pattern',
               'pattern_box': 'Pattern Boxplot'}

    g_lr_n, e_lr_n = lr_n
    g_lr = math.pow(0.1, g_lr_n)
    e_lr = math.pow(0.1, e_lr_n)
    gkernel = torch.from_numpy(gkern(7, nsig=2)).unsqueeze(0).unsqueeze(1)  # [1, 1, Hk, Wk]
    gkernel = gkernel.float().cuda()
    print('learning_rate: %.1e/%.1e' % (g_lr, e_lr))

    pattern_network = GeneratorNet(root_path=root_path, batch_size=batch_size)
    sparse_network = SparseNet(root_path=root_path, batch_size=batch_size, down_k=down_k)
    criterion = torch.nn.SmoothL1Loss()
    pattern_opt = torch.optim.Adam(pattern_network.parameters(), lr=g_lr, betas=(0.9, 0.999))
    sparse_opt = torch.optim.Adam(sparse_network.parameters(), lr=e_lr, betas=(0.9, 0.999))
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    pattern_schedular = torch.optim.lr_scheduler.LambdaLR(pattern_opt, lr_lambda=lr_change)
    sparse_schedular = torch.optim.lr_scheduler.LambdaLR(sparse_opt, lr_lambda=lr_change)
    if os.path.exists('./model_sparse.pt'):
        print('    Sparse Model found. Import parameters.')
        sparse_network.load_state_dict(torch.load('./model_sparse.pt'), strict=False)
        sparse_network.train()  # For BN
    if os.path.exists('./model_pattern.pt'):
        print('    Pattern Model found. Import parameters.')
        pattern_network.load_state_dict(torch.load('./model_pattern.pt'), strict=False)
        pattern_network.train()
    print('Step 2: Model loading finished.')

    assert check_nan_param(pattern_network)

    # Step 3: Training
    report_period = 10
    save_period = 10
    # criterion = criterion.cuda()
    sparse_network = sparse_network.cuda()
    pattern_network = pattern_network.cuda()
    pattern_seed = torch.from_numpy(np.load('random_seed.npy')).float().cuda()
    # pattern_seed = torch.ones(64).cuda()
    # pattern_seed = torch.ones(1).cuda()
    pattern_seed = (pattern_seed - 0.5) * 2
    pattern_seed = torch.stack([pattern_seed] * batch_size, dim=0).unsqueeze(1)
    for epoch in range(start_epoch - 1, 5000):
        ##############
        # Train part #
        ##############
        g_loss_running = 0.0
        d_loss_running = 0.0
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        epoch_loss = 0.0
        g_flag = True
        pattern_schedular.step()
        sparse_schedular.step()
        pat_param_group = pattern_opt.param_groups[0]
        spa_param_group = sparse_opt.param_groups[0]
        # print(param_group)
        pat_now_lr = pat_param_group['lr']
        spa_now_lr = spa_param_group['lr']
        print('learning_rate: %.1e/%.1e' % (pat_now_lr, spa_now_lr))
        for i, data in enumerate(train_loader, 0):
            # Get data
            mask_mat = data['mask_mat'].cuda()
            disp_c = data['disp_c'].cuda()
            mask_c = data['mask_c'].cuda()
            # shade_mat = data['shade_mat'].cuda()
            idx_vec = data['idx_vec'].cuda()

            ###################
            # Generator Train #
            ###################
            pattern_opt.zero_grad()
            sparse_pattern = pattern_network(pattern_seed)
            dense_pattern = torch.nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
                                                            align_corners=False)
            pattern_mat = dense_pattern
            image_mat = render_image(pattern=sparse_pattern, idx_vec=idx_vec, mask_mat=mask_mat, gkernel=gkernel)

            # sparse_prob_pre = sparse_network((image_mat, pattern_mat, g_flag))
            # selected_prob = select_prob(sparse_prob_pre, disp_c, mask_c)
            # Calculate loss
            # cross_entropy = - torch.log(selected_prob.masked_select(mask_c))
            # loss_entropy = cross_entropy.mean()
            # loss_entropy.backward()
            # g_loss_add = loss_entropy.item()
            sparse_prob_pre = None
            sparse_disp = sparse_network((image_mat, pattern_mat, not g_flag))
            loss_coarse = criterion(sparse_disp.masked_select(mask_c), disp_c.masked_select(mask_c))
            loss_coarse.backward()
            g_loss_add = loss_coarse.item()
            g_loss_running += g_loss_add
            pattern_opt.step()
            assert not np.isnan(g_loss_add)

            #######################
            # Discriminator Train #
            #######################
            sparse_opt.zero_grad()
            sparse_disp = sparse_network((image_mat.detach(), pattern_mat.detach(), not g_flag))
            loss_coarse = criterion(sparse_disp.masked_select(mask_c), disp_c.masked_select(mask_c))
            loss_coarse.backward()
            d_loss_add = loss_coarse.item()
            d_loss_running += d_loss_add
            sparse_opt.step()
            assert not np.isnan(d_loss_add)

            # Visualization and report
            print('.', end='', flush=True)
            g_loss_epoch += g_loss_add
            d_loss_epoch += d_loss_add
            if i % report_period == report_period - 1:
                report_info = vm.iter_visual_report(vis=vis, win_set=win_set, input_set=(
                    (i, epoch, len(train_loader), report_period),
                    (g_loss_running, d_loss_running),
                    (pattern_mat[0, :, :, :], sparse_pattern[0, :, :, :]),
                    (mask_c, image_mat, sparse_disp, sparse_prob_pre, disp_c)))
                g_loss_running = 0
                d_loss_running = 0
                print(report_info)
                # print('')

        # Epoch visualization:
        report_info = vm.iter_visual_epoch(vis=vis, win_set=win_set, input_set=(
            (epoch, len(train_loader)),
            (g_loss_epoch, d_loss_epoch)))
        print(report_info)

        ##############
        # Test part: #
        ##############
        with torch.no_grad():
            g_loss_test = 0
            d_loss_test = 0
            for i, data in enumerate(test_loader, 0):
                # Get data
                mask_mat = data['mask_mat'].cuda()
                disp_c = data['disp_c'].cuda()
                mask_c = data['mask_c'].cuda()
                # shade_mat = data['shade_mat'].cuda()
                idx_vec = data['idx_vec'].cuda()

                ###################
                # Generator Test  #
                ###################
                sparse_pattern = pattern_network(pattern_seed)
                dense_pattern = torch.nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
                                                                align_corners=False)
                pattern_mat = dense_pattern
                image_mat = render_image(pattern=sparse_pattern, idx_vec=idx_vec, mask_mat=mask_mat, gkernel=gkernel)
                # sparse_prob_pre = sparse_network((image_mat, pattern_mat, g_flag))
                # selected_prob = select_prob(sparse_prob_pre, disp_c, mask_c)
                # # Calculate loss
                # cross_entropy = - torch.log(selected_prob.masked_select(mask_c))
                # loss_entropy = cross_entropy.mean()
                # g_loss_add = loss_entropy.item()
                # g_loss_test += g_loss_add
                # assert not np.isnan(g_loss_add)

                #######################
                # Discriminator Test  #
                #######################
                sparse_disp = sparse_network((image_mat, pattern_mat, not g_flag))
                loss_coarse = criterion(sparse_disp.masked_select(mask_c), disp_c.masked_select(mask_c))
                d_loss_add = loss_coarse.item()
                d_loss_test += d_loss_add
                assert not np.isnan(d_loss_add)
                # Visualization and report
                print('.', end='', flush=True)

            print('Test finished.')
            report_info = vm.iter_visual_test(vis=vis, win_set=win_set, input_set=(
                (epoch, len(test_loader)),
                (d_loss_test, d_loss_test)))
            print(report_info)

        # Check Parameter nan number
        try:
            assert check_nan_param(sparse_network)
            assert check_nan_param(pattern_network)
        except AssertionError as inst:
            torch.save(pattern_network.state_dict(), 'model_pattern_error.pt')
            torch.save(sparse_network.state_dict(), 'model_sparse_error.pt')
            print(inst)
            raise

        # Save
        pattern = pattern_mat[0, :, :, :].detach().cpu().numpy()
        if epoch % save_period == save_period - 1:
            torch.save(pattern_network.state_dict(), './model_pair/model_pattern' + str(epoch) + '.pt')
            torch.save(sparse_network.state_dict(), './model_pair/model_sparse' + str(epoch) + '.pt')
            print('Save model at epoch %d.' % (epoch + 1))
        torch.save(pattern_network.state_dict(), './model_pattern.pt')
        torch.save(sparse_network.state_dict(), './model_sparse.pt')
        np.save('./model_pair/res_pattern' + str(epoch) + '.npy', pattern)
        np.save('./res_pattern.npy', pattern)

    print('Step 3: Finish training.')


def main(argv):
    # Input parameters
    g_lr_n = int(argv[1])
    e_lr_n = int(argv[2])

    # Get start epoch num
    start_epoch = 1
    if len(argv) >= 4:
        start_epoch = int(argv[3])

    train_iteration(root_path='./SLDataSet/20181204/', start_epoch=start_epoch, lr_n=(g_lr_n, e_lr_n))


if __name__ == '__main__':
    assert len(sys.argv) >= 3
    main(sys.argv)
