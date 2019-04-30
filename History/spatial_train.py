import sys
import configparser
from History.data_set import CameraDataSet
from torch.utils.data import DataLoader
# from Module.generator_net import GeneratorNet
# from Module.generator_net_dot import GeneratorNet
# from Module.generator_net_grey import GeneratorNet
from Module.generator_net_grey import GeneratorNet
# from Module.sparse_net import SparseNet
from Module.sparse_net_grey import SparseNet
import torch
import numpy as np
import scipy.stats as st
import os
import visdom
from History import visual_module_old as vm


def lr_change(epoch):
    epoch = epoch // config.getint('NetworkPara', 'lr_period')
    return config.getfloat('NetworkPara', 'lr_base') ** epoch


def gkern(kernlen=21, nsig=3.0):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.0) / kernlen
    x = np.linspace(-nsig-interval/2.0, nsig+interval/2.0, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def render_image(pattern, idx_vec, mask_mat, gkernel=None, add_noise=True):
    ###############
    # parameters: #
    ###############
    p_noise_rad = config.getfloat('RenderPara', 'p_range')
    i_noise_rad = config.getfloat('RenderPara', 'i_range')
    p_bias_rad = config.getfloat('RenderPara', 'bias_range')
    batch_size = pattern.shape[0]
    pattern_channel = pattern.shape[1]
    # pro_height = pattern.shape[2] * 8
    # pro_width = pattern.shape[3] * 8
    pro_height = pattern.shape[2]
    pro_width = pattern.shape[3]
    cam_height = config.getint('Global', 'cam_height')
    cam_width = config.getint('Global', 'cam_width')

    #################
    # Render part:  #
    #################
    '''
    Render part:
        Use idx_vec to fill image.
        include pattern noise and image noise.
    '''
    if add_noise:
        pattern_bias = torch.randn(1).item() * p_bias_rad
        pattern_noise = torch.randn(pattern.shape).cuda() * p_noise_rad + pattern_bias
    else:
        pattern_noise = torch.zeros(pattern.shape).cuda()
    # dense_pattern = torch.nn.functional.interpolate(input=pattern + pattern_noise, scale_factor=8, mode='bilinear',
    #                                                 align_corners=False)
    dense_pattern = pattern_noise + pattern
    pattern_rearrange = dense_pattern.transpose(1, 0)
    pattern_search = pattern_rearrange.reshape(pattern_channel, batch_size * pro_height * pro_width)
    pattern_search = torch.clamp(pattern_search, min=-1, max=1)
    idx_vec_plain = idx_vec.reshape(batch_size * cam_height * cam_width)
    est_img_vec = torch.index_select(input=pattern_search, dim=1, index=idx_vec_plain)
    image_mat = est_img_vec.reshape(pattern_channel, batch_size, cam_height, cam_width).transpose(1, 0)

    if add_noise:
        pad_size = config.getint('RenderPara', 'gkern_size') // 2
        image_mat = torch.nn.functional.conv2d(image_mat, gkernel, padding=pad_size)
        image_noise = torch.randn(image_mat.shape).cuda() * i_noise_rad
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


def spatial_train():
    # Step 0: Set data_loader, visual
    opts_train = {'header': config.get('DataLoader', 'opt_header').split(','),
                  'stride': config.getint('DataLoader', 'train_stride'),
                  'bias': config.getint('DataLoader', 'train_bias')}
    opts_test = {'header': config.get('DataLoader', 'opt_header').split(','),
                 'stride': config.getint('DataLoader', 'test_stride'),
                 'bias': config.getint('DataLoader', 'test_bias')}
    disp_range = [int(x) for x in config.get('DataLoader', 'disp_range').split(',')]
    train_dataset = CameraDataSet(root_dir=config.get('FilePath', 'root_path'),
                                  list_name=config.get('DataLoader', 'train_list'),
                                  down_k=config.getint('Paras', 'down_k'),
                                  disp_range=disp_range,
                                  opts=opts_train)
    train_loader = DataLoader(train_dataset, batch_size=config.getint('Paras', 'batch_size'),
                              shuffle=True, num_workers=0)
    test_dataset = CameraDataSet(root_dir=config.get('FilePath', 'root_path'),
                                 list_name=config.get('DataLoader', 'test_list'),
                                 down_k=config.getint('Paras', 'down_k'),
                                 disp_range=disp_range,
                                 opts=opts_test)
    test_loader = DataLoader(test_dataset, batch_size=config.getint('Paras', 'batch_size'),
                             shuffle=True, num_workers=0)
    vis_env = config.get('Paras', 'vis_env')
    vis = visdom.Visdom(env=vis_env)

    gkernel_numpy = gkern(kernlen=config.getint('RenderPara', 'gkern_size'),
                          nsig=config.getfloat('RenderPara', 'gkern_sigma'))
    gkernel = torch.from_numpy(gkernel_numpy).unsqueeze(0).unsqueeze(1)  # [1, 1, Hk, Wk]
    gkernel = gkernel.float().cuda()
    print('Step 0: DataSet initialize finished.')
    print('    DataLoader size (tr/te): (%d/%d).' % (len(train_loader), len(test_loader)))

    # Step 1: Create network, set optim.
    pattern_network = GeneratorNet(para_sec=config['Global'],
                                   batch_size=config.getint('Paras', 'batch_size'))
    sparse_network = SparseNet(root_path=config.get('FilePath', 'root_path'),
                               batch_size=config.getint('Paras', 'batch_size'),
                               down_k=config.getint('Paras', 'down_k'))
    criterion = torch.nn.SmoothL1Loss()
    g_lr = config.getfloat('NetworkPara', 'g_lr')
    e_lr = config.getfloat('NetworkPara', 'e_lr')
    pattern_opt = torch.optim.Adam([
        {'params': pattern_network.parameters(), 'lr': g_lr},
        {'params': sparse_network.dn_convs.parameters()},
        {'params': sparse_network.dn_self_convs.parameters()},
        {'params': sparse_network.res_convs.parameters()},
        {'params': sparse_network.self_out.parameters()}],
        lr=e_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sparse_opt = torch.optim.Adam([
        {'params': sparse_network.volume_convs.parameters()},
        {'params': sparse_network.volume_out.parameters()}],
        lr=e_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    print('Step 1: Network & optimizer initialize finished.')

    # Step 2: Model loading
    pattern_schedular = torch.optim.lr_scheduler.LambdaLR(pattern_opt, lr_lambda=lr_change)
    sparse_schedular = torch.optim.lr_scheduler.LambdaLR(sparse_opt, lr_lambda=lr_change)
    flag_set = [False, False]
    if os.path.exists(config.get('FilePath', 'pattern_model') + '.pt'):
        flag_set[0] = True
        pattern_network.load_state_dict(torch.load(config.get('FilePath', 'pattern_model') + '.pt'), strict=False)
        pattern_network.train()
        assert check_nan_param(pattern_network)
    if os.path.exists(config.get('FilePath', 'sparse_model') + '.pt'):
        flag_set[1] = True
        sparse_network.load_state_dict(torch.load(config.get('FilePath', 'sparse_model') + '.pt'), strict=False)
        sparse_network.train()  # For BN
        assert check_nan_param(sparse_network)
    print('Step 2: Model loading finished.')
    print('    Load model(P/S): ', flag_set)

    # Step 3: Training
    sparse_network = sparse_network.cuda()
    pattern_network = pattern_network.cuda()
    pattern_grid = torch.from_numpy(np.load('pattern_grid.npy')).float().cuda()
    pattern_grid = (pattern_grid - 0.5) * 2
    pattern_grid = torch.stack([pattern_grid] * config.getint('Paras', 'batch_size'), dim=0).unsqueeze(1)
    # pattern_seed = torch.from_numpy(np.load('random_seed.npy')).float().cuda()
    # pattern_seed = (pattern_seed - 0.5) * 2
    # pattern_seed = torch.stack([pattern_seed] * config.getint('Paras', 'batch_size'), dim=0).unsqueeze(1)
    report_period = config.getint('Paras', 'report_period')
    save_period = config.getint('Paras', 'save_period')
    for epoch in range(config.getint('Paras', 'start_epoch'), config.getint('Paras', 'total_epoch')):
        ##############
        # Train part #
        ##############
        g_loss_running = 0.0
        d_loss_running = 0.0
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
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
            idx_vec = data['idx_vec'].cuda()

            ###################
            # Generator Train #
            ###################
            pattern_opt.zero_grad()
            grid_image = render_image(pattern=pattern_grid, idx_vec=idx_vec, mask_mat=mask_mat, add_noise=False)
            sparse_pattern = pattern_network(grid_image)
            # dense_pattern = torch.nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
            #                                                 align_corners=False)
            # pattern_mat = dense_pattern
            pattern_mat = sparse_pattern
            image_mat = render_image(pattern=sparse_pattern, idx_vec=idx_vec, mask_mat=mask_mat, gkernel=gkernel)

            sparse_prob_pre = sparse_network((image_mat, pattern_mat, g_flag))
            selected_prob = select_prob(sparse_prob_pre, disp_c, mask_c)
            # Calculate loss
            cross_entropy = - torch.log(selected_prob.masked_select(mask_c))
            loss_entropy = cross_entropy.mean()
            loss_entropy.backward()
            g_loss_add = loss_entropy.item()
            # sparse_prob_pre = None
            # sparse_disp = sparse_network((image_mat, pattern_mat, not g_flag))
            # loss_coarse = criterion(sparse_disp.masked_select(mask_c), disp_c.masked_select(mask_c))
            # loss_coarse.backward()
            # g_loss_add = loss_coarse.item()
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
                report_info = vm.iter_visual_report(vis=vis, win_set=config['WinSet'], input_set=(
                    (i, epoch, len(train_loader), report_period),
                    (g_loss_running, d_loss_running),
                    (pattern_mat[0, :, :, :], sparse_pattern[0, :, :, :]),
                    (mask_c, grid_image, image_mat, sparse_disp, sparse_prob_pre, disp_c)))
                g_loss_running = 0
                d_loss_running = 0
                print(report_info)
                # print('')

        # Epoch visualization:
        report_info = vm.iter_visual_epoch(vis=vis, win_set=config['WinSet'], input_set=(
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
                grid_image = render_image(pattern=pattern_grid, idx_vec=idx_vec, mask_mat=mask_mat, add_noise=False)
                sparse_pattern = pattern_network(grid_image)
                # dense_pattern = torch.nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
                #                                                 align_corners=False)
                # pattern_mat = dense_pattern
                pattern_mat = sparse_pattern
                image_mat = render_image(pattern=sparse_pattern, idx_vec=idx_vec, mask_mat=mask_mat, gkernel=gkernel)
                sparse_prob_pre = sparse_network((image_mat, pattern_mat, g_flag))
                selected_prob = select_prob(sparse_prob_pre, disp_c, mask_c)
                # Calculate loss
                cross_entropy = - torch.log(selected_prob.masked_select(mask_c))
                loss_entropy = cross_entropy.mean()
                g_loss_add = loss_entropy.item()
                g_loss_test += g_loss_add
                assert not np.isnan(g_loss_add)

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
            report_info = vm.iter_visual_test(vis=vis, win_set=config['WinSet'], input_set=(
                (epoch, len(test_loader)),
                (g_loss_test, d_loss_test),
                (mask_c, grid_image, image_mat, sparse_disp, disp_c)))
            print(report_info)

        # Check Parameter nan number
        try:
            assert check_nan_param(sparse_network)
            assert check_nan_param(pattern_network)
        except AssertionError as inst:
            torch.save(pattern_network.state_dict(), config.get('FilePath', 'pattern_model') + '_error.pt')
            torch.save(sparse_network.state_dict(), config.get('FilePath', 'sparse_model') + '_error.pt')
            print(inst)
            raise

        # Save
        pattern = pattern_mat[0, :, :, :].detach().cpu().numpy()
        if epoch % save_period == save_period - 1:
            torch.save(pattern_network.state_dict(),
                       ''.join([config.get('FilePath', 'save_model'),
                                config.get('FilePath', 'pattern_model'),
                                str(epoch),
                                '.pt']))
            torch.save(sparse_network.state_dict(),
                       ''.join([config.get('FilePath', 'save_model'),
                                config.get('FilePath', 'pattern_model'),
                                str(epoch),
                                '.pt']))
            print('    Save model at epoch %d.' % epoch)
        torch.save(pattern_network.state_dict(), config.get('FilePath', 'pattern_model') + '.pt')
        torch.save(sparse_network.state_dict(), config.get('FilePath', 'sparse_model') + '.pt')
        np.save(''.join([config.get('FilePath', 'save_model'),
                         config.get('FilePath', 'pattern_val'),
                         str(epoch), '.pt']), pattern)
    print('Step 3: Finish training.')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    config_path = sys.argv[1]

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)

    spatial_train()
