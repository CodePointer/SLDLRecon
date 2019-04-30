import sys
import configparser
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import visdom
import History.visual_module as vm
from Module.depth_net import DepthNet
from Module.rigid_net import RigidNet
from Module.flow_estimator import FlowEstimator
from History.flow_data_set import FlowDataSet, SampleSet


def lr_change(epoch):
    epoch = epoch // config.getint('NetworkPara', 'lr_period')
    return config.getfloat('NetworkPara', 'lr_base') ** epoch


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
    disp_range = [float(x) for x in config.get('DataLoader', 'disp_range').split(',')]
    alpha_range = [float(x) for x in config.get('DataLoader', 'alpha_range').split(',')]
    opts_train = {'header': config.get('DataLoader', 'opt_header').split(','),
                  'stride': config.getint('DataLoader', 'train_stride'),
                  'bias': config.getint('DataLoader', 'train_bias')}
    train_dataset = FlowDataSet(root_dir=config.get('FilePath', 'root_path'),
                                list_name=config.get('DataLoader', 'train_list'),
                                jump_k=config.getint('Paras', 'jump_k'),
                                opts=opts_train)
    train_loader = DataLoader(train_dataset, batch_size=config.getint('Paras', 'batch_size'),
                              shuffle=True, num_workers=0)
    opts_test = {'header': config.get('DataLoader', 'opt_header').split(','),
                 'stride': config.getint('DataLoader', 'test_stride'),
                 'bias': config.getint('DataLoader', 'test_bias')}
    test_dataset = FlowDataSet(root_dir=config.get('FilePath', 'root_path'),
                               list_name=config.get('DataLoader', 'test_list'),
                               jump_k=config.getint('Paras', 'jump_k'),
                               opts=opts_test)
    test_loader = DataLoader(test_dataset, batch_size=config.getint('Paras', 'batch_size'),
                             shuffle=True, num_workers=0)
    vis_env = config.get('Paras', 'vis_env')
    vis = visdom.Visdom(env=vis_env)
    print('Step 0: DataSet initialize finished.')
    print('    DataLoader size (tr/te): (%d/%d).' % (len(train_loader), len(test_loader)))

    # Step 1: Create network model.
    # Loss function
    rigid_loss = torch.nn.L1Loss()
    # Network
    flow_estimator = FlowEstimator(root_dir=config.get('FilePath', 'root_path'), disp_range=disp_range)
    depth_net = DepthNet(alpha_range=alpha_range)
    rigid_net = RigidNet()
    flag_set = [False, False]
    if os.path.exists(config.get('FilePath', 'depth_model') + '.pt'):
        flag_set[0] = True
        depth_net.load_state_dict(torch.load(config.get('FilePath', 'depth_model') + '.pt'), strict=False)
        depth_net.train()
        assert check_nan_param(depth_net)
    if os.path.exists(config.get('FilePath', 'rigid_model') + '.pt'):
        flag_set[1] = True
        rigid_net.load_state_dict(torch.load(config.get('FilePath', 'rigid_model') + '.pt'), strict=False)
        rigid_net.train()  # For BN
        assert check_nan_param(rigid_net)
    if cuda:
        flow_estimator.to_cuda()
        depth_net = depth_net.cuda()
        rigid_net = rigid_net.cuda()
        rigid_loss = rigid_loss.cuda()
    print('Step 1: Network finished. Load model: ', flag_set)

    # Step 2: Optimizers
    g_lr = config.getfloat('NetworkPara', 'g_lr')
    d_lr = config.getfloat('NetworkPara', 'd_lr')
    optimizer_g = torch.optim.RMSprop(params=depth_net.parameters(), lr=g_lr)
    optimizer_d = torch.optim.RMSprop(params=rigid_net.parameters(), lr=d_lr)
    schedular_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lr_change)
    schedular_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lr_change)
    print('Step 2: Optimizers setting finished.')

    # -----------
    # Training
    # -----------

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    report_period = config.getint('Paras', 'report_period')
    save_period = config.getint('Paras', 'save_period')
    batch_done = 0
    for epoch in range(config.getint('Paras', 'start_epoch'), config.getint('Paras', 'total_epoch')):
        # g_loss_add, d_loss_add = 0, 0
        g_loss_running = 0.0
        d_loss_running = 0.0
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        schedular_g.step()
        schedular_d.step()
        for i, data in enumerate(train_loader, 0):
            # 1. Get data
            data = SampleSet(data)
            if cuda:
                data.to_cuda()
            else:
                data.to_cpu()
            idx_vec = data['idx']
            mask_cam = data['mask_cam']
            disp_cam = data['disp_cam']
            disp_cam_t = data['disp_cam_t']
            cor_xc = data['cor_xc']
            cor_xc_t = data['cor_xc_t']
            cor_yc = data['cor_yc']
            cor_yc_t = data['cor_yc_t']
            mask_pro = data['mask_pro']
            flow_mat, mask_flow = flow_estimator.get_flow_value(disp_cam, mask_cam, cor_xc_t, cor_yc_t, mask_pro,
                                                                cor_xc, cor_yc)
            # show_flow_raw = torch.clamp(flow_mat[:, 0, :, :], -2.0, 2.0).unsqueeze(1)
            # show_flow_red = show_flow_raw.clone() / 2
            # show_flow_red[show_flow_raw <= 0] = 0
            # show_flow_blue = show_flow_raw.clone() / 2
            # show_flow_blue[show_flow_raw >= 0] = 0
            # show_flow_green = torch.zeros_like(show_flow_red)
            # show_flow_x = torch.cat((show_flow_red, show_flow_green, show_flow_blue), dim=1)
            # show_flow_x[mask_flow == 0] = 0
            # vis.boxplot(X=flow_mat[:, 0, :, :].unsqueeze(1).masked_select(mask_flow.byte()), win='test_box')
            # vis.images(show_flow_x, nrow=1, padding=2, win='test')

            op_center = flow_estimator.get_op_center(flow_mat, mask_flow)
            flow_estimator.set_cam_info(op_center)

            # cos_alpha, sin_alpha = flow_estimator.disp2alpha(disp_cam, mask_cam)
            # disp_mat = flow_estimator.alpha2disp(cos_alpha, sin_alpha, mask_flow)
            # sin_alpha, cos_alpha = flow_estimator.disp2alpha(disp_mat, mask_cam)
            # disp_mat = flow_estimator.alpha2disp(cos_alpha, sin_alpha, mask_flow)
            # sin_alpha, cos_alpha = flow_estimator.disp2alpha(disp_mat, mask_cam)
            # disp_mat = flow_estimator.alpha2disp(cos_alpha, sin_alpha, mask_flow)
            # compare_mat = torch.cat((disp_cam, disp_mat), dim=3)
            # show_alpha = torch.cat((sin_alpha, cos_alpha), dim=3)
            # vis.images(compare_mat, nrow=1, padding=2, win='alpha')
            # return

            # Adversarial ground truths
            # valid = Tensor(idx_vec.shape[0], 1).fill_(0.0)
            # fake = Tensor(idx_vec.shape[0], 1).fill_(1.0)

            # Train Easy Version:
            optimizer_g.zero_grad()
            alpha_mat = depth_net(flow_mat)
            disp_fake, disp_fake_t, mask_flow_t = flow_estimator.alpha2disps(alpha_mat, flow_mat, mask_flow)
            g_loss = rigid_loss(disp_fake.masked_select(mask_flow.byte()), disp_cam.masked_select(mask_flow.byte()))
            g_loss.backward()
            optimizer_g.step()
            g_loss_running += g_loss.item()
            g_loss_epoch += g_loss.item()

            # Train Discriminator
            # alpha_mat = depth_net(flow_mat).detach()
            # disp_mat, disp_mat_t, mask_flow_t = flow_estimator.alpha2disps(alpha_mat, flow_mat, mask_flow)
            # disp_cam[mask_flow == 0] = 0
            # disp_cam_t[mask_flow_t == 0] = 0
            # disp_cam_comb = torch.cat((disp_cam, disp_cam_t), dim=1)
            # disp_mat_comb = torch.cat((disp_mat, disp_mat_t), dim=1)
            # d_loss = -torch.mean(rigid_net(disp_cam_comb)) + torch.mean(rigid_net(disp_mat_comb))
            # d_loss.backward()
            # optimizer_d.step()
            # d_loss_running += d_loss.item()
            # d_loss_add += 1

            # Clip weights of distriminator
            # for p in rigid_net.parameters():
            #     p.data.clamp_(-config.getfloat('NetworkPara', 'clip_val'), config.getfloat('NetworkPara', 'clip_val'))

            # Report: easy version
            print('.', end='', flush=True)
            if (i + 1) % report_period == 0:
                report_info = vm.iter_visual_report(vis=vis, win_set=config['WinSet'], input_set=(
                    (i, epoch, len(train_loader), report_period),
                    (g_loss_running, g_loss_running),
                    (disp_cam, None, disp_fake, None, mask_flow, None),
                    disp_range))
                g_loss_running = 0
                print(report_info)
                # Save
                np.save('show_output/disp_real%d.npy' % (i + 1), disp_cam.detach().cpu().numpy())
                np.save('show_output/disp_fake%d.npy' % (i + 1), disp_fake.detach().cpu().numpy())
                np.save('show_output/mask_flow%d.npy' % (i + 1), mask_flow.detach().cpu().numpy())

            # Train the generator every n_critic iterations
            # print('.', end='', flush=True)
            # if (i + 1) % config.getint('NetworkPara', 'n_critic') == 0:
                # optimizer_g.zero_grad()
                # alpha_mat = depth_net(flow_mat)
                # alpha -> depth_mat 1, 2
                # disp_mat, disp_mat_t, mask_flow_t = flow_estimator.alpha2disps(alpha_mat, flow_mat, mask_flow)
                # disp_mat_comb = torch.cat((disp_mat, disp_mat_t), dim=1)
                # g_loss = -torch.mean(rigid_net(disp_mat_comb))
                # g_loss.backward()
                # optimizer_g.step()
                # g_loss_running += g_loss.item()
                # g_loss_add += 1

                # report_info = vm.iter_visual_report(vis=vis, win_set=config['WinSet'], input_set=(
                #     (i, epoch, len(train_loader), report_period),
                #     (g_loss_running, d_loss_running),
                #     (disp_cam, disp_cam_t, disp_mat, disp_mat_t, mask_flow, mask_flow_t),
                #     disp_range))
                # g_loss_running = 0
                # d_loss_running = 0
                # print(report_info)

                try:
                    assert check_nan_param(depth_net)
                    assert check_nan_param(rigid_net)
                except AssertionError as inst:
                    torch.save(depth_net.state_dict(), config.get('FilePath', 'depth_model') + '_error.pt')
                    torch.save(rigid_net.state_dict(), config.get('FilePath', 'rigid_model') + '_error.pt')
                    print(inst)
                    raise

        # Epoch visualization:
        g_now_lr = optimizer_g.param_groups[0]['lr']
        d_now_lr = optimizer_d.param_groups[0]['lr']
        report_info = vm.iter_visual_epoch(vis=vis, win_set=config['WinSet'], input_set=(
            (epoch, len(train_loader)),
            (g_loss_epoch, g_loss_epoch),
            (g_now_lr, d_now_lr)))
        print(report_info)

        # Check Parameter nan number
        try:
            assert check_nan_param(depth_net)
            assert check_nan_param(rigid_net)
        except AssertionError as inst:
            torch.save(depth_net.state_dict(), config.get('FilePath', 'depth_model') + '_error.pt')
            torch.save(rigid_net.state_dict(), config.get('FilePath', 'rigid_model') + '_error.pt')
            print(inst)
            raise

        # Save
        if epoch % save_period == save_period - 1:
            torch.save(depth_net.state_dict(),
                       ''.join([config.get('FilePath', 'save_model'),
                                config.get('FilePath', 'depth_model'),
                                str(epoch),
                                '.pt']))
            torch.save(rigid_net.state_dict(),
                       ''.join([config.get('FilePath', 'save_model'),
                                config.get('FilePath', 'rigid_model'),
                                str(epoch),
                                '.pt']))
            print('    Save model at epoch %d.' % epoch)
        torch.save(depth_net.state_dict(), config.get('FilePath', 'depth_model') + '.pt')
        torch.save(rigid_net.state_dict(), config.get('FilePath', 'rigid_model') + '.pt')

        # ------------
        # Test part:
        # ------------
        with torch.no_grad():
            g_loss_test = 0
            for i, data in enumerate(test_loader, 0):
                # 1. Get data
                data = SampleSet(data)
                if cuda:
                    data.to_cuda()
                else:
                    data.to_cpu()
                mask_cam = data['mask_cam']
                disp_cam = data['disp_cam']
                cor_xc = data['cor_xc']
                cor_xc_t = data['cor_xc_t']
                cor_yc = data['cor_yc']
                cor_yc_t = data['cor_yc_t']
                mask_pro = data['mask_pro']
                flow_mat, mask_flow = flow_estimator.get_flow_value(disp_cam, mask_cam, cor_xc_t, cor_yc_t, mask_pro,
                                                                    cor_xc, cor_yc)
                op_center = flow_estimator.get_op_center(flow_mat, mask_flow)
                flow_estimator.set_cam_info(op_center)

                # Test Easy Version:
                alpha_mat = depth_net(flow_mat)
                disp_fake, disp_fake_t, mask_flow_t = flow_estimator.alpha2disps(alpha_mat, flow_mat, mask_flow)
                g_loss = rigid_loss(disp_fake.masked_select(mask_flow.byte()), disp_cam.masked_select(mask_flow.byte()))
                g_loss_test += g_loss.item()
                print('.', end='', flush=True)
                # Save
                np.save('show_output/disp_real_test%d.npy' % (i + 1), disp_cam.cpu().numpy())
                np.save('show_output/disp_fake_test%d.npy' % (i + 1), disp_fake.cpu().numpy())
                np.save('show_output/mask_flow_test%d.npy' % (i + 1), mask_flow.cpu().numpy())

            report_info = vm.iter_visual_test(vis=vis, win_set=config['WinSet'], input_set=(
                (epoch, len(test_loader)),
                (g_loss_test, g_loss_test)))
            print(report_info)

    print('Step 3: Finish training.')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    config_path = sys.argv[1]

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)

    cuda = True if torch.cuda.is_available() else False

    spatial_train()
