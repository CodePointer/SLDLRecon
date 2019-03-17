import torch
import visdom


def volume_visual(gt_v, res_v, mask, vis, win_imgs, win_fig, nrow=4):
    # Detach & cpu
    # gt_v = gt_v.detach().cpu()
    # res_v = res_v.detach().cpu()
    # mask = mask.detach().cpu()

    # Input size
    vol_shape = gt_v.shape  # [N, D, H, W]
    N = vol_shape[0]
    D = vol_shape[1]
    H = vol_shape[2]
    W = vol_shape[3]

    # Make disp for 64
    disp_range = (torch.Tensor(range(D, 0, -1)) - 1) / D  # [D]
    # disp_vol = torch.stack([disp_range] * N, 0)  # [N, D]
    # disp_vol = torch.stack([disp_vol] * H, 2)
    # disp_vol = torch.stack([disp_vol] * W, 3)  # [N, D, H, W]

    # Calculate disp for N set
    # gt_disp_set = torch.sum(input=gt_v * disp_vol, dim=1, keepdim=True)  # [N, 1, H, W]
    # res_disp_set = torch.sum(input=res_v * disp_vol, dim=1, keepdim=True)  # [N, 1, H, W]
    # res_disp_set = res_disp_set * mask.float()
    # show_disp_set = torch.cat((gt_disp_set, res_disp_set), dim=3)
    # show_disp_set = torch.nn.functional.interpolate(input=show_disp_set, scale_factor=4.0, mode='nearest')
    # vis.images(show_disp_set, nrow=nrow, padding=2, win=win_imgs)

    # Choose one part as plot show
    opt = {'width': 512, 'height': 256}
    for idx in range(0, N, 2):
        mask_part = mask[idx, :, :, :]  # [1, H, W]
        idx_list = mask_part.nonzero()  # [num, 3]
        pt_num = idx_list.shape[0]
        mid_num = int(pt_num / 2)
        gt_vec = gt_v[idx, :, idx_list[mid_num, 1], idx_list[mid_num, 2]]
        res_vec = res_v[idx, :, idx_list[mid_num, 1], idx_list[mid_num, 2]]
        show_vec = torch.stack((gt_vec, res_vec), dim=1)
        vis.line(Y=show_vec, X=torch.stack((disp_range, disp_range), dim=1), opts=opt, win=win_imgs + str(idx))


def disp_visual(gt_c, res_c, mask_c, vis, win_imgs, nrow=4):
    # Detach & cpu
    gt_c = gt_c.detach().cpu()  # [N, 1, H, W]
    res_c = res_c.detach().cpu()
    mask = mask_c.detach().cpu()
    # beta = 716
    # alpha = 16.0 * 63

    # Calculate disp for N set
    gt_disp_set = gt_c * mask.float()
    res_disp_set = res_c * mask.float()
    show_disp_set = torch.cat((gt_disp_set, res_disp_set), dim=3)
    show_disp_set = torch.nn.functional.interpolate(input=show_disp_set, scale_factor=4.0, mode='nearest')
    vis.images(show_disp_set * 255.0, nrow=nrow, padding=2, win=win_imgs)


def pattern_visual(input_set, vis, win_imgs, win_cam, win_pattern):
    # disp_c = input_set[0]
    selected_prob = input_set[0]
    mask_c = input_set[1]
    pattern = input_set[2][0, :, :, :]
    image = input_set[3]
    sparse_pattern = input_set[4][0, :, :, :]

    # Calculate disp for N set
    selected_prob[mask_c == 0] = 0
    show_disp_set = selected_prob
    show_disp_set = torch.nn.functional.interpolate(input=show_disp_set, scale_factor=2.0, mode='nearest')
    vis.images(show_disp_set * 255.0, nrow=1, padding=2, win=win_imgs)

    # Show pattern & image
    vis.image((pattern / 2 + 0.5), win=win_pattern)
    show_img_set = torch.nn.functional.interpolate(input=image, scale_factor=0.25, mode='nearest')
    vis.images(show_img_set / 2 + 0.5, nrow=1, padding=2, win=win_cam)

    # Show pattern boxinfo
    pattern_c_base = sparse_pattern.reshape(sparse_pattern.shape[0],
                                            sparse_pattern.shape[1] * sparse_pattern.shape[2]).transpose(1, 0)
    vis.boxplot(X=pattern_c_base, opts=dict(legend=['R', 'G', 'B']), win='Pattern_box')


def dense_visual(input_set, output_set, vis, win_img, win_disp):
    """
    :param input_set: (image_obs, image_est, disp_in, mask). [N, 3, H, W] for image. [N, 1, H, W] for disp.
    :param output_set: (disp_gt, disp_res). [N, 1, H, W] for them.
    :param vis: visdom handle
    :param win_img: window for image
    :param win_disp: window for disp
    :return: None
    """
    img_obs = input_set[0][0, :, :, :]
    img_est = input_set[1][0, :, :, :].cpu()
    show_img = torch.stack((img_obs, img_est), dim=0)
    show_img = (show_img + 1) / 2
    show_img = torch.nn.functional.interpolate(input=show_img, scale_factor=0.5, mode='nearest')
    vis.images(show_img * 255.0, nrow=2, padding=2, win=win_img)

    mask = input_set[3][0, :, :, :]
    disp_in = input_set[2][0, :, :, :]
    disp_in[mask == 0] = 0
    disp_gt = output_set[0][0, :, :, :]
    disp_gt[mask == 0] = 0
    disp_res = output_set[1][0, :, :, :]
    disp_res[mask == 0] = 0
    show_disp = torch.stack((disp_in, disp_res, disp_gt), dim=0)
    show_disp = torch.nn.functional.interpolate(input=show_disp, scale_factor=0.5, mode='nearest')
    vis.images(show_disp * 255.0, nrow=3, padding=4, win=win_disp)


def iter_visual_report(vis, win_set, input_set):
    """
    Draw visual elements for train_iter.py, report version
    :param vis: visdom environment.
    :param win_set: {'g_loss', 'd_loss', 'image', 'match', 'disp', 'pattern', 'pattern_box'}
    :param input_set: ((i, epoch, length, report_period), (g_loss, d_loss), (pattern_mat, sparse_pattern),
                       (mask_c, image_mat, disp_est, selected_prob, disp_gt))
    :return: report_message
    """
    # Draw loss line
    i, epoch, length, report = input_set[0]
    g_loss, d_loss = input_set[1]
    g_average = g_loss / report
    d_average = d_loss / report
    g_opts = dict(showlegend=True, title='Generator Loss', width=480, height=360)
    d_opts = dict(showlegend=True, title='Discriminator Loss', width=480, height=360)
    x_pos = torch.FloatTensor([epoch + i / length])
    vis.line(X=x_pos, Y=torch.FloatTensor([g_average]), win=win_set['g_loss'],
             update='append', name='train_report', opts=g_opts)
    vis.line(X=x_pos, Y=torch.FloatTensor([d_average]), win=win_set['d_loss'],
             update='append', name='train_report', opts=d_opts)

    # Show pattern, pattern_boxplot
    pattern, sparse_pattern = input_set[2]
    vis.image((pattern / 2 + 0.5), win=win_set['pattern'])
    # pattern_c_base = sparse_pattern.reshape((sparse_pattern.shape[0],
    #                                          sparse_pattern.shape[1] * sparse_pattern.shape[2])).transpose(1, 0)
    # box_opts = dict(showlegend=True, title='Pattern Boxplot', width=480, height=360, legend=['Pixel'])
    # vis.boxplot(X=pattern_c_base, opts=box_opts, win=win_set['pattern_box'])

    # Show rendered image, disparity, prob
    mask_c, grid_image, image, disp_mat, volume_prob, disp_gt = input_set[3]
    # image[mask_c == 0] = 0
    disp_mat[mask_c == 0] = 0
    disp_gt[mask_c == 0] = 0
    show_disp_mat = torch.cat((disp_mat, disp_gt), dim=2)
    show_disp_mat = torch.nn.functional.interpolate(input=show_disp_mat, scale_factor=2.0, mode='nearest')
    # vis.images(show_disp_mat, nrow=4, padding=2, win=win_set['disp'])
    show_img_mat = torch.cat((grid_image, image), dim=2)
    show_img_set = torch.nn.functional.interpolate(input=show_img_mat, scale_factor=0.25, mode='nearest')
    show_mat = torch.cat((show_img_set / 2 + 0.5, show_disp_mat), dim=2)
    vis.images(show_mat, nrow=4, padding=2, win=win_set['image'])

    # Draw function plot
    # plot_opt = dict(showlegend=False, title='Pixelvise Function', width=360, height=180)
    # disp_range = (torch.Tensor(range(64, 0, -1)) - 1) / 64  # [D]
    # for idx in range(0, volume_prob.shape[0]):
    #     mask_part = mask_c[idx, :, :, :]  # [1, H, W]
    #     idx_list = mask_part.nonzero()  # [num, 3]
    #     pt_num = idx_list.shape[0]
    #     mid_num = int(pt_num / 2)
    #     res_vec = volume_prob[idx, :, idx_list[mid_num, 1], idx_list[mid_num, 2]]
    #     vis.line(Y=res_vec, X=disp_range, opts=plot_opt, win=win_set['disp'] + str(idx))

    # Generate report info
    report_message = '[%d, %4d/%d]: %.2e, %.2e' % (epoch, i + 1, length, g_average, d_average)
    return report_message


def iter_visual_epoch(vis, win_set, input_set):
    """
    Draw visual elements for train_iter.py, epoch version
    :param vis: visdom environment
    :param win_set: {'g_loss', 'd_loss'}
    :param input_set: ((epoch, length), (g_loss, d_loss))
    :return: report message
    """
    # Draw loss line
    epoch, length = input_set[0]
    g_loss, d_loss = input_set[1]
    g_average = g_loss / length
    d_average = d_loss / length
    g_opts = dict(showlegend=True, title='Generator Loss', width=480, height=360)
    d_opts = dict(showlegend=True, title='Discriminator Loss', width=480, height=360)
    x_pos = torch.FloatTensor([epoch + 0.5])
    vis.line(X=x_pos, Y=torch.FloatTensor([g_average]), win=win_set['g_loss'],
             update='append', name='train_epoch', opts=g_opts)
    vis.line(X=x_pos, Y=torch.FloatTensor([d_average]), win=win_set['d_loss'],
             update='append', name='train_epoch', opts=d_opts)
    # Generate report info
    report_message = '    Epoch Train[%d]: %.2e, %.2e' % (epoch, g_average, d_average)
    return report_message


def iter_visual_test(vis, win_set, input_set):
    """
    Draw visual elements for train_iter.py, test version
    :param vis: visdom environment
    :param win_set: {'g_loss', 'd_loss'}
    :param input_set: ((epoch, length), (g_loss, d_loss))
    :return: report message
    """
    # Draw loss line
    epoch, length = input_set[0]
    g_loss, d_loss = input_set[1]
    g_average = g_loss / length
    d_average = d_loss / length
    g_opts = dict(showlegend=True, title='Generator Loss', width=480, height=360)
    d_opts = dict(showlegend=True, title='Discriminator Loss', width=480, height=360)
    x_pos = torch.FloatTensor([epoch + 0.5])
    vis.line(X=x_pos, Y=torch.FloatTensor([g_average]), win=win_set['g_loss'],
             update='append', name='test_epoch', opts=g_opts)
    vis.line(X=x_pos, Y=torch.FloatTensor([d_average]), win=win_set['d_loss'],
             update='append', name='test_epoch', opts=d_opts)
    mask_c, grid_image, image, disp_mat, disp_gt = input_set[2]
    disp_mat[mask_c == 0] = 0
    disp_gt[mask_c == 0] = 0
    show_disp_mat = torch.cat((disp_mat, disp_gt), dim=2)
    show_disp_mat = torch.nn.functional.interpolate(input=show_disp_mat, scale_factor=2.0, mode='nearest')
    # vis.images(show_disp_mat, nrow=4, padding=2, win=win_set['disp'])
    show_img_mat = torch.cat((grid_image, image), dim=2)
    show_img_set = torch.nn.functional.interpolate(input=show_img_mat, scale_factor=0.25, mode='nearest')
    show_mat = torch.cat((show_img_set / 2 + 0.5, show_disp_mat), dim=2)
    vis.images(show_mat, nrow=4, padding=2, win=win_set['test_set'])

    # Generate report info
    report_message = '    Epoch Test[%d]: %.2e, %.2e' % (epoch, g_average, d_average)
    return report_message
