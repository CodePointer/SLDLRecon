import torch
import visdom


def normalize_mat(input_mat, mask, val_range):
    min_val, max_val = val_range
    alpha, beta = max_val - min_val, min_val
    new_mat = (input_mat - beta) / alpha
    new_mat[mask == 0] = 0
    return new_mat


def iter_visual_report(vis, win_set, input_set):
    """
    Draw visual elements for train_iter.py, report version
    :param vis: visdom environment.
    :param win_set: {'g_loss', 'd_loss', 'image', 'match', 'disp', 'pattern', 'pattern_box'}
    :param input_set: ((i, epoch, length, report_period), (g_loss, d_loss),
                       (disp_gt, disp_gt_t, disp_est, disp_est_t, mask_flow, mask_flow_t),
                       (min_val, max_val))
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
    # vis.line(X=x_pos, Y=torch.FloatTensor([g_average]), win=win_set['g_loss'],
    #          update='append', name='train_report', opts=g_opts)
    # vis.line(X=x_pos, Y=torch.FloatTensor([d_average]), win=win_set['d_loss'],
    #          update='append', name='train_report', opts=d_opts)

    # Show generated disparity
    disp_gt, disp_gt_t, disp_est, disp_est_t, mask_flow, mask_flow_t = input_set[2]
    disp_range = input_set[3]
    # print('disp_est: ', torch.min(disp_est), torch.max(disp_est))
    # disp_gt = normalize_mat(disp_gt, mask_flow, disp_range)
    # print('disp_gt: ', torch.min(disp_gt), torch.max(disp_gt))
    # disp_gt_t = normalize_mat(disp_gt_t, mask_flow_t, disp_range)
    # print('disp_gt_t: ', torch.min(disp_gt_t), torch.max(disp_gt_t))
    # disp_est = normalize_mat(disp_est, mask_flow, disp_range)
    # disp_est_t = normalize_mat(disp_est_t, mask_flow_t, disp_range)
    if disp_gt_t is None:
        show_disp_mat = torch.cat((disp_gt, disp_est), dim=3)
    else:
        show_disp_mat = torch.cat((disp_est, disp_est_t, disp_gt, disp_gt_t), dim=3)
    show_disp_mat = torch.nn.functional.interpolate(input=show_disp_mat, scale_factor=0.25, mode='nearest')
    vis.images(show_disp_mat, nrow=1, padding=2, win=win_set['disp'])

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
    # vis.line(X=x_pos, Y=torch.FloatTensor([g_average]), win=win_set['g_loss'],
    #          update='append', name='train_epoch', opts=g_opts)
    # vis.line(X=x_pos, Y=torch.FloatTensor([d_average]), win=win_set['d_loss'],
    #          update='append', name='train_epoch', opts=d_opts)
    # Draw lr line (log10)
    g_lr, d_lr = input_set[2]
    # vis.line(X=x_pos, Y=torch.log10(torch.FloatTensor([g_lr])), win=win_set['lr_info'],
    #          update='append', name='train_lr', opts=g_opts)

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
    # vis.line(X=x_pos, Y=torch.FloatTensor([g_average]), win=win_set['g_loss'],
    #          update='append', name='test_epoch', opts=g_opts)
    # vis.line(X=x_pos, Y=torch.FloatTensor([d_average]), win=win_set['d_loss'],
    #          update='append', name='test_epoch', opts=d_opts)

    # Generate report info
    report_message = '    Epoch Test[%d]: %.2e, %.2e' % (epoch, g_average, d_average)
    return report_message
