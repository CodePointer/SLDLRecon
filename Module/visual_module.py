import torch
import visdom


def normalize_mat(input_mat, mask, val_range):
    min_val, max_val = val_range
    alpha, beta = max_val - min_val, min_val
    new_mat = (input_mat - beta) / alpha
    new_mat[mask == 0] = 0
    return new_mat


def iter_report(vis, win_set, input_set):
    """Visualization function for every iteration

    Draw error line, learning_rate line.

    :param vis: visdom environment.
    :param win_set: {'pre_loss'}
    :param input_set: (
            (iter, loss, lr),
        )
    :return: report_message
    """
    # Draw loss line
    iter_n, loss, lr = input_set[0]
    pre_opts = dict(showlegent=True, title='Predictor', width=480, height=360)
    x_pos = torch.FloatTensor([iter_n])
    vis.line(X=x_pos, Y=torch.FloatTensor([loss]), win=win_set['pre_loss'],
             update='append', name='iteration', opts=pre_opts)
    # lr_opts = dict(showlegent=True, title='Learning rate', width=480, height=360)
    # vis.line(X=x_pos, Y=torch.log10(torch.FloatTensor([lr])), win=win_set['lr_info'],
    #          update='append', name='train_lr', opts=lr_opts)
    report_message = 'iter[%7d](lr=%.2e): loss %.4e' % (iter_n, lr, loss)
    return report_message


def show_report(vis, win_set, input_set):
    """Visualization function for report period.

    Show depth_estimation result.

    :param vis: visdom environment.
    :param win_set: {'depth'}
    :param input_set: ((depth_est, depth_gt))
    :return: None
    """
    depth_est, depth_gt, mask_mat = input_set[0]
    depth_est[mask_mat == 0] = 0
    show_depth_mat = torch.cat((depth_est, depth_gt), dim=3)
    show_depth_mat = torch.nn.functional.interpolate(input=show_depth_mat, scale_factor=0.35, mode='nearest')
    show_depth_mat = (show_depth_mat - 1.8) / (3.0 - 1.8)       # Normalize
    vis.images(show_depth_mat, nrow=2, padding=2, win=win_set['depth'])


def epoch_report(vis, win_set, input_set):
    """Draw epoch loss in another window.

    :param vis: visdom environment.
    :param win_set: {'epoch_loss'}
    :param input_set: ((epoch, epoch_loss))
    :return: None
    """
    epoch, epoch_loss = input_set[0]
    epoch_opts = dict(showlegent=True, title='Predictor Epoch', width=480, height=360)
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([epoch_loss]), win=win_set['epoch_loss'],
             update='append', name='epoch', opts=epoch_opts)
    return


def test_report(vis, win_set, input_set):
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
