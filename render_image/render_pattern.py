import sys
import configparser
from History.data_set import CameraDataSet
from torch.utils.data import DataLoader
import torch
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def gkern(kernlen=21, nsig=3.0):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.0) / kernlen
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
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
        pattern_noise = torch.randn(pattern.shape) * p_noise_rad + pattern_bias
    else:
        pattern_noise = torch.zeros(pattern.shape)
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
        image_noise = torch.randn(image_mat.shape) * i_noise_rad
        image_mat = image_mat + image_noise

    image_mat = torch.clamp(image_mat, min=-1, max=1)
    image_mat.masked_fill_(mask_mat == 0, -1)
    return image_mat


def render_patterns():
    # Step 0: Set data_loader, visual
    opts = {'header': config.get('DataLoader', 'opt_header').split(',')}
    disp_range = [int(x) for x in config.get('DataLoader', 'disp_range').split(',')]

    cam_data_set = CameraDataSet(root_dir=config.get('FilePath', 'root_path'),
                                 list_name=config.get('DataLoader', 'total_list'),
                                 down_k=config.getint('Paras', 'down_k'),
                                 disp_range=disp_range,
                                 opts=opts)
    data_loader = DataLoader(cam_data_set, batch_size=config.getint('Paras', 'batch_size'),
                             shuffle=False, num_workers=0)

    gkernel_numpy = gkern(kernlen=config.getint('RenderPara', 'gkern_size'),
                          nsig=config.getfloat('RenderPara', 'gkern_sigma'))
    gkernel = torch.from_numpy(gkernel_numpy).unsqueeze(0).unsqueeze(1)  # [1, 1, Hk, Wk]
    gkernel = gkernel.float()
    print('Step 0: DataSet initialize finished.')
    print('    DataLoader size: (%d).' % len(data_loader))

    # Step 3: Training
    pattern_raw = plt.imread(config.get('FilePath', 'pattern_name'))
    pattern = torch.from_numpy(pattern_raw.transpose((2, 0, 1)) * 1.9 - 1) + 0.1
    pattern = pattern[0, :, :]
    N = 1
    Hc = config.getint('Global', 'cam_height')
    Wc = config.getint('Global', 'cam_width')
    Hp = config.getint('Global', 'pro_height')
    Wp = config.getint('Global', 'pro_width')
    pattern = pattern.reshape((1, 1, Hp, Wp)).repeat((N, 1, 1, 1))

    report_period = config.getint('Paras', 'report_period')
    for i, data in enumerate(data_loader, 0):
        # Get data
        idx = data['idx']
        mask_mat = data['mask_mat']
        idx_vec = data['idx_vec']

        ren_image = render_image(pattern=pattern, idx_vec=idx_vec, mask_mat=mask_mat, gkernel=gkernel)

        # Find save path
        save_path = cam_data_set.root_dir + cam_data_set.image_frame[idx][0]
        file_idx = str(i % 1000)
        ren_name = ''.join((save_path, 'dot_image', file_idx, '.png'))
        plt.imsave(ren_name, ren_image.squeeze().numpy(), cmap='Greys_r')

        print('.', flush=True, end='')
        if i % report_period == report_period - 1:
            print('(%4d/%d)' % (i + 1, len(data_loader)))

    print('Step 3: Finish training.')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    config_path = sys.argv[1]

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)

    render_patterns()
