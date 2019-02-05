import sys
# from Module.generator_net import GeneratorNet
from Module.generator_net_grey import GeneratorNet
import torch
import numpy as np
import visdom


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

    pattern_noise = torch.randn(pattern.shape).cuda() / 3 * p_noise_rad
    dense_pattern = torch.nn.functional.interpolate(input=pattern + pattern_noise, scale_factor=8, mode='bilinear',
                                                    align_corners=False)
    # pattern_rearrange = pattern_mat.transpose(1, 0)
    pattern_rearrange = dense_pattern.transpose(1, 0)
    pattern_search = pattern_rearrange.reshape(pattern_channel, batch_size * pro_height * pro_width)
    pattern_search = torch.clamp(pattern_search, min=-1, max=1)
    idx_vec_plain = idx_vec.reshape(batch_size * cam_height * cam_width)
    est_img_vec = torch.index_select(input=pattern_search, dim=1, index=idx_vec_plain)
    image_mat = est_img_vec.reshape(pattern_channel, batch_size, cam_height, cam_width).transpose(1, 0)
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


def train_iteration(root_path):
    # Step 1: Set data_loader, create net, visual
    batch_size = 8

    vis_env = 'K' + str(3) + '_Network_PatTest'
    vis = visdom.Visdom(env=vis_env)
    win_pat = 'Pattern Test'

    pattern_network = GeneratorNet(root_path=root_path, batch_size=batch_size)
    print('Step 1: Initialize finished.')

    # Step 2: Model loading
    pattern_network.load_state_dict(torch.load('./model_pattern.pt'), strict=False)
    pattern_network.train()
    print('Step 2: Model loading finished.')

    # Step 3: Training
    pattern_network = pattern_network.cuda()
    pattern_seed = torch.from_numpy(np.load('random_seed.npy')).float()
    pattern_seed = (pattern_seed - 0.5) * 2
    pattern_seed = torch.stack([pattern_seed] * 1, dim=0).unsqueeze(1)
    pattern_seed_zero = torch.zeros(1, 1, 64)
    pattern_seed_bias = torch.randn(batch_size-1, 1, 64)
    pattern_seed_noise = torch.cat([pattern_seed_zero, pattern_seed_bias], dim=0)
    # pattern_seed = (pattern_seed + pattern_seed_noise).cuda()
    pattern_seed = torch.cat([pattern_seed, pattern_seed_bias], dim=0).cuda()
    print(pattern_seed.shape)

    sparse_pattern = pattern_network(pattern_seed)
    dense_pattern = torch.nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
                                                    align_corners=False)
    vis.images(dense_pattern / 2 + 0.5, nrow=2, padding=2, win=win_pat)

    print('Step 3: Finish training.')


def main(argv):
    train_iteration(root_path='./SLDataSet/20181204/')


if __name__ == '__main__':
    main(sys.argv)
