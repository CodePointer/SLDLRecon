# -*- coding: utf-8 -*-

"""
This file is to process data processed by coord_generator.py
Data generated by OpenGL:
    depth_cam, depth_pro1, depth_pro2 -- from different views;
    Binary files;
    No mask info.
Data processed by coord_generator.py:
    prefix = 'm%02df%03d' % (idx_set[idx][0], idx_set[idx][1])
        np.save('%s/%s_depth_cam.npy' % (file_path, prefix), depth_cam)
        plt.imsave('%s/%s_mask_cam.png' % (file_path, prefix), mask_cam, cmap='Greys_r')
        np.save('%s/%s_depth_pro1.npy' % (file_path, prefix), depth_pro1)
        plt.imsave('%s/%s_mask_pro1.png' % (file_path, prefix), mask_pro1, cmap='Greys_r')
        np.save('%s/%s_xy_pro1_cv.npy' % (file_path, prefix), xy_pro1_cv)
        np.save('%s/%s_xy_cam_p1v.npy' % (file_path, prefix), xy_cam_p1v)
        plt.imsave('%s/%s_shade_cam.png' % (file_path, prefix), shade_cam.reshape(image_shape), cmap='Greys_r')
        plt.imsave('%s/%s_shade_pro1.png' % (file_path, prefix), shade_pro1.reshape(image_shape), cmap='Greys_r')
        print('%s/%s writing finished.' % (file_path, prefix))
Main work:
    1. Set jump value for flow calculation.
        The jump value is used for frame calculation. Flow will be calculated with i and i+jump frame.
        Default: 1
    2. Load needed information. Including:
        depth_cam(only load and save)
        mask_cam(only load and save)
        xy_pro1_cv.npy
        xy_cam_p1v_j.npy
        mask_pro1_j.png
        shade_cam.png
        pattern.png (For image generation)
    3. Calculate flow by using grid_sample.
        (https://pytorch.org/docs/stable/nn.html?highlight=grid_sample#torch.nn.functional.grid_sample)
        sample xy_cam_p1v_j.npy, by using xy_pro1_cv.npy.
        Get: xy_cam_cv_j.npy
        With xy_cam_cv.npy, calculate flow1_j1.npy.
        Also, sample mask_pro1_j.png to get mask_flow1.png.
    4. Draw patterns on the depth map. (With noise)
        a) Pattern noise. Added on every pixel of pattern.
        b) Shading effect. Plus shade mat for image rendering.
        c) Image noise. Added on every pixel of image.
        sample pattern.png, by using xy_pro1_cv.npy.
        Get: img_cam.png
    5. Save information
        depth_cam(only load and save)
        mask_cam(only load and save)
        flow1_j1.npy
        mask_flow1.png
        img_cam.png
"""

import configparser
import sys
import os
import torch
import numpy as np
import DataProcess.util as dp


def main(main_path, data_set):

    # Load config
    main_path = main_path if main_path[-1] == '/' else main_path + '/'
    cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    cfg.read(main_path + '000config.ini')

    # Some parameters (You should edit here every time):
    in_path = main_path + '2_DepthWithCorres/%s_corres/' % data_set
    out_path = main_path + '3_PatternedImage/%s_dataset/' % data_set
    model_num = cfg.getint('DataInfo', '%s_model' % data_set)
    frame_num = cfg.getint('DataInfo', 'frame_num')
    cam_shape = (cfg.getint('Global', 'cam_height'), cfg.getint('Global', 'cam_width'))
    pro_shape = (cfg.getint('Global', 'pro_height'), cfg.getint('Global', 'pro_width'))
    jump = cfg.getint('DataInfo', 'jump')
    flow_thred = cfg.getfloat('DataInfo', 'flow_thred')
    pattern_pix_size = cfg.getint('DataInfo', 'pat_size')
    # Noise parameters
    render_paras = dict(i_gn_rad=cfg.getfloat('Render', 'i_gn_rad'),
                        p_gn_rad=cfg.getfloat('Render', 'p_gn_rad'),
                        i_gb_rad=cfg.getint('Render', 'i_gb_rad'),
                        i_gb_sig=cfg.getfloat('Render', 'i_gb_sig'),
                        p_gb_rad=cfg.getint('Render', 'p_gb_rad'),
                        p_gb_sig=cfg.getfloat('Render', 'p_gb_sig'))

    # Main Loop
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    x_cam_grid = torch.arange(0, cam_shape[1]).reshape(1, -1).repeat(cam_shape[0], 1)
    y_cam_grid = torch.arange(0, cam_shape[0]).reshape(-1, 1).repeat(1, cam_shape[1])
    xy_cam_grid = torch.stack((x_cam_grid, y_cam_grid), dim=0).reshape(1, 2, cam_shape[0], cam_shape[1]).float()
    xy_cam_grid = xy_cam_grid.cuda()

    pattern = dp.load_as_torch(full_name=main_path + '4pix_pattern.png')
    for m_idx in range(1, model_num + 1):
        # Step 1: Load all information needed.
        print("Loading No.%02d model..." % m_idx, end='', flush=True)
        depth_cam = []
        mask_cam = []
        xy_pro1_cv = []
        xy_cam_p1v = []
        mask_pro1 = []
        shade_cam = []
        for f_idx in range(0, frame_num):
            full_pre = in_path + 'm%02df%03d_' % (m_idx, f_idx)
            depth_cam.append(dp.load_as_torch(full_pre + 'depth_cam.npy'))
            mask_cam.append(dp.load_as_torch(full_pre + 'mask_cam.png', dtype=np.uint8))
            xy_pro1_cv.append(dp.load_as_torch(full_pre + 'xy_pro1_cv.npy'))
            xy_cam_p1v.append(dp.load_as_torch(full_pre + 'xy_cam_p1v.npy'))
            mask_pro1.append(dp.load_as_torch(full_pre + 'mask_pro1.png', dtype=np.uint8))
            shade_cam.append(dp.load_as_torch(full_pre + 'shade_cam.png'))
        print("Finished.")

        # Step 2: Process every frame, calculate flow.
        img_cam_list = []
        for f_idx in range(0, frame_num):
            i = f_idx
            j = i + jump
            if j >= frame_num:
                j -= frame_num
            # xy_pro1_cv to [-1, 1]
            xy_pro1_cv[i][:, 0, :, :] = xy_pro1_cv[i][:, 0, :, :] / (pro_shape[1] - 1) * 2.0 - 1.0
            xy_pro1_cv[i][:, 1, :, :] = xy_pro1_cv[i][:, 1, :, :] / (pro_shape[0] - 1) * 2.0 - 1.0
            xy_pro1_cv[i][mask_cam[i].repeat(1, 2, 1, 1) == 0] = -1
            xy_pro1_cv[i] = xy_pro1_cv[i].permute(0, 2, 3, 1)  # [N(1), H, W, 2]

            # Sample xy_cam_cv from xy_cam_p1v_j
            xy_cam_cv = torch.nn.functional.grid_sample(input=xy_cam_p1v[j].float(), grid=xy_pro1_cv[i].float())
            mask_flow = torch.nn.functional.grid_sample(input=mask_pro1[j].float(), grid=xy_pro1_cv[i].float())
            mask_flow[mask_flow < 0.99] = 0
            mask_flow[mask_flow > 0] = 1
            xy_cam_cv[mask_flow.repeat(1, 2, 1, 1) == 0] = 0

            # Sample img_cam from xy_cam_p1v
            img_cam = dp.render_image(pattern=pattern,
                                      xy_pro=xy_pro1_cv[i].float(),
                                      shade_mat=shade_cam[i],
                                      pattern_size=pattern_pix_size,
                                      paras=render_paras)
            img_cam[mask_cam[i] == 0] = 0
            img_cam_list.append(img_cam)

            # Calculate flow mat and check valid
            flow_j = xy_cam_cv - xy_cam_grid
            mask_flow[torch.abs(flow_j[:, :1, :, :]) > flow_thred] = 0
            mask_flow[torch.abs(flow_j[:, 1:, :, :]) > flow_thred] = 0
            flow_j[mask_flow.repeat(1, 2, 1, 1) == 0] = 0

            # Save (as i frame)
            full_pre = out_path + 'm%02df%03d_' % (m_idx, f_idx)
            dp.save_from_torch(full_name=full_pre + 'depth_cam.npy', mat=depth_cam[i].float())
            dp.save_from_torch(full_name=full_pre + 'mask_cam.png', mat=mask_cam[i])
            dp.save_from_torch(full_name=full_pre + 'flow1_cv.npy', mat=flow_j.float())
            dp.save_from_torch(full_name=full_pre + 'mask_flow1.png', mat=mask_flow.float())
            # dp.save_from_torch(full_name=full_pre + 'img_cam_1.png', mat=img_cam)

            if i % 5 == 4:
                print('.', end='', flush=True)

        # Save img_cam_2
        for f_idx in range(0, frame_num):
            i = f_idx
            j = i + jump
            j = j if j < frame_num else j - frame_num
            full_pre = out_path + 'm%02df%03d_' % (m_idx, f_idx)
            dp.save_from_torch(full_name=full_pre + 'img_cam_1.png', mat=img_cam_list[i])
            dp.save_from_torch(full_name=full_pre + 'img_cam_2.png', mat=img_cam_list[j])
        print('Finished.')


if __name__ == '__main__':
    assert len(sys.argv) >= 3

    main(main_path=sys.argv[1], data_set=sys.argv[2])
