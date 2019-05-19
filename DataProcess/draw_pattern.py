# -*- coding: utf-8 -*-

"""
This file is to create patterned image. The input is needed from coord_generator.py .

Data needed:
    prefix = 'm%02df%03d' % (idx_set[idx][0], idx_set[idx][1])
    1. xy_pro1_cv.npy
    2. mask_cam
    3. pattern.png

Main work:
    1. Set jump value for image pair calculation.
        The jump value is used for frame calculation. Image pair will be calculated with i and i+jump frame.
        Default: 1
    2. Load needed information. Including:
        xy_pro1_cv.npy
        mask_cam.png
        pattern.png
    3. Draw pattern by using grid_sample.
        (https://pytorch.org/docs/stable/nn.html?highlight=grid_sample#torch.nn.functional.grid_sample)
        sample pattern.png, by using xy_pro1_cv.npy.
        Get: image_cam.png
    4. Save information
        <image_cam>_1.png
        <image_cam>_2.png

"""

import csv
import sys
import os
import torch
from matplotlib import pyplot as plt
import numpy as np


def load_as_torch(name, suffix, main_path, prefix, channel=1):
    """Load file as torch tensor.

    Possible data:
        depth_cam.npy   [H, W], float
        mask_cam.png    [H, W], byte
        xy_pro1_cv.npy  [H, W, 2], float
        xy_cam_p1v.npy  [H, W, 2], float
        mask_pro.png    [H, W], byte

    Args:
        :param name: the data name.
        :param suffix: '.png' or '.npy'.
        :param main_path: the file path with all the m00f000 data.
        :param prefix: m00f000.

    Return:
        :return: Loaded data or None(if suffix is not available).
                Data is always 4D [N, C, H, W].

    Raises:
        None.
    """
    full_name = '%s/%s_%s%s' % (main_path, prefix, name, suffix)
    if suffix == '.png':
        item = torch.from_numpy(plt.imread(full_name))
        if channel == 1:
            item = item[:, :, 0].unsqueeze(0)
        else:
            item = item.permute(2, 0, 1)
        item = item.unsqueeze(0)
        item = item.byte()
        return item.cuda()
    elif suffix == '.npy':
        item = torch.from_numpy(np.load(full_name).astype(float))
        if len(item.shape) == 2:
            item = item.unsqueeze(0)
        else:
            item = item.permute(2, 0, 1)
        item[torch.isnan(item)] = 0
        item = item.unsqueeze(0)
        return item.cuda()
    else:
        return None


def save_from_torch(mat, name, suffix, out_path, prefix):
    """Save file from torch tensor to npy or png file.

    detach, cpu, numpy. use plt.imsave and np.save function.

    :param name:
    :param suffix:
    :param out_path:
    :param prefix:
    :return:
    """
    full_name = '%s/%s_%s%s' % (out_path, prefix, name, suffix)
    if suffix == '.png':
        # mat_np = mat.detach().cpu().squeeze().numpy()
        mat_np = mat.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        plt.imsave(full_name, mat_np)
    elif suffix == '.npy':
        mat = mat.detach().cpu().squeeze()
        if len(mat.shape) == 3:
            mat = mat.permute(1, 2, 0)
        mat_np = mat.numpy().astype(np.float32)
        np.save(full_name, mat_np)


def main():

    # Some parameters (You should edit here every time):
    main_path = '/media/qiao/数据文档/SLDataSet/Thing10K/test_dataset'
    out_path = '/media/qiao/数据文档/SLDataSet/Thing10K/4pix_dataset/test'
    pattern_path = '/media/qiao/数据文档/SLDataSet/Thing10K/'
    model_num = 2
    frame_num = 100
    jump = 1
    image_shape = (1024, 1280)

    # Main Loop
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # x_cam_grid = torch.arange(0, image_shape[1]).reshape(1, -1).repeat(image_shape[0], 1)
    # y_cam_grid = torch.arange(0, image_shape[0]).reshape(-1, 1).repeat(1, image_shape[1])
    # xy_cam_grid = torch.stack((x_cam_grid, y_cam_grid), dim=0).reshape(1, 2, image_shape[0], image_shape[1]).float()
    # xy_cam_grid = xy_cam_grid.cuda()
    for m_idx in range(1, model_num + 1):
        # Step 1: Load all information needed.
        print("Loading No.%02d model..." % m_idx, end='', flush=True)
        mask_cam = []
        xy_pro1_cv = []
        pattern = load_as_torch('pattern', '.png', pattern_path, '4pix', channel=3)
        for f_idx in range(0, frame_num):
            prefix = 'm%02df%03d' % (m_idx, f_idx)
            mask_cam.append(load_as_torch('mask_cam', '.png', main_path, prefix))
            xy_pro1_cv.append(load_as_torch('xy_pro1_cv', '.npy', main_path, prefix))
        print("Finished.")

        # Step 2: Process every frame, calculate image_cam.
        print("Calculate No.%02d image_cam..." % m_idx, end='', flush=True)
        image_cam = []
        for f_idx in range(0, frame_num):
            i = f_idx
            # xy_pro1_cv to [-1, 1]
            xy_pro1_cv[i][:, 0, :, :] = xy_pro1_cv[i][:, 0, :, :] / (image_shape[1] - 1) * 2.0 - 1.0
            xy_pro1_cv[i][:, 1, :, :] = xy_pro1_cv[i][:, 1, :, :] / (image_shape[0] - 1) * 2.0 - 1.0
            xy_pro1_cv[i][mask_cam[i].repeat(1, 2, 1, 1) == 0] = -1
            xy_pro1_cv[i] = xy_pro1_cv[i].permute(0, 2, 3, 1)  # [N(1), H, W, 2]
            # Sample from xy_cam_p1v
            image = torch.nn.functional.grid_sample(input=pattern.float(), grid=xy_pro1_cv[i].float())
            image[mask_cam[i].repeat(1, 3, 1, 1) == 0] = 0
            image_cam.append(image)
        print("Finished.")

        # Step 3: Save image as image pairs
        print("Save No.%02d image pairs..." % m_idx, end='', flush=True)
        for f_idx in range(0, frame_num):
            prefix = 'm%02df%03d' % (m_idx, f_idx)
            i = f_idx
            j = i + jump
            if j >= frame_num:
                j -= frame_num
            save_from_torch(image_cam[i], 'img_cam_img1', '.png', out_path, prefix)
            save_from_torch(image_cam[j], 'img_cam_img2', '.png', out_path, prefix)
        print("Finished.")


if __name__ == '__main__':
    assert len(sys.argv) >= 1

    main()
