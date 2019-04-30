# -*- coding: utf-8 -*-

"""
This file is to process data generated by OpenGL.
Data generated by OpenGL:
    depth_cam, depth_pro1, depth_pro2 -- from different views;
    Binary files;
    No mask info.
Main work:
    1. Read binary file and store them as npy file.
        a) mXXfXX_depth_camXX.npy  str(x).zfill(3)
        b) mask_mat for every type of output
    2. Calculate some calibrated information:
        a) Calculate: M_cp1, D_cp1, M_p1c, D_p1c
    3. Generate flow data.
        a) calculate xy_pro1_cv.npy data
        b) calculate xy_cam_p1v.npy data
        c) save all information.
    4. Usage
        When using the data, load the xy_cam_p1v.npy & xy_pro_cv.npy
            grid_sample: xy_cam_cv.npy data
            (https://pytorch.org/docs/stable/nn.html?highlight=grid_sample#torch.nn.functional.grid_sample)
"""

import csv
import sys
import os
import torch
from matplotlib import pyplot as plt
import numpy as np


def load_bin_file(file_name, shape):
    """load binary file from disk and convert them into torch tensor/numpy array.

    The binary file is generated from OpenGL program outside. As the OpenGL is row-major, we don't need to do anything
    about the shape, just use reshape directly. However if the mat is generated from matlab, a permute() method is
    needed.
    Also, the OpenGL program is running on the windows platform. The out put variables are float with little endian.
    Output of this method is torch tensor.

    Args:
        :param file_name: '.bin' file name. Will be load directly.
        :param shape: The shape of image. Needed for binary file. Dimention is 2.

    Returns:
        :return: A torch tensor with given shape.

    Raises:
        IOError: Error occurred if cannot load file.
    """
    try:
        item_vec = np.fromfile(file_name, dtype='<f4')
    except IOError as error:
        print("File_name is %s" % file_name)
        raise error

    # item = torch.from_numpy(item_vec.reshape(shape[0], shape[1]))
    item = item_vec.reshape(shape[0], shape[1])
    # item = item.unsqueeze(0)
    # item[torch.isnan(item)] = 0
    return item


def main():

    # Some parameters (You should edit here every time):
    main_path = 'E:/SLDataSet/Thing10K'
    folder_name = 'test_dataset'
    model_num = 2
    frame_num = 100
    image_shape = (1024, 1280)
    cam_intrinsic = np.array([[2000, 0, 640], [0, 2000, 512], [0, 0, 1]])
    trans_vec = np.array([0.4799, -0.0500, 0.1406]).reshape(3, 1)
    rot_mat = np.array([[0.9659, 0.0, -0.2588], [0.0, 1.0, 0.0], [0.2588, 0.0, 0.9659]])

    # Step 1: Get dataset file_name list
    depth_cam_names = []
    depth_pro1_names = []
    depth_pro2_names = []
    idx_set = []
    for m_idx in range(0, model_num + 1):
        for f_idx in range(0, frame_num):
            depth_cam_names.append('%s/%d/depth_cam/depth_view%d.bin' % (main_path, m_idx, f_idx))
            depth_pro1_names.append('%s/%d/depth_pro1/depth_view%d.bin' % (main_path, m_idx, f_idx))
            depth_pro2_names.append('%s/%d/depth_pro2/depth_view%d.bin' % (main_path, m_idx, f_idx))
            idx_set.append((m_idx, f_idx))
    total_frame_num = len(depth_cam_names)

    # Step 2: Calculate epipolar information here
    # Calculate Mcp1, Dcp1; Mp1c, Dp1c
    pro_matrix = np.dot(cam_intrinsic, np.hstack((rot_mat, trans_vec)))
    cam_matrix = np.dot(cam_intrinsic, np.hstack((rot_mat.transpose(),
                                                  -np.dot(rot_mat.transpose(), trans_vec))))
    par_mcp1 = np.zeros((image_shape[0], image_shape[1], 3))
    par_mp1c = np.zeros((image_shape[0], image_shape[1], 3))
    par_dcp1 = pro_matrix[:, 3]
    par_dp1c = cam_matrix[:, 3]
    for h in range(0, image_shape[0]):
        for w in range(0, image_shape[1]):
            tmp_vec = np.array([(w - cam_intrinsic[0, 2]) / cam_intrinsic[0, 0],
                                (h - cam_intrinsic[1, 2]) / cam_intrinsic[1, 1],
                                1])
            par_mcp1[h, w, :] = np.dot(pro_matrix[:, :3], tmp_vec)
            par_mp1c[h, w, :] = np.dot(cam_matrix[:, :3], tmp_vec)
    # print(par_mp1c[378, 597, :])
    # print(par_dp1c)

    # Step 3: Process every frame
    file_path = '%s/%s' % (main_path, folder_name)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for idx in range(0, total_frame_num):
        depth_cam = load_bin_file(depth_cam_names[idx], image_shape)
        depth_pro1 = load_bin_file(depth_pro1_names[idx], image_shape)
        # depth_pro2 = load_bin_file(depth_pro2_names[idx], image_shape)

        # Set mask
        mask_cam = depth_cam < 9.0
        depth_cam[mask_cam == 0] = 0.0
        mask_pro1 = depth_pro1 < 9.0
        depth_pro1[mask_pro1 == 0] = 0.0
        # mask_pro2 = depth_pro2 < 9.0
        # depth_pro2[mask_pro2 == 0] = 0.0

        # Calculate xy_pro1_cv
        xy_pro1_cv = np.zeros((image_shape[0], image_shape[1], 2))
        tmp_mat_cv = par_mcp1[:, :, 2] * depth_cam + par_dcp1[2]
        xy_pro1_cv[:, :, 0] = par_mcp1[:, :, 0] * depth_cam + par_dcp1[0]
        xy_pro1_cv[:, :, 0] = xy_pro1_cv[:, :, 0] / tmp_mat_cv
        xy_pro1_cv[:, :, 1] = par_mcp1[:, :, 1] * depth_cam + par_dcp1[1]
        xy_pro1_cv[:, :, 1] = xy_pro1_cv[:, :, 1] / tmp_mat_cv
        xy_pro1_cv[mask_cam == 0] = 0.0

        # Calculate xy_cam_p1v
        xy_cam_p1v = np.zeros((image_shape[0], image_shape[1], 2))
        tmp_mat_p1v = par_mp1c[:, :, 2] * depth_pro1 + par_dp1c[2]
        xy_cam_p1v[:, :, 0] = par_mp1c[:, :, 0] * depth_pro1 + par_dp1c[0]
        xy_cam_p1v[:, :, 0] = xy_cam_p1v[:, :, 0] / tmp_mat_p1v
        xy_cam_p1v[:, :, 1] = par_mp1c[:, :, 1] * depth_pro1 + par_dp1c[1]
        xy_cam_p1v[:, :, 1] = xy_cam_p1v[:, :, 1] / tmp_mat_p1v
        xy_cam_p1v[mask_pro1 == 0] = 0.0

        # Save
        prefix = 'm%02df%03d' % (idx_set[idx][0], idx_set[idx][1])
        np.save('%s/%s_depth_cam.npy' % (file_path, prefix), depth_cam.astype(np.float32))
        plt.imsave('%s/%s_mask_cam.png' % (file_path, prefix), mask_cam, cmap='Greys_r')
        np.save('%s/%s_depth_pro1.npy' % (file_path, prefix), depth_pro1.astype(np.float32))
        plt.imsave('%s/%s_mask_pro1.png' % (file_path, prefix), mask_pro1, cmap='Greys_r')
        np.save('%s/%s_xy_pro1_cv.npy' % (file_path, prefix), xy_pro1_cv.astype(np.float32))
        np.save('%s/%s_xy_cam_p1v.npy' % (file_path, prefix), xy_cam_p1v.astype(np.float32))
        print('%s/%s writing finished.' % (file_path, prefix))

    return


if __name__ == '__main__':
    assert len(sys.argv) >= 1

    main()
