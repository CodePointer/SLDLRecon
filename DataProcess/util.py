# -*- coding: utf-8 -*-

"""
Some functions often used in data processing.

Mainly some coordinate transforming things, or loading functions.
"""


import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import scipy.stats as st
import cv2


# ----------------------------------- #
# Small functions
# ----------------------------------- #
def gkern(kernlen=21, nsig=3.0):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.0) / kernlen
    x = np.linspace(-nsig-interval/2.0, nsig+interval/2.0, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel.astype(np.float32)


# ----------------------------------- #
# IO functions
# ----------------------------------- #
def load_bin_file(file_name, shape, row_major=True):
    """load binary file from disk and convert them into torch numpy array.

    The binary file is generated from OpenGL program outside. As the OpenGL is row-major, we don't need to do anything
    about the shape, just use reshape directly. However if the mat is generated from matlab, a permute() method is
    needed.
    Also, the OpenGL program is running on the windows platform. The out put variables are float with little endian.
    Output of this method is torch tensor.

    Args:
        :param file_name: '.bin' file name. Will be load directly.
        :param shape: The shape of image. Needed for binary file. Dimention is 2.
        :param row_major:   If the input binary is saved as row major. For matlab, the binary is saved as col_major by
                            default.

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
    item = item_vec.reshape(shape[0], shape[1]) if row_major else item_vec.reshape(shape[1], shape[0]).transpose()
    # item = item.unsqueeze(0)
    # item[torch.isnan(item)] = 0
    return item


def load_as_torch(full_name, dtype=np.float32):
    """Load npy or png file as torch tensor.

    Possible data:
        depth_cam.npy   [H, W], float
        mask_cam.png    [H, W], byte
        xy_pro1_cv.npy  [H, W, 2], float
        xy_cam_p1v.npy  [H, W, 2], float
        mask_pro.png    [H, W], byte

    Args:
        :param full_name: the data name.
        :param dtype: Wanted data type for saving. Float32 for default.

    Return:
        :return: Loaded data or None(if suffix is not available).
                Data is always 4D [N, C, H, W].

    Raises:
        None.
    """
    extension = os.path.splitext(full_name)[1]
    if extension == '.png':
        item = torch.from_numpy(plt.imread(full_name).astype(dtype))
        if len(item.shape) == 3:
            item = item[:, :, 0].unsqueeze(0)
        else:
            item = item.unsqueeze(0)
        item = item.unsqueeze(0)
        return item.cuda()
    elif extension == '.npy':
        item = torch.from_numpy(np.load(full_name).astype(dtype))
        if len(item.shape) == 2:
            item = item.unsqueeze(0)
        else:
            item = item.permute(2, 0, 1)
        item[torch.isnan(item)] = 0
        item = item.unsqueeze(0)
        return item.cuda()
    else:
        return None


def save_from_torch(full_name, mat):
    """Save file from torch tensor to npy or png file.

    detach, cpu, numpy. use plt.imsave and np.save function.

    Param:
        :param mat:
        :param full_name:

    Return:
        :return:
    """
    extension = os.path.splitext(full_name)[1]
    if extension == '.png':
        mat_np = mat.detach().cpu().squeeze().numpy()
        plt.imsave(full_name, mat_np, cmap='Greys_r')
    elif extension == '.npy':
        mat = mat.detach().cpu().squeeze()
        if len(mat.shape) == 3:
            mat = mat.permute(1, 2, 0)
        mat_np = mat.numpy().astype(np.float32)
        np.save(full_name, mat_np)


# ----------------------------------- #
# Coordination transformation
# ----------------------------------- #
def process_epi_info(cam_intrinsic, pro_intrinsic, rotation, transition, cam_shape=None, pro_shape=None, scale=1.0):
    """Calculate M, D for camera & projector

    Param:
        :param cam_intrinsic:   4 parameters (fx, dx, fy, dy).
        :param pro_intrinsic:   4 parameters (fx, dx, fy, dy).
        :param rotation:        Rotation matrix, row major.
        :param transition:      Transition vector.
        :param cam_shape:       Camera shape. If None, then calculate from (dx, dy).
        :param pro_shape:       Projector shape. If None, calculate from (dx, dy).
        :param scale:           Scale for transition.

    Return:
        :return: A list contains: Mcp, Dcp, Mpc, Dpc.

    Raise:
        None.
    """
    # Reshape calibrated parameters
    fx, dx, fy, dy = cam_intrinsic
    cam_mat = np.array([fx, 0.0, dx, 0.0, fy, dy, 0.0, 0.0, 1.0]).reshape(3, 3)
    fx, dx, fy, dy = pro_intrinsic
    pro_mat = np.array([fx, 0.0, dx, 0.0, fy, dy, 0.0, 0.0, 1.0]).reshape(3, 3)
    rot_mat = np.array(rotation).reshape(3, 3)
    trans_vec = np.array(transition).reshape(3, 1) * scale

    pro_matrix = np.dot(pro_mat, np.hstack((rot_mat, trans_vec)))
    cam_matrix = np.dot(cam_mat, np.hstack((rot_mat.transpose(),
                                            -np.dot(rot_mat.transpose(), trans_vec))))
    cam_shape = (int(cam_mat[1, 2] * 2), int(cam_mat[0, 2] * 2)) if cam_shape is None else cam_shape
    pro_shape = (int(pro_mat[1, 2] * 2), int(pro_mat[0, 2] * 2)) if pro_shape is None else pro_shape
    par_mcp1 = np.zeros((cam_shape[0], cam_shape[1], 3))
    par_mp1c = np.zeros((pro_shape[0], pro_shape[1], 3))
    par_dcp1 = pro_matrix[:, 3]
    par_dp1c = cam_matrix[:, 3]
    for h in range(0, cam_shape[0]):
        for w in range(0, cam_shape[1]):
            tmp_vec_cam = np.array([(w - cam_mat[0, 2]) / cam_mat[0, 0],
                                    (h - cam_mat[1, 2]) / cam_mat[1, 1],
                                    1])
            par_mcp1[h, w, :] = np.dot(pro_matrix[:, :3], tmp_vec_cam)
    for h in range(0, pro_shape[0]):
        for w in range(0, pro_shape[1]):
            tmp_vec_pro = np.array([(w - pro_mat[0, 2]) / pro_mat[0, 0],
                                    (h - pro_mat[1, 2]) / pro_mat[1, 1],
                                    1])
            par_mp1c[h, w, :] = np.dot(cam_matrix[:, :3], tmp_vec_pro)
    return par_mcp1, par_dcp1, par_mp1c, par_dp1c


def fill_depth_mat(depth_mat, mask_mat, hole_mat, pat_rad=5):
    """The function to fill the depth_mat's hole

    :param depth_mat:
    :param mask_mat:
    :param hole_mat:
    :param pat_rad:
    :return: depth_mat with hole_filling.
    """
    valid_mat = mask_mat.copy()
    valid_mat[hole_mat == 1] = 0
    depth_mat[valid_mat == 0] = 0
    valid_sum = cv2.blur(valid_mat.astype(np.float32), (pat_rad, pat_rad))
    depth_sum = cv2.blur(depth_mat, (pat_rad, pat_rad))
    valid_sum[valid_sum == 0] = 1
    depth_avg = depth_sum / valid_sum
    depth_mat[hole_mat == 1] = depth_avg[hole_mat == 1]
    return depth_mat.astype(np.float32)


def depth2xy(depth_mat, par_m, par_d, mask_mat=None):
    """Calculate xy_pro_coord given depth and parameters.

    Param:
        :param depth_mat:   The depth of view point.
        :param par_m:   The output of epi info.
        :param par_d:   The output of epi info.
        :param mask_mat:    The mask for valid points.

    Return:
        :return: xy_coord. Shape [H, W, 2]
    """
    img_shape = depth_mat.shape[-2:]
    xy_coord = np.zeros((img_shape[0], img_shape[1], 2))
    tmp_mat = par_m[:, :, 2] * depth_mat + par_d[2]
    xy_coord[:, :, 0] = (par_m[:, :, 0] * depth_mat + par_d[0]) / tmp_mat
    xy_coord[:, :, 1] = (par_m[:, :, 1] * depth_mat + par_d[1]) / tmp_mat
    if mask_mat is not None:
        xy_coord[mask_mat == 0] = 0
    return xy_coord


def xy2depth(xy_coord, par_m, par_d, mask_mat=None, main_dim=0):
    """Calculate xy_pro_coord given depth and parameters.

        Param:
            :param xy_coord:   The coord of new view, in shape of camera.
            :param par_m:   The output of epi info.
            :param par_d:   The output of epi info.
            :param mask_mat:    The mask for valid points.
            :param main_dim:    Which dim will be used for depth calculation.

        Return:
            :return: depth
    """
    tmp_mat = par_d[main_dim] - par_d[2] * xy_coord[:, :, main_dim]
    depth_map = - tmp_mat / (par_m[:, :, main_dim] - par_m[:, :, 2] * xy_coord[:, :, main_dim])
    if mask_mat is not None:
        depth_map[mask_mat == 0] = 0
    return depth_map


def norm_from_depth(depth_mat, mask_mat, m_per_pix):
    """Calculate norm map from given depth map.

    Use (-dz/dx, -dz/dy, 1). Use (x,y) (x+1, y), (x+1, y+1), (x, y+1).

    Args:
        :param depth_mat: view point from depth_mat. Depth is in meter.
        :param mask_mat: mask mat of depth_mat
        :param m_per_pix: How many pixels contained per meter.

    Returns:
        :return: norm_mat, mask_norm

    Raises:
        None.
    """

    # Calculate mask_norm
    mask_xy = mask_mat
    mask_x1y = np.zeros(mask_mat.shape, mask_mat.dtype)
    mask_x1y[1:, :] = mask_xy[:-1, :]
    mask_xy1 = np.zeros(mask_mat.shape, mask_mat.dtype)
    mask_xy1[:, 1:] = mask_xy[:, :-1]
    mask_norm = np.logical_and(mask_xy, mask_x1y)
    mask_norm = np.logical_and(mask_xy1, mask_norm)
    # mask_norm[mask_norm < 3] = 0
    # mask_norm[mask_norm > 0] = 1

    # Calculate depth_derv_x/y
    # Apply gaussian filter to depth map
    depth_xy = cv2.GaussianBlur(depth_mat, (9, 9), 3.0)
    # depth_xy = depth_mat
    depth_x1y = np.zeros(depth_xy.shape, depth_xy.dtype)
    depth_x1y[1:, :] = depth_xy[:-1, :]
    depth_xy1 = np.zeros(depth_xy.shape, depth_xy.dtype)
    depth_xy1[:, 1:] = depth_xy[:, :-1]
    depth_derv_x = depth_x1y - depth_xy
    depth_derv_y = depth_xy1 - depth_xy

    # Calculate norm_mat
    norm_mat = np.ones([depth_xy.shape[0], depth_xy.shape[1], 3], np.float32)
    norm_mat[:, :, 0] = -depth_derv_x / m_per_pix
    norm_mat[:, :, 1] = -depth_derv_y / m_per_pix
    mod_mat = np.sqrt(np.sum(norm_mat*norm_mat, 2))
    norm_mat = norm_mat / mod_mat.reshape(norm_mat.shape[0], norm_mat.shape[1], 1)
    norm_mat[mask_norm == 0] = 0

    return norm_mat, mask_norm


def shade_from_norm(norm_mat, rotation, mask_mat=None):
    """Calculate shade map from normal map.

    Param:
        :param norm_mat:    The normal of camera image.
        :param rotation:    Rotation from view to light source. Used for shade calculation.
        :param mask_mat:    A mask mat for valid points.

    Return:
        :return: Shade map. Light is projected from projector.

    Raise:
        None.
    """
    # Calculate pro_vec
    rot_mat = np.array(rotation).reshape(3, 3)
    front = np.array([[0.0], [0.0], [1.0]])
    pro_vec = np.matmul(rot_mat.transpose(), front)
    # Calculate shade map
    shade_map = np.dot(norm_mat, pro_vec)
    if mask_mat is not None:
        mask_mat[shade_map.reshape(mask_mat.shape) <= 0] = 0
        shade_map[mask_mat == 0] = 0.0
    return shade_map.astype(np.float32).reshape(shade_map.shape[:2])


def render_image(pattern, xy_pro, shade_mat, pattern_size, par_mp, par_dp, paras=None):
    """Rendering image by given parameters

    Param:
        :param pattern:     [N, C, Hc, Wc]. The pattern for sampling.
        :param xy_pro:      [N, 2, Hc, Wc]. The xy_pro coordinates for grid sample.
        :param shade_mat:   [N, 1, Hc, Wc]. The shade mat for normal estimation.
        :param pattern_size:Pattern pixel size.
        :param paras:       Parameters for rendering. Including:
            i_gn_rad:   The scale of image gaussian noise. Default 0 means no image noise.
            p_gn_rad:   The scale of pattern gaussian noise. Default 0 means no image noise.
            i_gb_rad:   The rad of gaussian blur kernel. Default 0 means no gaussian noise.
            i_gb_sig:   The sigma for gaussian blur kernel.
            p_gb_rad:
            p_gb_sig:

    Return:
        :return: image_cam, [N, C, H, W].
    """

    paras = {} if paras is None else paras
    paras['i_gn_rad'] = 0.0 if 'i_gn_rad' not in paras else paras['i_gn_rad']
    paras['p_gn_rad'] = 0.0 if 'p_gn_rad' not in paras else paras['p_gn_rad']
    paras['i_gb_rad'] = 0 if 'i_gb_rad' not in paras else paras['i_gb_rad']
    paras['i_gb_sig'] = 1.0 if 'i_gb_sig' not in paras else paras['i_gb_sig']
    paras['p_gb_rad'] = 0 if 'p_gb_rad' not in paras else paras['p_gb_rad']
    paras['p_gb_sig'] = 1.0 if 'p_gb_sig' not in paras else paras['p_gb_sig']

    # Parameters for rendering
    pro_shape = pattern.shape[-2:]
    cam_shape = xy_pro.shape[-2:]
    i_ker_len = paras['i_gb_rad'] * 2 + 1
    i_ker = torch.from_numpy(gkern(kernlen=i_ker_len, nsig=paras['i_gb_sig'])).reshape(1, 1, i_ker_len, i_ker_len)
    p_ker_len = paras['p_gb_rad'] * 2 + 1
    p_ker = torch.from_numpy(gkern(kernlen=p_ker_len, nsig=paras['p_gb_sig'])).reshape(1, 1, p_ker_len, p_ker_len)

    # Get pattern noise added on every pixel of pattern
    noise_shape = tuple(int(i / pattern_size) for i in pro_shape)
    pattern_noise = torch.randn(pattern.shape[:2] + noise_shape).cuda() * paras['p_gn_rad']
    pattern_noise_up = torch.nn.functional.interpolate(input=pattern_noise, scale_factor=pattern_size,
                                                       mode='nearest')
    pattern_add = pattern + pattern_noise_up
    pattern_add = torch.nn.functional.conv2d(pattern_add, p_ker, padding=paras['p_gb_rad'])

    # xy_pro to [-1, 1]
    xy_pro_grid = xy_pro.clone()
    xy_pro_grid[:, 0, :, :] = xy_pro_grid[:, 0, :, :] / (pro_shape[1] - 1) * 2.0 - 1.0
    xy_pro_grid[:, 1, :, :] = xy_pro_grid[:, 1, :, :] / (pro_shape[0] - 1) * 2.0 - 1.0
    xy_pro_grid = xy_pro_grid.permute(0, 2, 3, 1)
    # Get patterned image
    img_cam = torch.nn.functional.grid_sample(input=pattern_add, grid=xy_pro_grid.float())

    # shade_noise here
    shade_img = img_cam * shade_mat

    # image_noise here
    img_noise = torch.randn(cam_shape).cuda() * paras['i_gn_rad']
    img_cam = shade_img + img_noise

    # Intensity decay
    # par_mp_ten = torch.from_numpy(par_mp.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda()
    # m_val = torch.nn.functional.grid_sample(input=par_mp_ten, grid=xy_pro_grid.float()).squeeze()
    # d_val = par_dp
    # x_cam = torch.arange(0, cam_shape[1]).reshape(1, -1).repeat(cam_shape[0], 1).float().cuda()
    # depth_pro = -(d_val[0] - d_val[2] * x_cam) / (m_val[0] - m_val[2] * x_cam)
    # img_decay = float(1e-3) / depth_pro / depth_pro
    # img_decay = img_decay.unsqueeze(0).unsqueeze(1)
    # img_decay[shade_mat == 0] = 0
    # img_decay = img_decay.clamp(0, 1)
    # plt.imshow(img_decay.cpu().squeeze())
    # plt.show()
    # img_cam = img_cam * img_decay

    # Gaussian blur for image here
    img_cam = torch.nn.functional.conv2d(img_cam, i_ker, padding=paras['i_gb_rad'])
    img_cam = torch.clamp(img_cam, min=0, max=1)

    return img_cam


# ----------------------------------- #
# Visualization
# ----------------------------------- #
def coord_visualization(xy_mat, range_shape):
    show_mat = np.ones(xy_mat.shape[:2] + tuple([3]))
    show_mat[:, :, 0] = 1.0 - xy_mat[:, :, 0] / range_shape[1]
    show_mat[:, :, 1] = 1.0 - xy_mat[:, :, 1] / range_shape[0]
    return show_mat.astype(np.float32)
