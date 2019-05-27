import os
import torch
import configparser
import numpy as np
import DataProcess.util as dp
from matplotlib import pyplot as plt

# input_folder = '/media/qiao/Data/SLDataSet/20190516/3/dyna/'
# output_folder = '/media/qiao/Data/SLDataSet/20190516/3/img_pair/'
#
# start_num = 60
# stride = 5
# end_num = 1000
#
# if not os.path.exists(output_folder):
#     os.mkdir(output_folder)
#
# for i in range(start_num, end_num, stride):
#     first_in = 'dyna_mat%d.png' % i
#     first_out = 'cam_img%d_1.png' % i
#     if i + stride >= end_num:
#         continue
#     second_in = 'dyna_mat%d.png' % (i + stride)
#     second_out = 'cam_img%d_2.png' % i
#     os.system('cp %s %s' % (input_folder + first_in, output_folder + first_out))
#     os.system('cp %s %s' % (input_folder + second_in, output_folder + second_out))


pro_folder = '/media/qiao/Data/SLDataSet/20190516/3/pro/'
calib_name = '/media/qiao/Data/SLDataSet/20190516/calib.ini'
pattern_path = '/media/qiao/Data/SLDataSet/20190516/'

cfg = configparser.ConfigParser()
cfg.read(calib_name)

# Get parameters
paras = dp.process_epi_info(cam_intrinsic=[float(x) for x in cfg.get('Intrinsic', 'camera').split(',')],
                            pro_intrinsic=[float(x) for x in cfg.get('Intrinsic', 'projector').split(',')],
                            rotation=[float(x) for x in cfg.get('Extrinsic', 'rotation').split(',')],
                            transition=[float(x) for x in cfg.get('Extrinsic', 'transition').split(',')],
                            scale=1e-3)
par_mcp, par_dcp, par_mpc, par_dpc = paras

# Load x,y datas
x_pro = dp.load_bin_file(pro_folder + 'x_pro0.bin', [1024, 1280], row_major=False)
x_pro[x_pro < 0] = 0
y_pro = dp.load_bin_file(pro_folder + 'y_pro0.bin', [1024, 1280], row_major=False)
y_pro[y_pro < 0] = 0
mask_mat = dp.load_as_torch('mask', '.png', pro_folder, '').squeeze().cpu().numpy()
hole_mat = np.zeros_like(mask_mat)
hole_mat[x_pro == 0] = 1
hole_mat[y_pro == 0] = 1
hole_mat[mask_mat == 0] = 0
xy_pro = np.array([x_pro, y_pro]).transpose(1, 2, 0)

# Calculate depth
depth_map = dp.xy2depth(xy_coord=xy_pro, par_m=par_mcp, par_d=par_dcp, mask_mat=mask_mat, main_dim=1)
depth_map = dp.fill_depth_mat(depth_mat=depth_map, mask_mat=mask_mat, hole_mat=hole_mat)
xy_pro_tmp = dp.depth2xy(depth_mat=depth_map, par_m=par_mcp, par_d=par_dcp, mask_mat=mask_mat)
xy_pro[:, :, 1] = xy_pro_tmp[:, :, 1]
depth_map = dp.xy2depth(xy_coord=xy_pro, par_m=par_mcp, par_d=par_dcp, mask_mat=mask_mat, main_dim=0)
depth_map = dp.fill_depth_mat(depth_mat=depth_map, mask_mat=mask_mat, hole_mat=hole_mat)
xy_pro_tmp = dp.depth2xy(depth_mat=depth_map, par_m=par_mcp, par_d=par_dcp, mask_mat=mask_mat)
xy_pro[:, :, 0] = xy_pro_tmp[:, :, 0]
# depth_map = dp.xy2depth(xy_coord=xy_pro, par_m=par_mcp, par_d=par_dcp, mask_mat=mask_mat, main_dim=1)
# depth_map = dp.fill_depth_mat(depth_mat=depth_map, mask_mat=mask_mat, hole_mat=hole_mat)
# xy_pro_tmp = dp.depth2xy(depth_mat=depth_map, par_m=par_mcp, par_d=par_dcp, mask_mat=mask_mat)
# xy_pro[:, :, 1] = xy_pro_tmp[:, :, 1]

# plt.imshow(xy_pro[:, :, 0])
# plt.show()

# Calculate shade
norm_map, mask_mat = dp.norm_from_depth(depth_mat=depth_map, mask_mat=mask_mat,
                                        m_per_pix=cfg.getfloat('Extrinsic', 'm_per_pix'))
shade_map = dp.shade_from_norm(norm_mat=norm_map,
                               rotation=[float(x) for x in cfg.get('Extrinsic', 'rotation').split(',')],
                               mask_mat=mask_mat)
dp.save_from_torch(torch.from_numpy(shade_map), name='shade', suffix='.png', out_path=pro_folder, prefix='img')

# Render image
pattern = dp.load_as_torch('pattern0', '.png', pattern_path, '4pix_')
pattern = pattern * 0.95 + 0.05
img = dp.render_image(pattern=pattern, xy_pro=torch.from_numpy(xy_pro).permute(2, 0, 1).unsqueeze(0).cuda(),
                      shade_mat=torch.from_numpy(shade_map).unsqueeze(0).unsqueeze(1).cuda(),
                      par_mp=par_mpc, par_dp=par_dpc,
                      pattern_size=4, p_noise_rad=0.0, i_noise_rad=0.0, i_decay=1e2, gker_rad=0)
img[torch.from_numpy(mask_mat.astype(np.float32)).unsqueeze(0).unsqueeze(1) == 0] = 0
dp.save_from_torch(mat=img, name='render', suffix='.png', out_path=pro_folder, prefix='img')
