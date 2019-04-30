import cv2
import numpy as np

# train_set = range(20, 200, 20)
# test_set = range(1, 21, 1)

# data_set = range(20, 60 + 1, 20)
# disp_fake_name = ['disp_fake%d.npy' % x for x in data_set]
# disp_real_name = ['disp_real%d.npy' % x for x in data_set]
# mask_name = ['mask_flow%d.npy' % x for x in data_set]

data_set = [1, 2, 3]
disp_fake_name = ['disp_fake_test%d.npy' % x for x in data_set]
disp_real_name = ['disp_real_test%d.npy' % x for x in data_set]
mask_name = ['mask_flow_test%d.npy' % x for x in data_set]

disp_fake_mats = []
disp_real_mats = []
mask_mats = []
for name in disp_fake_name:
    mat = np.load(name)
    disp_fake_mats.append(mat)
for name in disp_real_name:
    mat = np.load(name)
    disp_real_mats.append(mat)
for name in mask_name:
    mat = np.load(name)
    mask_mats.append(mat)
batch_num = len(mask_name)
batch_size = 4

# Check size for every image
h_max_size = 0
w_max_size = 0
for i in range(0, batch_num):
    mask_mat = mask_mats[i]
    for c in range(0, batch_size):
        h_valid, w_valid = mask_mat[c, 0, :, :].nonzero()
        h_len = h_valid.max() - h_valid.min()
        w_len = w_valid.max() - w_valid.min()
        h_max_size = max(h_len, h_max_size)
        w_max_size = max(w_len, w_max_size)
print(h_max_size, w_max_size)

half_rad = 500 // 2
image_size = half_rad*2 + 1
total_mat = np.zeros((image_size * batch_size, image_size * batch_num * 2), dtype=np.uint8)
period = 64
print(total_mat.shape)
for i in range(0, batch_num):
    for c in range(0, batch_size):
        disp_fake = disp_fake_mats[i][c, 0, :, :]
        disp_real = disp_real_mats[i][c, 0, :, :]
        mask_flow = mask_mats[i][c, 0, :, :]
        disp_fake[mask_flow == 0] = 0
        disp_real[mask_flow == 0] = 0
        # Get
        h_valid, w_valid = mask_mat[c, 0, :, :].nonzero()
        h_cen = (h_valid.max() + h_valid.min()) // 2
        w_cen = (w_valid.max() + w_valid.min()) // 2
        disp_fake_part = disp_fake[h_cen - half_rad:h_cen + half_rad + 1, w_cen - half_rad:w_cen + half_rad + 1]
        disp_real_part = disp_real[h_cen - half_rad:h_cen + half_rad + 1, w_cen - half_rad:w_cen + half_rad + 1]
        # mask_flow_part = mask_flow[h_cen - half_rad:h_cen + half_rad + 1, w_cen - half_rad:w_cen + half_rad + 1]
        # Get show value
        show_disp_fake = np.mod(disp_fake_part, period) / period * 255
        show_disp_real = np.mod(disp_real_part, period) / period * 255
        # Fill
        total_mat[c * image_size:(c + 1) * image_size, 2 * i * image_size:(2 * i + 1) * image_size] \
            = show_disp_fake.astype(np.uint8)
        total_mat[c * image_size:(c + 1) * image_size, (2 * i + 1) * image_size:(2 * i + 2) * image_size] \
            = show_disp_real.astype(np.uint8)
cv2.imwrite('test_set.png', total_mat)
