import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.data import Dataset


class CameraDataSet(Dataset):
    """My camera image dataset."""

    def __init__(self, root_dir, csv_name, down_k=4, disp_range=None, opts=None):
        if disp_range is None:
            disp_range = [716, 1724]
        if opts is None:
            opts = {'vol': False, 'disp_c': False, 'dense': False, 'stride': 1}
        self.root_dir = root_dir
        self.opt = opts
        if 'dense' not in self.opt.keys():
            self.opt['dense'] = False
        if 'vol' not in self.opt.keys():
            self.opt['vol'] = False
        if 'disp_c' not in self.opt.keys():
            self.opt['disp_c'] = False
        if 'stride' not in self.opt.keys():
            self.opt['stride'] = 1
        if 'shade' not in self.opt.keys():
            self.opt['shade'] = False
        if 'idx_vec' not in self.opt.keys():
            self.opt['idx_vec'] = False

        self.image_frame = []
        count = 0

        # Load csv
        with open(root_dir + csv_name + '.csv') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                count += 1
                if count % opts['stride'] == 0:
                    self.image_frame.append(row)

        # Load header
        self.header = np.load(root_dir + csv_name + '.npy').item()

        # Load para_M, para_D
        self.para_M = torch.from_numpy(np.load(root_dir + 'para_M.npy')).float()
        self.para_D = torch.from_numpy(np.load(root_dir + 'para_D.npy')).float()
        self.f_tvec_mul = 1185.92488

        self.pattern = plt.imread(root_dir + 'pattern_part0.png')
        self.H = 1024
        self.W = 1280
        self.H_p = 128
        self.W_p = 1024
        self.K = down_k
        self.disp_range = disp_range
        self.D = 64
        self.Hc = int(self.H / pow(2, self.K))
        self.Wc = int(self.W / pow(2, self.K))

    def __len__(self):
        return len(self.image_frame)

    def get_pattern(self):
        return torch.from_numpy((self.pattern.transpose(2, 0, 1) - 0.5) * 2)

    def get_opt(self):
        return self.opt

    def get_path_by_name(self, name, idx):
        assert name in self.header
        item_idx = self.header[name]
        return ''.join((self.root_dir, self.image_frame[idx][0], self.image_frame[idx][item_idx]))

    def __getitem__(self, idx):
        sample = {'idx': idx}

        # Load image
        img_name = self.get_path_by_name(name='cam_img', idx=idx)
        image = plt.imread(img_name)
        # print('Img: ', image.shape)
        norm_image = torch.from_numpy((image.transpose((2, 0, 1)) - 0.5) * 2)
        sample['cam_img'] = norm_image

        # Load mask
        mask_name = self.get_path_by_name(name='mask_mat', idx=idx)
        mask = plt.imread(mask_name)
        norm_mask = torch.from_numpy(mask).byte().unsqueeze(0)
        sample['mask_mat'] = norm_mask

        # Load disp_set
        disp_name = self.get_path_by_name(name='disp_mat', idx=idx)
        disp_np_mat = np.fromfile(disp_name, dtype='<f4').reshape(self.W, self.H).transpose(1, 0)
        disp_np_mat = (disp_np_mat - self.disp_range[0]) / (self.disp_range[1] - self.disp_range[0])
        disp_np_mat = np.nan_to_num(disp_np_mat)
        norm_disp_mat = torch.from_numpy(disp_np_mat).unsqueeze(0)
        norm_disp_mat[norm_mask == 0] = 0
        sample['disp_mat'] = norm_disp_mat

        # Generate idx_mat
        if self.opt['idx_vec']:
            disp_raw = norm_disp_mat * (self.disp_range[1] - self.disp_range[0]) + self.disp_range[0]
            tmp_mat = (disp_raw * self.para_D[2] + self.para_M[2, :, :] * self.f_tvec_mul)
            x_pro_mat = (disp_raw * self.para_D[0] + self.para_M[0, :, :] * self.f_tvec_mul) / tmp_mat
            x_pro_mat = torch.remainder(torch.round(x_pro_mat), self.W_p).type(torch.LongTensor)
            y_pro_mat = (disp_raw * self.para_D[1] + self.para_M[1, :, :] * self.f_tvec_mul) / tmp_mat
            y_pro_mat = torch.remainder(torch.round(y_pro_mat), self.H_p).type(torch.LongTensor)
            idx_pro_mat = y_pro_mat * self.W_p + x_pro_mat
            idx_pro_mat[norm_mask == 0] = 0
            idx_pro_vec = idx_pro_mat.reshape(norm_image.shape[1] * norm_image.shape[2])
            assert torch.max(idx_pro_vec).item() <= self.H_p * self.W_p - 1 and torch.min(idx_pro_vec).item() >= 0
            sample['idx_vec'] = idx_pro_vec

        # Load shade_mat
        if self.opt['shade']:
            shade_name = self.get_path_by_name(name='shade_mat', idx=idx)
            shade_mat = plt.imread(shade_name)
            shade_mat = shade_mat[:, :, 1]
            norm_shade = torch.from_numpy(shade_mat).unsqueeze(0)
            sample['shade_mat'] = norm_shade

        # Load disp_out
        if self.opt['dense']:
            disp_out_name = self.get_path_by_name(name='disp_out', idx=idx)
            disp_out_np_mat = np.load(disp_out_name)  # In (0, 1], .npy
            disp_out_np_mat = np.nan_to_num(disp_out_np_mat)
            norm_disp_out_mat = torch.from_numpy(disp_out_np_mat).unsqueeze(0)  # [1, H, W]
            # norm_disp_out_mat[norm_mask == 0] = 0
            sample['disp_out'] = norm_disp_out_mat

        # Load img_est
        if self.opt['dense']:
            img_est_name = self.get_path_by_name(name='est_img', idx=idx)
            img_est = plt.imread(img_est_name)
            img_est = img_est[:, :, :3]
            # print('Est: ', img_est.shape)
            norm_est_img = torch.from_numpy((img_est.transpose((2, 0, 1)) - 0.5) * 2)
            sample['est_img'] = norm_est_img

        # Load mask_c
        if self.opt['disp_c'] or self.opt['vol']:
            mask_c_name = self.get_path_by_name(name='mask_c', idx=idx)
            mask_c = plt.imread(mask_c_name)
            mask_c = mask_c[:, :, 1]
            norm_mask_c = torch.from_numpy(mask_c).byte().unsqueeze(0)  # [1, Hc, Wc]
            sample['mask_c'] = norm_mask_c

        # Load disp_c
        if self.opt['disp_c']:
            disp_c_name = self.get_path_by_name(name='disp_c', idx=idx)
            disp_c_np_mat = np.load(disp_c_name)  # In (0, 1], .npy
            # disp_c_np_mat = np.fromfile(disp_c_name, dtype='<f4').reshape(self.Wc, self.Hc).transpose(1, 0)
            # disp_c_np_mat = (disp_c_np_mat - self.disp_range[0]) / (self.disp_range[1] - self.disp_range[0])
            disp_c_np_mat = np.nan_to_num(disp_c_np_mat)
            disp_c_mat = torch.from_numpy(disp_c_np_mat).unsqueeze(0)  # [1, Hc, Wc]
            # disp_c_mat[sample['mask_c'] == 0] = 0
            sample['disp_c'] = disp_c_mat

        # Load disp_v
        if self.opt['vol']:
            disp_v_name = self.get_path_by_name(name='disp_v', idx=idx)
            disp_v_np_mat = np.fromfile(disp_v_name, dtype='<u1').reshape(self.D, self.Wc, self.Hc).transpose([0, 2, 1])
            disp_v_np_mat = np.nan_to_num(disp_v_np_mat)
            disp_v_mat = torch.from_numpy(disp_v_np_mat).type(torch.FloatTensor)  # [D, Hc, Wc]
            sum_disp_v_mat = torch.sum(input=disp_v_mat, dim=0, keepdim=True)
            norm_v_mat = disp_v_mat / sum_disp_v_mat
            norm_v_mat[torch.isnan(norm_v_mat)] = 0
            sample['disp_v'] = norm_v_mat

        return sample
