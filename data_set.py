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
            opts = {'vol': False, 'dense': False}
        self.root_dir = root_dir

        self.image_frame = []
        count = 0
        with open(root_dir + csv_name) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                count += 1
                if count % 20 == 0:
                    self.image_frame.append(row)
        self.pattern = plt.imread(root_dir + 'pattern_part0.png')
        self.H = 1024
        self.W = 1280
        self.K = down_k
        self.disp_range = disp_range
        self.D = 64
        self.Hc = int(self.H / pow(2, self.K))
        self.Wc = int(self.W / pow(2, self.K))
        self.opt = opts
        if 'dense' not in self.opt.keys():
            self.opt['dense'] = False
        if 'vol' not in self.opt.keys():
            self.opt['vol'] = False

    def __len__(self):
        return len(self.image_frame)

    def get_pattern(self):
        return torch.from_numpy((self.pattern.transpose(2, 0, 1) - 0.5) * 2)

    def get_opt(self):
        return self.opt

    def get_disp_out_path(self, idx):
        return ''.join((self.root_dir, self.image_frame[idx][0], self.image_frame[idx][7]))

    def get_img_est_path(self, idx):
        return ''.join((self.root_dir, self.image_frame[idx][0], self.image_frame[idx][8]))

    def __getitem__(self, idx):
        # Load image
        img_name = self.root_dir + self.image_frame[idx][0] + self.image_frame[idx][1]
        image = plt.imread(img_name)
        # print('Img: ', image.shape)
        norm_image = torch.from_numpy((image.transpose((2, 0, 1)) - 0.5) * 2)

        # Load mask
        mask_name = self.root_dir + self.image_frame[idx][0] + self.image_frame[idx][3]
        mask = plt.imread(mask_name)
        norm_mask = torch.from_numpy(mask).byte().unsqueeze(0)

        # Load disp_set
        disp_name = self.root_dir + self.image_frame[idx][0] + self.image_frame[idx][2]
        disp_np_mat = np.fromfile(disp_name, dtype='<f4').reshape(self.W, self.H).transpose(1, 0)
        disp_np_mat = (disp_np_mat - self.disp_range[0]) / (self.disp_range[1] - self.disp_range[0])
        disp_np_mat = np.nan_to_num(disp_np_mat)
        norm_disp_mat = torch.from_numpy(disp_np_mat).unsqueeze(0)
        norm_disp_mat[norm_mask == 0] = 0

        sample = {'idx': idx, 'image': norm_image, 'disp': norm_disp_mat, 'mask': norm_mask}

        # Load disp_out
        if self.opt['dense']:
            disp_out_name = self.get_disp_out_path(idx)
            disp_out_np_mat = np.load(disp_out_name)  # In (0, 1], .npy
            disp_out_np_mat = np.nan_to_num(disp_out_np_mat)
            norm_disp_out_mat = torch.from_numpy(disp_out_np_mat).unsqueeze(0)  # [1, H, W]
            norm_disp_out_mat[norm_mask == 0] = 0
            sample['disp_out'] = norm_disp_out_mat

        # Load img_est
        if self.opt['dense']:
            img_est_name = self.get_img_est_path(idx)
            img_est = plt.imread(img_est_name)
            img_est = img_est[:, :, :3]
            # print('Est: ', img_est.shape)
            norm_est_img = torch.from_numpy((img_est.transpose((2, 0, 1)) - 0.5) * 2)
            sample['img_est'] = norm_est_img

        # Load mask_c
        if not self.opt['dense']:
            mask_c_name = self.root_dir + self.image_frame[idx][0] + self.image_frame[idx][6]
            mask_c = plt.imread(mask_c_name)
            norm_mask_c = torch.from_numpy(mask_c).byte().unsqueeze(0)  # [1, Hc, Wc]
            sample['mask_c'] = norm_mask_c

        # Load disp_c
        if not self.opt['dense'] and not self.opt['vol']:
            disp_c_name = self.root_dir + self.image_frame[idx][0] + self.image_frame[idx][4]
            disp_c_np_mat = np.fromfile(disp_c_name, dtype='<f4').reshape(self.Wc, self.Hc).transpose(1, 0)
            disp_c_np_mat = (disp_c_np_mat - self.disp_range[0]) / (self.disp_range[1] - self.disp_range[0])
            disp_c_np_mat = np.nan_to_num(disp_c_np_mat)
            disp_c_mat = torch.from_numpy(disp_c_np_mat).unsqueeze(0)  # [1, Hc, Wc]
            disp_c_mat[sample['mask_c'] == 0] = 0
            sample['disp_c'] = disp_c_mat

        # Load disp_v
        if not self.opt['dense'] and self.opt['vol']:
            disp_v_name = self.root_dir + self.image_frame[idx][0] + self.image_frame[idx][5]
            disp_v_np_mat = np.fromfile(disp_v_name, dtype='<u1').reshape(self.D, self.Wc, self.Hc).transpose([0, 2, 1])
            disp_v_np_mat = np.nan_to_num(disp_v_np_mat)
            disp_v_mat = torch.from_numpy(disp_v_np_mat).type(torch.FloatTensor)  # [D, Hc, Wc]
            sum_disp_v_mat = torch.sum(input=disp_v_mat, dim=0, keepdim=True)
            norm_v_mat = disp_v_mat / sum_disp_v_mat
            norm_v_mat[torch.isnan(norm_v_mat)] = 0
            sample['disp_v'] = norm_v_mat

        return sample
