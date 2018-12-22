import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, utils


def imshow(image):
    np_image = image.cpu().numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.savefig('input_image.png')
    plt.show()


class CameraDataSet(Dataset):
    """My camera image dataset."""

    def __init__(self, root_dir, csv_name, alpha=1024 - 16, beta=700 + 16, K=4):
        self.root_dir = root_dir
        self.image_frame = pd.read_csv(root_dir + csv_name, header=None)
        self.pattern = plt.imread(root_dir + 'pattern_part0.png')
        self.H = 1024
        self.W = 1280
        self.K = K
        self.D = 64
        self.alpha = alpha
        self.beta = beta
        self.Hc = int(self.H / pow(2, K))
        self.Wc = int(self.W / pow(2, K))

        # self.image_list = []
        # # Load image: range(0, 1000, 10)
        # for idx in range(0, 1000, 10):
        #     img_name = ''.join([root_dir, 'DataSet1/cam_img', str(idx), '.png'])
        #     image = plt.imread(img_name)
        #     self.image_list.append(image)

    def __len__(self):
        return len(self.image_frame)

    def GetPatternTensor(self):
        return torch.from_numpy((self.pattern.transpose(2, 0, 1) - 0.5) * 2)

    def __getitem__(self, idx):
        # Load image
        img_name = self.root_dir + self.image_frame.iloc[idx][0] + self.image_frame.iloc[idx][1]
        image = plt.imread(img_name)
        norm_image = torch.from_numpy((image.transpose((2, 0, 1)) - 0.5) * 2)

        disp_name = self.root_dir + self.image_frame.iloc[idx][0] + self.image_frame.iloc[idx][2]
        disp_np_mat = np.fromfile(disp_name, dtype='<f4').reshape(self.W, self.H).transpose(1, 0)
        disp_np_mat = np.nan_to_num(disp_np_mat)
        norm_disp_mat = torch.from_numpy(disp_np_mat).unsqueeze(0)
        # norm_disp_mat = disp_mat.clamp(1e-10, 1724.0)

        mask_name = self.root_dir + self.image_frame.iloc[idx][0] + self.image_frame.iloc[idx][3]
        mask = plt.imread(mask_name)
        norm_mask = torch.from_numpy(mask).unsqueeze(0)

        disp_c_name = self.root_dir + self.image_frame.iloc[idx][0] + self.image_frame.iloc[idx][4]
        disp_c_np_mat = np.fromfile(disp_c_name, dtype='<f4').reshape(self.Wc, self.Hc).transpose(1, 0)
        disp_c_np_mat = np.nan_to_num(disp_c_np_mat)
        disp_c_mat = torch.from_numpy(disp_c_np_mat).unsqueeze(0)  # [1, Hc, Wc]

        disp_v_name = self.root_dir + self.image_frame.iloc[idx][0] + self.image_frame.iloc[idx][5]
        disp_v_np_mat = np.fromfile(disp_v_name, dtype='<u1').reshape(self.D, self.Wc, self.Hc).transpose([0, 2, 1])
        disp_v_np_mat = np.nan_to_num(disp_v_np_mat)
        disp_v_mat = torch.from_numpy(disp_v_np_mat).type(torch.FloatTensor)  # [D, Hc, Wc]
        sum_disp_v_mat = torch.sum(input=disp_v_mat, dim=0, keepdim=True)
        norm_v_mat = disp_v_mat / sum_disp_v_mat
        norm_v_mat[torch.isnan(norm_v_mat)] = 0
        # disp_idx = torch.round((disp_c_mat - self.beta) / self.alpha * self.D)
        # disp_idx = disp_idx.clamp(0, self.D - 1).type(torch.LongTensor).unsqueeze(3)
        # set_ones = torch.ones([1, self.Hc, self.Wc, 1])
        # volume_set = torch.zeros([1, self.Hc, self.Wc, self.D]).scatter_(dim=3, index=disp_idx, src=set_ones)

        mask_c_name = self.root_dir + self.image_frame.iloc[idx][0] + self.image_frame.iloc[idx][6]
        mask_c = plt.imread(mask_c_name)
        norm_mask_c = torch.from_numpy(mask_c).byte().unsqueeze(0)  # [1, Hc, Wc]

        # sample = {'image': norm_image, 'disp': norm_disp_mat, 'mask': norm_mask, 'disp_c': disp_c_mat,
        #           'mask_c': norm_mask_c}
        sample = {'image': norm_image, 'disp': norm_disp_mat, 'mask': norm_mask, 'disp_v': norm_v_mat,
                  'mask_c': norm_mask_c}
        return sample
