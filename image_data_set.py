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

    def __init__(self, root_dir, csv_name):
        self.root_dir = root_dir
        self.image_frame = pd.read_csv(root_dir + csv_name, header=None)
        self.image_list = []
        # Load image: range(0, 1000, 10)
        for idx in range(0, 1000, 10):
            img_name = ''.join([root_dir, 'DataSet1/cam_img', str(idx), '.png'])
            image = plt.imread(img_name)
            self.image_list.append(image)

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_idx = int(self.image_frame.iloc[idx][0] / 10)
        h_cen = int(self.image_frame.iloc[idx][1])
        w_cen = int(self.image_frame.iloc[idx][2])
        part_img = self.image_list[img_idx][h_cen-10:h_cen+11, w_cen-10:w_cen+11, :].copy()
        norm_part_img = torch.from_numpy((part_img.transpose((2, 0, 1)) - 0.5) * 2)
        x_pro = torch.tensor(self.image_frame.iloc[idx][3] / 1024.0).reshape((1, 1, 1))
        norm_x_pro = x_pro.clamp(0, 1)

        # x_pro = torch.from_numpy(np.loadtxt(pro_name)).unsqueeze(0)
        # norm_x_pro = (x_pro.clamp(0, 1024)) / 1024.0
        # print(torch.max(norm_x_pro), torch.min(norm_x_pro))
        sample = {'image': norm_part_img, 'x_pro': norm_x_pro.type(torch.FloatTensor)}
        return sample
