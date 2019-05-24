# TODO: Add note in this script.

import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from torch.utils.data import Dataset
import collections


class SampleSet(collections.MutableMapping):
    """A dictionary sample
        This is a dictionary class.
        Used for transform everything into cuda or cpu.
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def to_cuda(self):
        for key in self:
            self[key] = self[key].cuda()

    def to_cpu(self):
        for key in self:
            self[key] = self[key].cpu()

    def __str__(self):
        return '\n'.join(['%s: %s' % (key, str(self[key].shape)) for key in self])


class FlowDataSet(Dataset):
    """My camera image dataset.

    The types of data will be loaded:
        _depth_cam.npy      [H, W], float
        _mask_cam.png       [H, W], byte
        _flow1_cv.npy        [H, W, 2], float
        _mask_flow1.png      [H, W], byte
        _flow1_cv_est.npy   [H, W, 2], float
    """

    def __init__(self, root_dir, header_name, list_name, batch_size=4, opts=None):

        self.opt = opts
        if opts is None:
            self.opt = {}
        if 'header' not in self.opt.keys():
            self.opt['header'] = tuple()
        if 'stride' not in self.opt.keys():
            self.opt['stride'] = 2
        if 'bias' not in self.opt.keys():
            self.opt['bias'] = 0
        if 'depth_range' not in self.opt.keys():
            self.opt['depth_range'] = (0.0, 1.0)

        self.root_dir = root_dir
        self.header = np.load(root_dir + header_name, allow_pickle=True).item()

        # Load data.csv
        raw_list = []
        with open(root_dir + list_name) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                raw_list.append(row)
        self.image_frame = []
        for i in range(0, len(raw_list)):
            if i % self.opt['stride'] == self.opt['bias']:
                self.image_frame.append(raw_list[i])

        # self.H = 1024
        # self.W = 1280
        # self.H_p = 1024
        # self.W_p = 1280
        self.N = batch_size

    def __len__(self):
        return len(self.image_frame)

    def save_item(self, item, name, idx):
        full_name = self.get_path_by_name(name, idx)
        # Check and create folder
        folder_path = full_name[:full_name.rfind('/')]
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        # Write file according to suffix
        suffix = full_name[full_name.rfind('.'):]
        if suffix == '.png':
            np_item = item.cpu().squeeze().numpy()
            if item.shape[1] == 1:
                plt.imsave(full_name, np_item, cmap='Greys_r')
            elif item.shape[1] == 3:
                plt.imsave(full_name, np.transpose(np_item, (1, 2, 0)))
            else:
                return False
        elif suffix == '.npy':
            np_item = item.cpu().squeeze().numpy()
            np.save(full_name, np_item)
        else:
            return False
        return True

    def load_item(self, sample, name, idx, info=None):
        if info is None:
            info = {}
        full_name = self.get_path_by_name(name, idx)

        # Check suffix
        suffix = full_name[full_name.rfind('.'):]
        if suffix == '.png':
            item = torch.from_numpy(plt.imread(full_name))
            item = item[:, :, 0].unsqueeze(0)
            item = item.byte()
        elif suffix == '.npy':
            item = torch.from_numpy(np.load(full_name))
            if len(item.shape) == 2:
                item = item.unsqueeze(0)
            else:
                item = item.permute(2, 0, 1)
            item[torch.isnan(item)] = 0
        else:
            return False
        sample[name] = item
        return True

    def get_opt(self):
        return self.opt

    def get_path_by_name(self, name, idx):
        assert name in self.header
        item_idx = self.header[name]
        return ''.join((self.root_dir, self.image_frame[idx][0], self.image_frame[idx][item_idx]))

    def __getitem__(self, idx):
        sample = SampleSet()
        sample['idx'] = torch.tensor([idx])

        # Load everything
        if 'depth_cam' in self.opt['header']:
            assert self.load_item(sample, 'depth_cam', idx)
        if 'mask_cam' in self.opt['header']:
            assert self.load_item(sample, 'mask_cam', idx)
        if 'flow1_cv' in self.opt['header']:
            assert self.load_item(sample, 'flow1_cv', idx)
        if 'mask_flow1' in self.opt['header']:
            assert self.load_item(sample, 'mask_flow1', idx)
        if 'flow1_cv_est' in self.opt['header']:
            assert self.load_item(sample, 'flow1_cv_est', idx)
            tmp_mat = torch.nn.functional.interpolate(input=sample['flow1_cv_est'].unsqueeze(0), scale_factor=4.0,
                                                      mode='bilinear', align_corners=True)
            sample['flow1_cv_est'] = tmp_mat[0]

        # Apply normalization
        if 'depth_cam' in sample:
            alpha = self.opt['depth_range'][1] - self.opt['depth_range'][0]
            beta = self.opt['depth_range'][0]
            sample['depth_cam'] = (sample['depth_cam'] - beta) / alpha

        # Apply mask
        if 'mask_cam' in sample and 'depth_cam' in sample:
            sample['depth_cam'][sample['mask_cam'] == 0] = 0
        if 'mask_flow1' in sample and 'flow1_cv' in sample:
            sample['flow1_cv'][sample['mask_flow1'].repeat(2, 1, 1) == 0] = 0
        if 'mask_flow1' in sample and 'flow1_cv_est' in sample:
            sample['flow1_cv_est'][sample['mask_flow1'].repeat(2, 1, 1) == 0] = 0

        return sample
