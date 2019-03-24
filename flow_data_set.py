import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from torch.utils.data import Dataset
import collections


class SampleSet(collections.MutableMapping):
    """A dictionary sample"""
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
    """My camera image dataset."""

    def __init__(self, root_dir, list_name, batch_size=4, jump_k=4, opts=None):

        self.opt = opts
        if opts is None:
            self.opt = {}
        if 'header' not in self.opt.keys():
            self.opt['header'] = tuple()
        if 'stride' not in self.opt.keys():
            self.opt['stride'] = 1
        if 'bias' not in self.opt.keys():
            self.opt['bias'] = 0
        # if 'disp_range' not in self.opt.keys():
        #     self.opt['disp_range'] = (1010, 1640)

        self.root_dir = root_dir
        # self.alpha = self.opt['disp_range'][1] - self.opt['disp_range'][0]
        # self.beta = self.opt['disp_range'][0]

        # Load csv
        raw_list = []
        with open(root_dir + list_name + '.csv') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                raw_list.append(row)
        self.image_frame = []
        self.image_frame_dest = []
        for i in range(0, len(raw_list)):
            if i % self.opt['stride'] == self.opt['bias'] and i + jump_k < len(raw_list):
                self.image_frame.append(raw_list[i])
                self.image_frame_dest.append(raw_list[i + jump_k])

        # Load header
        self.header = np.load(root_dir + list_name + '.npy').item()

        # Load para_M, para_D
        self.para_M = torch.from_numpy(np.load(root_dir + 'para_M.npy')).float()
        self.para_D = torch.from_numpy(np.load(root_dir + 'para_D.npy')).float()
        self.f_tvec_mul = 1185.92488

        # self.pattern = plt.imread(root_dir + 'pattern_part0.png')
        self.H = 1024
        self.W = 1280
        self.H_p = 128
        self.W_p = 1024
        self.N = batch_size
        # self.D = 64
        # self.Hc = int(self.H / pow(2, self.K))
        # self.Wc = int(self.W / pow(2, self.K))

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
        if info.get('next'):
            full_name = self.get_path_by_name(name, idx, next_table=True)
            name += info.get('next')
        # Check suffix
        suffix = full_name[full_name.rfind('.'):]
        if suffix == '.png':
            item = torch.from_numpy(plt.imread(full_name))
            # Shape
            if len(item.shape) == 2:
                item = item.unsqueeze(0)
            else:
                item = item.permute(2, 0, 1)
            if info.get('type') == 'byte':
                item = item.byte()
                if item.shape[0] > 1:
                    item = item[0, :, :].unsqueeze(0)
            else:
                item = item.float()
                item = item * 2.0 - 1.0
        elif suffix == '.npy':
            item = torch.from_numpy(np.load(full_name))
            if len(item.shape) == 2:
                item = item.unsqueeze(0)
            if info.get('range'):
                min_val, max_val = info['range']
                alpha, beta = min_val, max_val - min_val
                item = (item - beta) / alpha
            item[torch.isnan(item)] = 0
        elif suffix == '.bin':
            item_vec = np.fromfile(full_name, dtype='<f4')
            if not info.get('shape'):
                return False
            item = torch.from_numpy(item_vec.reshape(info['shape'][1], info['shape'][0])).permute(1, 0)
            item = item.unsqueeze(0)
            if info.get('range'):
                min_val, max_val = info['range']
                alpha, beta = min_val, max_val - min_val
                item = (item - beta) / alpha
            item[torch.isnan(item)] = 0
        else:
            return False

        sample[name] = item
        return True

    def get_opt(self):
        return self.opt

    def get_path_by_name(self, name, idx, next_table=False):
        assert name in self.header
        item_idx = self.header[name]
        if next_table:
            return ''.join((self.root_dir, self.image_frame_dest[idx][0], self.image_frame_dest[idx][item_idx]))
        else:
            return ''.join((self.root_dir, self.image_frame[idx][0], self.image_frame[idx][item_idx]))

    def __getitem__(self, idx):
        sample = SampleSet()
        sample['idx'] = torch.tensor([idx])

        # Load everything
        if 'cam_img' in self.opt['header']:
            assert self.load_item(sample, 'cam_img', idx, info={'type': 'float'})
        if 'disp_mat' in self.opt['header']:
            assert self.load_item(sample, 'disp_mat', idx, info={'shape': [self.H, self.W]})
        if 'disp_cam' in self.opt['header']:
            assert self.load_item(sample, 'disp_cam', idx, info={})
        if 'disp_cam_t' in self.opt['header']:
            assert self.load_item(sample, 'disp_cam', idx, info={'next': '_t'})
        if 'mask_mat' in self.opt['header']:
            assert self.load_item(sample, 'mask_mat', idx, info={'type': 'byte'})
        if 'mask_cam' in self.opt['header']:
            assert self.load_item(sample, 'mask_cam', idx, info={'type': 'byte'})
        if 'cor_xc' in self.opt['header']:
            assert self.load_item(sample, 'cor_xc', idx, info={})
        if 'cor_xc_t' in self.opt['header']:
            assert self.load_item(sample, 'cor_xc', idx, info={'next': '_t'})  # Only next
        if 'cor_yc' in self.opt['header']:
            assert self.load_item(sample, 'cor_yc', idx, info={})
        if 'cor_yc_t' in self.opt['header']:
            assert self.load_item(sample, 'cor_yc', idx, info={'next': '_t'})  # Only next
        if 'mask_pro' in self.opt['header']:
            assert self.load_item(sample, 'mask_pro', idx, info={'type': 'byte'})

        # Apply mask
        if 'mask_mat' in sample and 'disp_mat' in sample:
            sample['disp_mat'][sample['mask_mat'] == 0] = 0
        if 'mask_cam' in sample and 'disp_cam' in sample:
            sample['disp_cam'][sample['mask_cam'] == 0] = 0
        if 'mask_pro' in sample and 'cor_xc' in sample:
            sample['cor_xc'][sample['mask_pro'] == 0] = 0
        if 'mask_pro' in sample and 'cor_yc' in sample:
            sample['cor_yc'][sample['mask_pro'] == 0] = 0

        return sample

        # Load mask
        # if 'mask_mat_dest' in self.opt['header']:
        #     mask_name = self.get_path_by_name(name='mask_mat', idx=idx, next_table=True)
        #     mask = plt.imread(mask_name)
        #     norm_mask = torch.from_numpy(mask).byte().unsqueeze(0)
        #     sample['mask_mat_dest'] = norm_mask

        # Load disp_mat (Need norm_mask)
        # if 'disp_mat' in self.opt['header']:
        #     disp_name = self.get_path_by_name(name='disp_mat', idx=idx)
        #     disp_np_mat = np.fromfile(disp_name, dtype='<f4').reshape(self.W, self.H).transpose(1, 0)
        #     disp_np_mat = (disp_np_mat - self.opt['disp_range'][0]) / (
        #                 self.opt['disp_range'][1] - self.opt['disp_range'][0])
        #     disp_np_mat = np.nan_to_num(disp_np_mat)
        #     norm_disp_mat = torch.from_numpy(disp_np_mat).unsqueeze(0)
        #     norm_mask = sample['mask_mat']
        #     norm_disp_mat[norm_mask == 0] = 0
        #     sample['disp_mat'] = norm_disp_mat

        # if 'disp_cam' in self.opt['header']:
        #     disp_cam_name = self.get_path_by_name(name='disp_cam', idx=idx)
        #     disp_np_cam = np.load(disp_cam_name)
        #     disp_np_cam = (disp_np_cam - self.beta) / self.alpha
        #     disp_np_cam = np.nan_to_num(disp_np_cam)
        #     norm_disp_cam = torch.from_numpy(disp_np_cam).unsqueeze(0)
        #     sample['disp_cam'] = norm_disp_cam

        # Load disp_mat_dest
        # if 'disp_mat_dest' in self.opt['header']:
        #     disp_name = self.get_path_by_name(name='disp_mat', idx=idx, next_table=True)
        #     disp_np_mat = np.fromfile(disp_name, dtype='<f4').reshape(self.W, self.H).transpose((1, 0))
        #     disp_np_mat = (disp_np_mat - self.opt['disp_range'][0]) / (
        #                 self.opt['disp_range'][1] - self.opt['disp_range'][0])
        #     disp_np_mat = np.nan_to_num(disp_np_mat)
        #     norm_disp_mat = torch.from_numpy(disp_np_mat).unsqueeze(0)
        #     norm_mask = sample['mask_mat_dest']
        #     norm_disp_mat[norm_mask == 0] = 0
        #     sample['disp_mat_dest'] = norm_disp_mat

        # Load cor_pro (Only next)
        # if 'cor_pro' in self.opt['header']:
        #     cor_name = self.get_path_by_name(name='cor_pro', idx=idx, next_table=True)
        #     cor_np_mat = np.fromfile(cor_name, dtype='<u2').reshape(2, self.W, self.H).transpose((0, 2, 1))
        #     cor_np_mat = np.nan_to_num(cor_np_mat)
        #     cor_pro_mat = torch.from_numpy(cor_np_mat)
        #     sample['cor_pro'] = cor_pro_mat

        # Load mask_flow
        # if 'mask_flow' in self.opt['header']:
        #     mask_flow_name = self.get_path_by_name(name='mask_flow', idx=idx)
        #     mask_flow = plt.imread(mask_flow_name)
        #     norm_mask_flow = torch.from_numpy(mask_flow).byte().unsqueeze(0)
        #     sample['mask_flow'] = norm_mask_flow

        # Load flow_mat
        # if 'flow_mat' in self.opt['header']:
        #     flow_name = self.get_path_by_name(name='flow_mat', idx=idx)
        #     flow_np_mat = np.load(flow_name)  # .npy
        #     flow_np_mat = np.nan_to_num(flow_np_mat)
        #     flow_mat = torch.from_numpy(flow_np_mat).transpose((2, 0, 1))  # [2, H, W]
        #     sample['flow_mat'] = flow_mat

        # Generate idx_vec (Need norm_disp)
        # if 'idx_vec' in self.opt['header']:
        #     norm_disp_mat = sample['disp_mat']
        #     disp_raw = norm_disp_mat * (self.opt['disp_range'][1]
        #                                 - self.opt['disp_range'][0]) + self.opt['disp_range'][0]
        #     tmp_mat = (disp_raw * self.para_D[2] + self.para_M[2, :, :] * self.f_tvec_mul)
        #     x_pro_mat = (disp_raw * self.para_D[0] + self.para_M[0, :, :] * self.f_tvec_mul) / tmp_mat
        #     x_pro_mat = torch.remainder(torch.round(x_pro_mat), self.W_p).type(torch.LongTensor)
        #     y_pro_mat = (disp_raw * self.para_D[1] + self.para_M[1, :, :] * self.f_tvec_mul) / tmp_mat
        #     y_pro_mat = torch.remainder(torch.round(y_pro_mat), self.H_p).type(torch.LongTensor)
        #     idx_pro_mat = y_pro_mat * self.W_p + x_pro_mat
        #     idx_pro_mat[norm_mask == 0] = 0
        #     idx_pro_vec = idx_pro_mat.reshape(norm_mask.shape[1] * norm_mask.shape[2])
        #     assert torch.max(idx_pro_vec).item() <= self.H_p * self.W_p - 1 and torch.min(idx_pro_vec).item() >= 0
        #     sample['idx_vec'] = idx_pro_vec

        # Load shade_mat
        # if 'shade_mat' in self.opt['header']:
        #     shade_name = self.get_path_by_name(name='shade_mat', idx=idx)
        #     shade_mat = plt.imread(shade_name)
        #     shade_mat = shade_mat[:, :, 1]
        #     norm_shade = torch.from_numpy(shade_mat).unsqueeze(0)
        #     sample['shade_mat'] = norm_shade

        # Load mask_c
        # if 'mask_c' in self.opt['header']:
        #     mask_c_name = self.get_path_by_name(name='mask_c', idx=idx)
        #     mask_c = plt.imread(mask_c_name)
        #     mask_c = mask_c[:, :, 1]
        #     norm_mask_c = torch.from_numpy(mask_c).byte().unsqueeze(0)  # [1, Hc, Wc]
        #     sample['mask_c'] = norm_mask_c

        # Load disp_c
        # if 'disp_c' in self.opt['header']:
        #     disp_c_name = self.get_path_by_name(name='disp_c', idx=idx)
        #     disp_c_np_mat = np.load(disp_c_name)  # In (0, 1], .npy
        #     # disp_c_np_mat = np.fromfile(disp_c_name, dtype='<f4').reshape(self.Wc, self.Hc).transpose(1, 0)
        #     # disp_c_np_mat = (disp_c_np_mat - self.opt['disp_range'][0])
        #     #                / (self.opt['disp_range'][1] - self.opt['disp_range'][0])
        #     disp_c_np_mat = np.nan_to_num(disp_c_np_mat)
        #     disp_c_mat = torch.from_numpy(disp_c_np_mat).unsqueeze(0)  # [1, Hc, Wc]
        #     # disp_c_mat[sample['mask_c'] == 0] = 0
        #     sample['disp_c'] = disp_c_mat

        # Load disp_v
        # if 'disp_v' in self.opt['header']:
        #     disp_v_name = self.get_path_by_name(name='disp_v', idx=idx)
        #     disp_v_np_mat = np.fromfile(disp_v_name, dtype='<u1').reshape(self.D, self.Wc, self.Hc).transpose(
        #         [0, 2, 1])
        #     disp_v_np_mat = np.nan_to_num(disp_v_np_mat)
        #     disp_v_mat = torch.from_numpy(disp_v_np_mat).type(torch.FloatTensor)  # [D, Hc, Wc]
        #     sum_disp_v_mat = torch.sum(input=disp_v_mat, dim=0, keepdim=True)
        #     norm_v_mat = disp_v_mat / sum_disp_v_mat
        #     norm_v_mat[torch.isnan(norm_v_mat)] = 0
        #     sample['disp_v'] = norm_v_mat

        # Load disp_out
        # if 'disp_out' in self.opt['header']:
        #     disp_out_name = self.get_path_by_name(name='disp_out', idx=idx)
        #     disp_out_np_mat = np.load(disp_out_name)  # In (0, 1], .npy
        #     disp_out_np_mat = np.nan_to_num(disp_out_np_mat)
        #     norm_disp_out_mat = torch.from_numpy(disp_out_np_mat).unsqueeze(0)  # [1, H, W]
        #     # norm_disp_out_mat[norm_mask == 0] = 0
        #     sample['disp_out'] = norm_disp_out_mat

        # Load img_est
        # if 'est_img' in self.opt['header']:
        #     img_est_name = self.get_path_by_name(name='est_img', idx=idx)
        #     img_est = plt.imread(img_est_name)
        #     img_est = img_est[:, :, :3]
        #     # print('Est: ', img_est.shape)
        #     norm_est_img = torch.from_numpy((img_est.transpose((2, 0, 1)) - 0.5) * 2)
        #     sample['est_img'] = norm_est_img
