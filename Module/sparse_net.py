import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from torch.nn import Sequential


def load_pro_mat(root_path, x_file_name, y_file_name, mat_size, width_pro):
    raw_input = torch.from_numpy(np.fromfile(root_path + x_file_name, dtype=np.uint8)).type(torch.LongTensor)
    x_pro_mat = raw_input.reshape([mat_size[2], mat_size[1], mat_size[0]]).transpose(0, 2)
    assert(list(x_pro_mat.shape) == mat_size)  # [H, W, D]
    raw_input = torch.from_numpy(np.fromfile(root_path + y_file_name, dtype=np.uint8)).type(torch.LongTensor)
    y_pro_mat = raw_input.reshape([mat_size[2], mat_size[1], mat_size[0]]).transpose(0, 2)
    assert(list(y_pro_mat.shape) == mat_size)  # [H, W, D]
    idx_mat = y_pro_mat * width_pro + x_pro_mat
    # print(x_pro_mat[15, 18, :])
    # print(y_pro_mat[15, 18, :])
    # print(idx_mat[15, 18, :])
    idx_pro_vector = idx_mat.reshape(mat_size[0] * mat_size[1] * mat_size[2])  # [H*W*D]
    assert torch.max(idx_pro_vector).item() <= 2047 and torch.min(idx_pro_vector).item() >= 0
    return idx_pro_vector.cuda()


def load_disp_volume(root_path, file_name, volume_size, batch_size):
    # raw_input = torch.from_numpy(np.fromfile(root_path + file_name, dtype='<f4'))
    raw_input = torch.Tensor(list(range(64, 0, -1)))
    raw_input = raw_input.float() / 64.0
    assert(raw_input.shape[0] == volume_size[2])  # [D]
    disp_volume = torch.stack([raw_input] * volume_size[0], 1)  # [D, H]
    disp_volume = torch.stack([disp_volume] * volume_size[1], 2)  # [D, H, W]
    disp_volume = torch.stack([disp_volume] * batch_size, 0)  # [N, D, H, W]
    # disp_volume = disp_volume.unsqueeze(1)
    return disp_volume.cuda()


class SparseNet(nn.Module):
    """
    The SparseNet is designed to predict sparse_disp from input image.
    Input: (RGB Image, Pattern)
        RGB Image: [N, C=3, H=1024, W=1280], range: [-1, 1]
        Pattern:   [N, C=3, H=128, W=1024], range: [-1, 1]
    Output: sparse or volume (depending on opts)
        sparse_disp: [N, C=1 or C=64, H_c=1024/2^K, W_c=1280/2^K], range: [0, 1]
    """

    def __init__(self, root_path, batch_size, down_k=4, opts=None):
        super(SparseNet, self).__init__()

        if opts is None:
            opts = {'vol': False}
        self.K = down_k
        self.N = batch_size
        self.H = int(1024 / pow(2, self.K))
        self.W = int(1280 / pow(2, self.K))
        self.D = 64
        self.Hp = int(128 / pow(2, self.K))
        self.Wp = int(1024 / pow(2, self.K))
        self.epsilon = 1e-10

        self.vol = False
        if 'vol' in opts.keys() and opts['vol'] is True:
            self.vol = True

        # Load parameters
        self.idx_pro_vec = load_pro_mat(root_path=root_path,
                                        x_file_name='x_pro_tensor_' + str(self.K) + '.bin',
                                        y_file_name='y_pro_tensor_' + str(self.K) + '.bin',
                                        mat_size=[self.H, self.W, self.D],
                                        width_pro=self.Wp)
        self.disp_mat = load_disp_volume(root_path=root_path, file_name='disp_range.bin',
                                         volume_size=[self.H, self.W, self.D], batch_size=self.N)

        # Pad layers:
        self.rep_pad_2 = nn.ReplicationPad2d(2)
        self.ref_pad_2 = nn.ReflectionPad2d(2)
        self.rep_pad_1 = nn.ReplicationPad2d(1)
        self.ref_pad_1 = nn.ReflectionPad2d(1)

        # Down sample layers
        self.dn_convs = []
        self.dn_self_convs = []
        dn_conv_planes = [3, 32, 32, 32, 32, 32]
        for i in range(0, self.K):
            tmp_conv = nn.Sequential(
                nn.Conv2d(dn_conv_planes[i], dn_conv_planes[i + 1], kernel_size=5, stride=2, padding=0),
                nn.ReLU(inplace=True)
            )
            tmp_self_conv = nn.Sequential(
                nn.Conv2d(dn_conv_planes[i + 1], dn_conv_planes[i + 1], kernel_size=3, padding=0),
                nn.ReLU(inplace=True)
            )
            self.dn_convs.append(tmp_conv)
            self.dn_self_convs.append(tmp_self_conv)
        self.dn_convs = nn.ModuleList(self.dn_convs)
        self.dn_self_convs = nn.ModuleList(self.dn_self_convs)

        # self_conv res
        self.res_convs = []
        res_conv_planes = [32, 32, 32, 32, 32, 32, 32]
        for i in range(0, 6):
            tmp_conv = nn.Sequential(
                nn.Conv2d(res_conv_planes[i], res_conv_planes[i + 1], kernel_size=3, padding=0),
                nn.BatchNorm2d(res_conv_planes[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.res_convs.append(tmp_conv)
        self.res_convs = nn.ModuleList(self.res_convs)
        self.self_out = nn.Conv2d(res_conv_planes[-1], 32, kernel_size=3, padding=0)

        # Volume conv3d layers
        if not self.vol:
            self.volume_convs = []
            for i in range(0, 4):
                tmp_conv = nn.Sequential(
                    nn.Conv3d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True)
                )
                self.volume_convs.append(tmp_conv)
            self.volume_convs = nn.ModuleList(self.volume_convs)
            self.volume_out = nn.Conv3d(32, 32, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def softmax_disp(self, volume_prob):
        sparse_disp = torch.sum(input=volume_prob * self.disp_mat, dim=1, keepdim=True)
        return sparse_disp

    def forward(self, x):

        # Image, pattern down sample:
        image_dn_conv_out = [x[0]]
        pattern_dn_conv_out = [x[1]]
        for i in range(0, self.K):
            img_padding = self.rep_pad_2(image_dn_conv_out[i])
            img_out_val = self.dn_convs[i](img_padding)
            img_padding = self.rep_pad_1(img_out_val)
            img_out_val = self.dn_self_convs[i](img_padding)
            image_dn_conv_out.append(img_out_val)
            # print('image_dn_conv_out[%d]: ' % i, torch.isnan(img_out_val).any().item())
            # print('image_dn_conv_out[%d]: ' % i, img_out_val.shape)

            pat_padding = self.ref_pad_2(pattern_dn_conv_out[i])
            pat_out_val = self.dn_convs[i](pat_padding)
            pat_padding = self.ref_pad_1(pat_out_val)
            pat_out_val = self.dn_self_convs[i](pat_padding)
            pattern_dn_conv_out.append(pat_out_val)
            # print('pattern_dn_conv_out[%d]: ' % i, torch.isnan(pat_out_val).any().item())
            # print('pattern_dn_conv_out[%d]: ' % i, pat_out_val.shape)

        # Image, pattern residual self conv
        img_feature_mat = image_dn_conv_out[-1]
        pat_feature_mat = pattern_dn_conv_out[-1]
        for i in range(0, 6):
            img_padding = self.rep_pad_1(img_feature_mat)
            img_feature_mat = self.res_convs[i](img_padding)
            # print('img_feature_mat[%d]: ' % i, torch.isnan(img_feature_mat).any().item())
            # print('img_feature_mat[%d]: ' % i, img_feature_mat.shape)

            pat_padding = self.ref_pad_1(pat_feature_mat)
            pat_feature_mat = self.res_convs[i](pat_padding)
            # print('pat_feature_mat[%d]: ' % i, torch.isnan(pat_feature_mat).any().item())
            # print('pat_feature_mat[%d]: ' % i, pat_feature_mat.shape)

        # Final conv
        img_padding = self.rep_pad_1(img_feature_mat)
        img_feature_mat = self.self_out(img_padding)  # [N, 32, H, W]
        # print('img_feature_mat: ', img_feature_mat[0, :, 32, 32])
        # print('img_feature_mat_out: ', img_feature_mat.shape)
        pat_padding = self.ref_pad_1(pat_feature_mat)
        pat_feature_mat = self.self_out(pat_padding)  # [N, 32, H_p, W_p]
        # print('pat_feature_mat_out: ', pat_feature_mat.shape)

        # Use index_select to generate 3D cost volume
        image_stack = torch.stack([img_feature_mat]*self.D, 4)
        # print('image_stack: ', image_stack.shape)
        pattern_shape = pat_feature_mat.shape
        # print('pat_feature_mat(pattern_shape): ', pattern_shape)
        pattern_search = pat_feature_mat.reshape(pattern_shape[0], pattern_shape[1],
                                                 pattern_shape[2] * pattern_shape[3])
        # print('pattern_search: ', pattern_search.shape)
        pattern_align_vec = torch.index_select(input=pattern_search, dim=2, index=self.idx_pro_vec)
        # print('pattern_align_vec: ', pattern_align_vec.shape)
        pattern_align = pattern_align_vec.reshape(pattern_shape[0], pattern_shape[1], self.H, self.W, self.D)
        # print('pattern_align: ', pattern_align.shape)
        cost_volume = image_stack - pattern_align  # [N, C, H, W, D]

        # self conv on cost_volume
        if not self.vol:
            for i in range(0, 4):
                cost_volume = self.volume_convs[i](cost_volume)
            cost_volume = self.volume_out(cost_volume)

        # Soft-max for select
        volume_cost = torch.norm(input=cost_volume, p=2, dim=1, keepdim=True)  # [N, 1, H, W, D]
        exp_volume_cost = torch.exp(-volume_cost) + self.epsilon  # [N, 1, H, W, D]
        sum_exp_volume_cost = torch.sum(input=exp_volume_cost, dim=4, keepdim=False)  # [N, 1, H, W]

        volume_prob = exp_volume_cost / sum_exp_volume_cost.unsqueeze(4)
        sparse_prob = volume_prob.permute(0, 4, 2, 3, 1).squeeze(4)
        return sparse_prob

        # if not self.vol:
        #     sum_exp_mul = torch.sum(input=exp_volume_cost * self.disp_mat, dim=4, keepdim=False)  # [N, 1, H, W]
        #     sparse_disp = sum_exp_mul / sum_exp_volume_cost  # [N, 1, H, W]
        #     return sparse_disp
        # else:
        #     sparse_volume = exp_volume_cost / sum_exp_volume_cost.unsqueeze(4)  # [N, 1, H, W, D]
        #     sparse_volume = sparse_volume.permute(0, 4, 2, 3, 1).squeeze(4)
        #     return sparse_volume
