import torch
import torch.nn as nn
import numpy as np


class GeneratorNet(nn.Module):
    """
    The GeneratorNet is designed to generate pattern from a random tensor.
    Input: (Random Tensor) 64
        latent tensor: [N, C=1, 64]
    Output: Pattern, [16, 128]
        pattern: [N, C=1, H_sp=16, W_sp=128] range: [0, 1]
        will be interpolated 8 times
    """

    def __init__(self, root_path, batch_size, opts=None):
        super(GeneratorNet, self).__init__()

        if opts is None:
            opts = {}
        self.in_size = 64
        self.N = batch_size
        self.scale = 1
        self.H = int(128/self.scale)
        self.Hc = int(128/8)
        self.W = int(1024/self.scale)
        self.Wc = int(1024/8)
        self.C = 1
        self.Lw = 64
        self.Lh = 4
        self.sigma = 1.0
        self.range_squeeze = 1 / 2.0
        self.lumi_k = np.sqrt(2*np.pi) * self.sigma**2 * 0.8

        self.epsilon = 1e-10

        self.thred = 1.0

        # Input layers
        self.in_layer = nn.Sequential(
            nn.Linear(64, 1024),
        )

        # Non-Linear Layers
        self.non_linears = []
        for i in range(0, 2):
            tmp_layer = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True)
            )
            self.non_linears.append(tmp_layer)
        self.non_linears = nn.ModuleList(self.non_linears)

        self.h_out_layer = nn.Sequential(
            nn.Linear(1024, self.H)
        )
        self.w_out_layer = nn.Sequential(
            nn.Linear(1024, self.W)
        )

        self.final_activate = nn.Tanh()

        # # h, w mat
        # tmp_h_vec = torch.Tensor(range(0, self.H, 1))
        # self.h_mat = torch.stack([tmp_h_vec] * batch_size, dim=0)  # [N, H]
        # self.h_mat = self.h_mat.unsqueeze(1)  # [N, 1, H]
        #
        # tmp_w_vec = torch.Tensor(range(0, self.W, 1))
        # self.w_mat = torch.stack([tmp_w_vec] * batch_size, dim=0)  # [N, W]
        # self.w_mat = self.w_mat.unsqueeze(1)  # [N, 1, W]
        #
        # self.h_mat = self.h_mat.cuda()
        # self.w_mat = self.w_mat.cuda()

    def forward(self, x):

        # Input layer
        fc_res = self.in_layer(x)

        # Non-Linear Layers
        for i in range(0, 2):
            fc_res = self.non_linears[i](fc_res)

        h_raw_out = self.h_out_layer(fc_res)
        h_up_out = torch.nn.functional.interpolate(input=h_raw_out, scale_factor=self.scale)
        w_raw_out = self.w_out_layer(fc_res)
        w_up_out = torch.nn.functional.interpolate(input=w_raw_out, scale_factor=self.scale)

        h_vec_out = h_up_out.unsqueeze(3)  # [N, 1, H, 1]
        w_vec_out = w_up_out.unsqueeze(2)  # [N, 1, 1, W]

        # h_dis = torch.pow(self.h_mat - h_vec_out, 2) * self.range_squeeze
        # w_dis = torch.pow(self.w_mat - w_vec_out, 2) * self.range_squeeze
        #
        # h_exp_val = torch.exp(-h_dis / (2 * self.sigma**2))  # [N, Lh, H]
        # w_exp_val = torch.exp(-w_dis / (2 * self.sigma**2))  # [N, Lw, W]
        #
        # h_lumi_val = h_exp_val * self.lumi_k / (np.sqrt(2 * np.pi) * self.sigma**2)  # [N, Lh, H]
        # w_lumi_val = w_exp_val * self.lumi_k / (np.sqrt(2 * np.pi) * self.sigma**2)  # [N, Lw, W]
        #
        # h_pattern_vec = torch.sum(input=h_lumi_val, dim=1, keepdim=True).unsqueeze(3)  # [N, 1, H, 1]
        # w_pattern_vec = torch.sum(input=w_lumi_val, dim=1, keepdim=True).unsqueeze(2)  # [N, 1, 1, W]
        #
        # pattern_mat = h_pattern_vec + w_pattern_vec
        pattern_mat = h_vec_out + w_vec_out

        # print(torch.min(pattern_mat), torch.max(pattern_mat))
        final_pattern = self.final_activate(pattern_mat)

        return final_pattern * 2 - 1

