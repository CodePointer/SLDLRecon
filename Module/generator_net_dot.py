import torch
import torch.nn as nn


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
        self.H = int(128)
        self.Hc = int(128/8)
        self.W = int(1024)
        self.Wc = int(1024/8)
        self.C = 1
        self.epsilon = 1e-10

        self.dot_num = 512
        self.thred = 1.0

        # Input layers
        self.in_layer = nn.Sequential(
            nn.Linear(1, self.dot_num * 2, bias=False),
            nn.Tanh()
        )

        self.thred_filter = nn.ReLU(inplace=True)

        self.lumi_layer = nn.Tanh()

        # h, w mat
        tmp_h_vec = torch.Tensor(range(0, self.Hc, 1))
        self.h_mat = torch.stack([tmp_h_vec] * batch_size, dim=0)
        self.h_mat = torch.stack([self.h_mat] * self.dot_num, dim=1)
        self.h_mat = torch.stack([self.h_mat] * self.Wc, dim=3)  # [N, dot_num, Hc, Wc]
        tmp_w_vec = torch.Tensor(range(0, self.Wc, 1))
        self.w_mat = torch.stack([tmp_w_vec] * batch_size, dim=0)
        self.w_mat = torch.stack([self.w_mat] * self.dot_num, dim=1)
        self.w_mat = torch.stack([self.w_mat] * self.Hc, dim=2)  # [N, dot_num, Hc, Wc]

        self.h_mat = self.h_mat.cuda()
        self.w_mat = self.w_mat.cuda()

    def forward(self, x):

        # Input layer
        fc_res = self.in_layer(x)
        point_set = fc_res.reshape(self.N, self.dot_num, 2)
        point_set = (point_set / 2 + 0.5)
        # print('sparse_mat: ', sparse_mat.shape)

        # Calculate distance
        x_set = point_set[:, :, 0] * self.Wc
        x_set = x_set.unsqueeze(2).unsqueeze(3)  # [N, dot_num, 1, 1]
        y_set = point_set[:, :, 1] * self.Hc
        y_set = y_set.unsqueeze(2).unsqueeze(3)
        dis_mat = torch.sqrt((self.w_mat - x_set) ** 2 + (self.h_mat - y_set) ** 2)

        weight_mat = (self.thred - dis_mat) * 3
        weight_mat = self.thred_filter(weight_mat)
        lumi_val = torch.sum(weight_mat, dim=1, keepdim=True)  # [N, 1, Hc, Wc]
        lumi_val = self.lumi_layer(lumi_val)

        return lumi_val * 2 - 1

