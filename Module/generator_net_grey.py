import torch
import torch.nn as nn
import math


class GeneratorNet(nn.Module):
    """
    The GeneratorNet is designed to generate pattern from a random tensor.
    Input: 2 Image with grid pattern
        Image: [N, C=1, H, W] * 1
    Output: Pattern, [128, 1024]
        pattern: [N, C=1, H_sp=16, W_sp=128] range: [-1, 1]
        2D grid version
    """

    def __init__(self, para_sec, batch_size, opts=None):
        super(GeneratorNet, self).__init__()

        if opts is None:
            opts = {}
        self.N = batch_size
        self.Hp = para_sec.getint('pro_height')
        self.Wp = para_sec.getint('pro_width')
        self.C = 1
        self.scale = 1
        self.in_size = 64
        self.conv_k = 6
        self.Hc = int(para_sec.getfloat('cam_height') / math.pow(2, self.conv_k))
        self.Wc = int(para_sec.getfloat('cam_width') / math.pow(2, self.conv_k))

        self.epsilon = 1e-10
        self.thred = 1.0

        # Down Sample layers
        self.dn_convs = []
        self.dn_self_convs = []
        dn_conv_planes = [self.C, 4, 8, 16, 32, 64, 64]
        for i in range(0, self.conv_k):
            tmp_conv = nn.Sequential(
                nn.Conv2d(dn_conv_planes[i], dn_conv_planes[i + 1],
                          kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
            tmp_self_conv = nn.Sequential(
                nn.Conv2d(dn_conv_planes[i + 1], dn_conv_planes[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(dn_conv_planes[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.dn_convs.append(tmp_conv)
            self.dn_self_convs.append(tmp_self_conv)
        self.dn_convs = nn.ModuleList(self.dn_convs)
        self.dn_self_convs = nn.ModuleList(self.dn_self_convs)

        # Vector extraction layer
        self.linear_out = nn.Linear(self.Hc * self.Wc * dn_conv_planes[-1], 64)

        # h_layers
        self.h_linears = []
        h_linear_planes = [64, self.Hp, self.Hp, self.Hp]
        for i in range(0, 3):
            tmp_layer = nn.Sequential(
                nn.Linear(h_linear_planes[i], h_linear_planes[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.h_linears.append(tmp_layer)
        self.h_linears = nn.ModuleList(self.h_linears)
        self.h_out_layer = nn.Sequential(
            nn.Linear(h_linear_planes[-1], self.Hp),
            nn.Tanh()
        )

        # h_layers
        self.w_linears = []
        w_linear_planes = [64, self.Wp, self.Wp, self.Wp]
        for i in range(0, 3):
            tmp_layer = nn.Sequential(
                nn.Linear(w_linear_planes[i], w_linear_planes[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.w_linears.append(tmp_layer)
        self.w_linears = nn.ModuleList(self.w_linears)
        self.w_out_layer = nn.Sequential(
            nn.Linear(w_linear_planes[-1], self.Wp),
            nn.Tanh()
        )

        self.final_layer = nn.Hardtanh()

    def forward(self, x):

        for i in range(0, self.conv_k):
            x = self.dn_convs[i](x)
            x = self.dn_self_convs[i](x)
        feature_vec = self.linear_out(x.reshape(self.N, 1, self.Hc * self.Wc * 64))
        # feature_vec = x

        # h_vector
        h_vec = feature_vec
        for i in range(0, 3):
            h_vec = self.h_linears[i](h_vec)
        h_vec = self.h_out_layer(h_vec)
        h_up_out = torch.nn.functional.interpolate(input=h_vec, scale_factor=self.scale)  # [N, 1, Hp]
        h_vec_out = h_up_out.unsqueeze(3)  # [N, 1, Hp, 1]

        # w_vector
        w_vec = feature_vec
        for i in range(0, 3):
            w_vec = self.w_linears[i](w_vec)
        w_vec = self.w_out_layer(w_vec)
        w_up_out = torch.nn.functional.interpolate(input=w_vec, scale_factor=self.scale)
        w_vec_out = w_up_out.unsqueeze(2)  # [N, 1, 1, Wp]

        pattern_mat = self.final_layer(h_vec_out + w_vec_out)

        return pattern_mat * 2 - 1

