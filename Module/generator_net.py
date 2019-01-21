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
        self.W = int(1024)
        self.C = 32
        self.epsilon = 1e-10

        # Input layers
        self.in_layer = nn.Sequential(
            nn.Linear(64, self.C * 2 * 16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Up sample layers
        self.up_convs = []
        self.up_self_convs = []
        up_conv_planes = [self.C, self.C, self.C, self.C]
        for i in range(0, 3):
            tmp_conv = nn.Sequential(
                nn.ConvTranspose2d(up_conv_planes[i], up_conv_planes[i + 1], kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            tmp_self_conv = nn.Sequential(
                nn.Conv2d(up_conv_planes[i + 1], up_conv_planes[i + 1], kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.up_convs.append(tmp_conv)
            self.up_self_convs.append(tmp_self_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.up_self_convs = nn.ModuleList(self.up_self_convs)

        # Output layers
        self.out_layer = nn.Sequential(
            nn.Conv2d(up_conv_planes[-1], 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):

        # Input layer
        pat_feature_vec = self.in_layer(x)
        pat_feature_mat = pat_feature_vec.reshape(self.N, self.C, 2, 16)
        # print('After input:', pat_feature_mat.shape)

        # Up sample
        for i in range(0, 3):
            pat_feature_mat = self.up_convs[i](pat_feature_mat)
            # print('Up sample:', pat_feature_mat.shape)
            pat_feature_mat = self.up_self_convs[i](pat_feature_mat)
            # print('Self conv:', pat_feature_mat.shape)

        # Output layers
        sparse_pattern = self.out_layer(pat_feature_mat)
        return sparse_pattern

        # print('sparse_pattern:', sparse_pattern.shape)
        # dense_pattern = nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
        #                                           align_corners=False)

        # print('dense_pattern:', dense_pattern.shape)
        # return dense_pattern
