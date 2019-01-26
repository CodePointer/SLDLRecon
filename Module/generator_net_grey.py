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

        # Input layers
        self.in_layer = nn.Sequential(
            nn.Linear(64, self.C * self.Hc * self.Wc),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Self linear layers
        self.fc_layers = []
        in_channel = self.C * self.Hc * self.Wc
        out_channel = self.C * self.Hc * self.Wc
        for i in range(0, 2):
            tmp_fc = nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.fc_layers.append(tmp_fc)
        self.fc_layers = nn.ModuleList(self.fc_layers)

        # Output layers
        self.out_layer = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
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
        fc_res = self.in_layer(x)

        # self linear layers
        for i in range(0, 2):
            fc_res = self.fc_layers[i](fc_res)

        # Reshape
        sparse_mat = fc_res.reshape(self.N, self.C, self.Hc, self.Wc)

        # Output layers
        sparse_pattern = self.out_layer(sparse_mat)
        return sparse_pattern

        # print('sparse_pattern:', sparse_pattern.shape)
        # dense_pattern = nn.functional.interpolate(input=sparse_pattern, scale_factor=8, mode='bilinear',
        #                                           align_corners=False)

        # print('dense_pattern:', dense_pattern.shape)
        # return dense_pattern
