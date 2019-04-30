import torch
import torch.nn as nn
import math


class DenseNet(nn.Module):
    """
    The DenseNet is designed to Generate dense_disp info.
    Input: (RGB Image, disp_c Image)
        RGB Image: [N, C=3, H=1024, W=1280], range: [-1, 1]
        Est Image: [N, C=3, H=1024, W=1280], range: [-1, 1]
        Disp_c:    [N, C=1, H=1024, W=1280], range: [0, 1]
    Output: dense_disp
        dense_disp: [N, C=1, H=1024, W=1280], range: [?, ?]
    """

    def __init__(self):
        super(DenseNet, self).__init__()

        # Residual estimation
        self.res_in = nn.Conv2d(4, 32, kernel_size=3, padding=1)

        res_conv_dilation = [1, 2, 4, 8, 1, 1]
        self.res_conv = []
        for i in range(0, 6):
            tmp_conv = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3,
                          padding=res_conv_dilation[i],
                          dilation=res_conv_dilation[i]),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.res_conv.append(tmp_conv)
        self.res_conv = nn.ModuleList(self.res_conv)
        self.res_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.last_relu = nn.ReLU(inplace=True)   # Make sure output is positive

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: (image, img_est, disp_out)
        :return: full_res disp
        """

        # Image conv:  [N, 16, 1024, 1280]
        concat_conv_in = torch.cat(x, 1)
        concat_conv_out = self.res_in(concat_conv_in)
        concat_conv_in = concat_conv_out
        # concat_conv_in = x
        for i in range(0, 6):
            concat_conv_out = self.res_conv[i](concat_conv_in)
            concat_conv_in = concat_conv_out
        res_disp = self.res_out(concat_conv_in)

        # Final output
        dense_disp = res_disp + x[-1]
        dense_disp = self.last_relu(dense_disp)

        return dense_disp
