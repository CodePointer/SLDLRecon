import torch
import torch.nn as nn
import math


class UpNet(nn.Module):
    """
    The UpNet is designed to Generate dense_disp info.
    Input: (RGB Image, disp_c Image)
        RGB Image: [N, C=3, H=1024, W=1280], range: [-1, 1]
        Disp_c:    [N, C=1, H_c=H/2^K, W_c=2^K], range: [716, 1724]
    Output: dense_disp
        dense_disp: [N, C=1, H=1024, W=1280], range: [716, 1724]
    """

    def __init__(self, down_k=5):
        super(UpNet, self).__init__()

        self.K = down_k
        self.sep_layer = 3

        # Upsample layer
        self.up_sample = nn.Upsample(scale_factor=math.pow(2, self.K), mode='bilinear')

        # Residual estimation
        res_conv_dilation = [1, 2, 4, 8, 1, 1]

        self.image_res_conv = []
        image_res_conv_plane = [3, 16, 16, 16]
        for i in range(0, self.sep_layer):
            tmp_conv = nn.Sequential(
                nn.Conv2d(image_res_conv_plane[i], image_res_conv_plane[i + 1], kernel_size=3,
                          padding=res_conv_dilation[i], dilation=res_conv_dilation[i]),
                nn.BatchNorm2d(image_res_conv_plane[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.image_res_conv.append(tmp_conv)
        self.image_res_conv = nn.ModuleList(self.image_res_conv)

        self.disp_res_conv = []
        disp_res_conv_plane = [1, 16, 16, 16]
        for i in range(0, self.sep_layer):
            tmp_conv = nn.Sequential(
                nn.Conv2d(disp_res_conv_plane[i], disp_res_conv_plane[i + 1], kernel_size=3,
                          padding=res_conv_dilation[i], dilation=res_conv_dilation[i]),
                nn.BatchNorm2d(disp_res_conv_plane[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.disp_res_conv.append(tmp_conv)
        self.disp_res_conv = nn.ModuleList(self.disp_res_conv)

        self.res_conv = []
        res_conv_plane = [32, 32, 32, 32]
        for i in range(0, 6 - self.sep_layer):
            tmp_conv = nn.Sequential(
                nn.Conv2d(res_conv_plane[i], res_conv_plane[i + 1], kernel_size=3,
                          padding=res_conv_dilation[i + self.sep_layer],
                          dilation=res_conv_dilation[i + self.sep_layer]),
                nn.BatchNorm2d(res_conv_plane[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.res_conv.append(tmp_conv)
        self.res_conv = nn.ModuleList(self.res_conv)

        self.res_out = nn.Conv2d(res_conv_plane[-1], 1, kernel_size=3, padding=1)
        self.last_relu = nn.ReLU(inplace=True)   # Make sure output is positive

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        The input disp_c is in size of H_c, W_c. image is the origin res.
        :param x: (image, disp_c)
        :return: full_res disp
        """
        # Upsample disp_c:
        sparse_disp = self.up_sample(x[1])
        # print('sparse_disp: ', sparse_disp.shape)

        # Image conv:  [N, 16, 1024, 1280]
        image_self_conv_in = x[0]
        for i in range(0, self.sep_layer):
            image_self_conv_out = self.image_res_conv[i](image_self_conv_in)
            image_self_conv_in = image_self_conv_out

        # disp conv:  [N, 16, 1024, 1280]
        disp_self_conv_in = sparse_disp
        for i in range(0, self.sep_layer):
            disp_self_conv_out = self.disp_res_conv[i](disp_self_conv_in)
            disp_self_conv_in = disp_self_conv_out

        # Combine and conv:
        concat_conv_in = torch.cat((image_self_conv_out, disp_self_conv_out), 1)
        for i in range(0, 6 - self.sep_layer):
            concat_conv_out = self.res_conv[i](concat_conv_in)
            concat_conv_in = concat_conv_out
        res_disp = self.res_out(concat_conv_in)

        # Final output
        dense_disp = sparse_disp + res_disp
        dense_disp = self.last_relu(dense_disp)

        return dense_disp, sparse_disp
