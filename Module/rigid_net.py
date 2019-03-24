import torch
import torch.nn as nn
import torch.nn.functional as func


def dn_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        # nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )


class RigidNet(nn.Module):
    """
    The RigidNet is designed to check disp mat is rigid or not.
    Input: range: (0, 1)
        Disp mat: [N, C=2, Hc=1024, Wc=1280]
    Output: (0, 1)
        pattern: [N, C, Hc=1024, Wc=1280] range: [0, 1]
    """

    def __init__(self, time_range=2):
        super(RigidNet, self).__init__()

        self.T = time_range
        self.max_val = torch.tensor([1 - 1e-5]).cuda()
        self.min_val = torch.tensor([1e-5]).cuda()

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = dn_conv(1*self.T, conv_planes[0], kernel_size=7)
        self.conv2 = dn_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = dn_conv(conv_planes[1], conv_planes[2])
        self.conv4 = dn_conv(conv_planes[2], conv_planes[3])
        self.conv5 = dn_conv(conv_planes[3], conv_planes[4])
        self.conv6 = dn_conv(conv_planes[4], conv_planes[5])
        self.conv7 = dn_conv(conv_planes[5], conv_planes[6])

        self.adv_layer = nn.Sequential(
            nn.Linear(conv_planes[6]*80, 1)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):

        # Down sample
        out_conv1 = self.conv1(x)
        # print('out_conv1: ', out_conv1.shape)
        out_conv2 = self.conv2(out_conv1)
        # print('out_conv2: ', out_conv2.shape)
        out_conv3 = self.conv3(out_conv2)
        # print('out_conv3: ', out_conv3.shape)
        out_conv4 = self.conv4(out_conv3)
        # print('out_conv4: ', out_conv4.shape)
        out_conv5 = self.conv5(out_conv4)
        # print('out_conv5: ', out_conv5.shape)
        out_conv6 = self.conv6(out_conv5)
        # print('out_conv6: ', out_conv6.shape)
        out_conv7 = self.conv7(out_conv6)
        # print('out_conv7: ', out_conv7.shape)
        linear_conv = out_conv7.reshape(out_conv7.shape[0], -1)
        # print('linear_conv: ', linear_conv.shape)

        validity = self.adv_layer(linear_conv)

        return validity
