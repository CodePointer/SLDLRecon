import torch
import torch.nn as nn
import torch.nn.functional as func


def dn_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        # nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Tanh()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        # nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def up_conv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(in_mat, ref):
    assert (in_mat.size(2) >= ref.size(2) and in_mat.size(3) >= ref.size(3))
    return in_mat[:, :, :ref.size(2), :ref.size(3)]


class DepthNet(nn.Module):
    """
    The DepthNet is designed to estimate depth from flow.
    Input: range: (0, max)
        Flow mat: [N, C, Hc=1024, Wc=1280]
    Output: (0, 1)
        pattern: [N, C, Hc=1024, Wc=1280] range: [0, 1]
    """

    def __init__(self, alpha_range=None):
        super(DepthNet, self).__init__()

        if alpha_range is None:
            alpha_range = (0.0, 3.14)
        self.b = alpha_range[0]
        self.a = alpha_range[1] - alpha_range[0]

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = dn_conv(2, conv_planes[0], kernel_size=7)
        self.conv2 = dn_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = dn_conv(conv_planes[1], conv_planes[2])
        self.conv4 = dn_conv(conv_planes[2], conv_planes[3])
        self.conv5 = dn_conv(conv_planes[3], conv_planes[4])
        self.conv6 = dn_conv(conv_planes[4], conv_planes[5])
        self.conv7 = dn_conv(conv_planes[5], conv_planes[6])

        # up_conv_planes = [512, 512, 256, 128, 64, 32, 16]
        up_conv_planes = [16, 32, 64, 128, 256, 512, 512]
        self.upconv7 = up_conv(conv_planes[6], up_conv_planes[6])
        self.upconv6 = up_conv(up_conv_planes[6], up_conv_planes[5])
        self.upconv5 = up_conv(up_conv_planes[5], up_conv_planes[4])
        self.upconv4 = up_conv(up_conv_planes[4], up_conv_planes[3])
        self.upconv3 = up_conv(up_conv_planes[3], up_conv_planes[2])
        self.upconv2 = up_conv(up_conv_planes[2], up_conv_planes[1])
        self.upconv1 = up_conv(up_conv_planes[1], up_conv_planes[0])

        self.iconv7 = conv(up_conv_planes[6] + conv_planes[5], up_conv_planes[6])
        self.iconv6 = conv(up_conv_planes[5] + conv_planes[4], up_conv_planes[5])
        self.iconv5 = conv(up_conv_planes[4] + conv_planes[3], up_conv_planes[4])
        self.iconv4 = conv(up_conv_planes[3] + conv_planes[2], up_conv_planes[3])
        self.iconv3 = conv(1 + up_conv_planes[2] + conv_planes[1], up_conv_planes[2])
        self.iconv2 = conv(1 + up_conv_planes[1] + conv_planes[0], up_conv_planes[1])
        self.iconv1 = conv(1 + up_conv_planes[0], up_conv_planes[0])

        self.predict_disp4 = predict_disp(up_conv_planes[3])
        self.predict_disp3 = predict_disp(up_conv_planes[2])
        self.predict_disp2 = predict_disp(up_conv_planes[1])
        self.predict_disp1 = predict_disp(up_conv_planes[0])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):

        # Down sample
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        # Up sample
        out_upconv7 = self.upconv7(out_conv7)
        concat7 = torch.cat((out_upconv7, out_conv6), dim=1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = self.upconv6(out_iconv7)
        concat6 = torch.cat((out_upconv6, out_conv5), dim=1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = self.upconv5(out_iconv6)
        concat5 = torch.cat((out_upconv5, out_conv4), dim=1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = self.upconv4(out_iconv5)
        concat4 = torch.cat((out_upconv4, out_conv3), dim=1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.predict_disp4(out_iconv4)

        out_upconv3 = self.upconv3(out_iconv4)
        disp4_up = func.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False)
        concat3 = torch.cat((disp4_up, out_upconv3, out_conv2), dim=1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.predict_disp3(out_iconv3)

        out_upconv2 = self.upconv2(out_iconv3)
        disp3_up = func.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False)
        concat2 = torch.cat((disp3_up, out_upconv2, out_conv1), dim=1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.predict_disp2(out_iconv2)

        out_upconv1 = self.upconv1(out_iconv2)
        disp2_up = func.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False)
        concat1 = torch.cat((disp2_up, out_upconv1), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.predict_disp1(out_iconv1)

        return self.a * (disp1 + 1) / 2 + self.b
