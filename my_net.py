import torch
import torch.nn as nn
import torch.nn.functional as Func


def dn_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True)
    )


def up_conv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def crop_like(in_planes, ref):
    assert (in_planes.size(2) >= ref.size(2) and in_planes.size(3) >= ref.size(3))
    return in_planes[:, :, :ref.size(2), :ref.size(3)]


class MyNet(nn.Module):
    """
    The MyNet is designed to predict pixel-label from input image.

    Input: RGB Image.
        Shape: Tensor[sample, channel=3, height=1024, width=1280]
        Range: [-1, 1]
        The image captured by camera.
    Output: The x_pro(correspondence) of projector label.
        Shape: Tensor[sample, channel=1, height=1024, width=1280]
        Range: (0, 1)
        The labels of x_pro. (0, 1)
    """

    def __init__(self, alpha=10.0, beta=0.01):
        super(MyNet, self).__init__()

        self.alpha = alpha
        self.beta = beta

        dn_conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = dn_conv(3, dn_conv_planes[0], kernel_size=7)
        self.conv2 = dn_conv(dn_conv_planes[0], dn_conv_planes[1], kernel_size=5)
        self.conv3 = dn_conv(dn_conv_planes[1], dn_conv_planes[2])
        self.conv4 = dn_conv(dn_conv_planes[2], dn_conv_planes[3])
        self.conv5 = dn_conv(dn_conv_planes[3], dn_conv_planes[4])
        self.conv6 = dn_conv(dn_conv_planes[4], dn_conv_planes[5])
        self.conv7 = dn_conv(dn_conv_planes[5], dn_conv_planes[6])

        up_conv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = up_conv(dn_conv_planes[6], up_conv_planes[0])
        self.upconv6 = up_conv(up_conv_planes[0], up_conv_planes[1])
        self.upconv5 = up_conv(up_conv_planes[1], up_conv_planes[2])
        self.upconv4 = up_conv(up_conv_planes[2], up_conv_planes[3])
        self.upconv3 = up_conv(up_conv_planes[3], up_conv_planes[4])
        self.upconv2 = up_conv(up_conv_planes[4], up_conv_planes[5])
        self.upconv1 = up_conv(up_conv_planes[5], up_conv_planes[6])

        self.iconv7 = conv(up_conv_planes[0] + dn_conv_planes[5], up_conv_planes[0])
        self.iconv6 = conv(up_conv_planes[1] + dn_conv_planes[4], up_conv_planes[1])
        self.iconv5 = conv(up_conv_planes[2] + dn_conv_planes[3], up_conv_planes[2])
        self.iconv4 = conv(up_conv_planes[3] + dn_conv_planes[2], up_conv_planes[3])
        self.iconv3 = conv(up_conv_planes[4] + dn_conv_planes[1] + 1, up_conv_planes[4])
        self.iconv2 = conv(up_conv_planes[5] + dn_conv_planes[0] + 1, up_conv_planes[5])
        self.iconv1 = conv(up_conv_planes[6] + 1, up_conv_planes[6])

        self.predict_disp4 = predict_disp(up_conv_planes[3])
        self.predict_disp3 = predict_disp(up_conv_planes[4])
        self.predict_disp2 = predict_disp(up_conv_planes[5])
        self.predict_disp1 = predict_disp(up_conv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Down Sample:
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        # Up Sample and combine:
        out_up_conv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_up_conv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_up_conv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_up_conv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_up_conv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_up_conv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_up_conv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_up_conv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_up_conv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(Func.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_up_conv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_up_conv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(Func.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_up_conv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_up_conv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(Func.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_up_conv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        return disp1
