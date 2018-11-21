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


def conv(in_planes, out_planes, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=0),
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


class PatchNet(nn.Module):
    """
    The PatchNet is designed to predict patch-label from input image.

    Input: RGB Patch at size 21x21.
        Shape: Tensor[sample, channel=3, height=21, width=21]
        Range: [-1, 1]
    Output: The x_pro(correspondence) of projector label.
        Shape: Tensor[sample, channel=1, height=1, width=1]
        Range: (0, 1)
        The labels of x_pro. (0, 1)
    """

    def __init__(self, alpha=10.0, beta=0.01):
        super(PatchNet, self).__init__()

        self.conv1 = conv(3, 32, 5)
        self.conv2 = conv(32, 64, 13)
        self.conv3 = conv(64, 1, 5)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        out_conv1 = self.conv1(x)  # 17x17x32
        out_conv2 = self.conv2(out_conv1)  # 5x5x64
        out_conv3 = self.conv3(out_conv2)  # 1x1x1

        return out_conv3
