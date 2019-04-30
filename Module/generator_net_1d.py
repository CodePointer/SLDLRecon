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

        self.fc_layer1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )
        self.fc_layer3_c1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.fc_layer3_c2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.fc_layer3_c3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):

        fc_1 = self.fc_layer1(x)
        fc_2 = self.fc_layer2(fc_1)
        fc_3_1 = self.fc_layer3_c1(fc_2)
        fc_3_2 = self.fc_layer3_c2(fc_2)
        fc_3_3 = self.fc_layer3_c3(fc_2)
        fc_3 = torch.stack([fc_3_1, fc_3_2, fc_3_3], dim=1).squeeze()
        return fc_3  # [N, 1, 128]
