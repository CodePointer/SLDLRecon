import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from torch.nn import Sequential


def load_pro_mat(root_path, x_file_name, y_file_name, mat_size, width_pro):
    raw_input = torch.from_numpy(np.fromfile(root_path + x_file_name, dtype=np.uint8)).type(torch.LongTensor)
    x_pro_mat = raw_input.reshape([mat_size[2], mat_size[1], mat_size[0]]).transpose(0, 2)
    assert(list(x_pro_mat.shape) == mat_size)  # [H, W, D]
    raw_input = torch.from_numpy(np.fromfile(root_path + y_file_name, dtype=np.uint8)).type(torch.LongTensor)
    y_pro_mat = raw_input.reshape([mat_size[2], mat_size[1], mat_size[0]]).transpose(0, 2)
    assert(list(y_pro_mat.shape) == mat_size)  # [H, W, D]
    idx_mat = y_pro_mat * width_pro + x_pro_mat
    # print(x_pro_mat[15, 18, :])
    # print(y_pro_mat[15, 18, :])
    # print(idx_mat[15, 18, :])
    idx_pro_vector = idx_mat.reshape(mat_size[0] * mat_size[1] * mat_size[2])  # [H*W*D]
    assert torch.max(idx_pro_vector).item() <= 127 and torch.min(idx_pro_vector).item() >= 0
    return idx_pro_vector.cuda()


def load_disp_volume(root_path, file_name, volume_size, batch_size):
    raw_input = torch.from_numpy(np.fromfile(root_path + file_name, dtype='<f4'))
    assert(raw_input.shape[0] == volume_size[2])
    disp_volume = torch.stack([raw_input] * volume_size[0], 0)
    disp_volume = torch.stack([disp_volume] * volume_size[1], 1)
    disp_volume = torch.stack([disp_volume] * batch_size, 0)
    disp_volume = disp_volume.unsqueeze(1)
    return disp_volume.cuda()


def dn_conv(in_planes, out_planes, kernel_size=5, padding=True):
    padding_size = 0
    if padding:
        padding_size = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=padding_size),
        nn.ReLU(inplace=True),
    )


def up_conv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def conv(in_planes, out_planes, padding=True):
    padding_size = 0
    if padding:
        padding_size = 1
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=padding_size),
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

    def __init__(self, root_path, batch_size, alpha=1024.0, beta=700, K=4):
        super(MyNet, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.N = batch_size
        self.H = int(1024 / pow(2, K))
        self.W = int(1280 / pow(2, K))
        self.D = 64
        self.Hp = int(128 / pow(2, K))
        self.Wp = int(1024 / pow(2, K))
        self.epsilon = 1e-10

        self.idx_pro_vec = load_pro_mat(root_path=root_path,
                                        x_file_name='x_pro_tensor_' + str(self.K) + '.bin',
                                        y_file_name='y_pro_tensor_' + str(self.K) + '.bin',
                                        mat_size=[self.H, self.W, self.D],
                                        width_pro=self.Wp)
        self.disp_mat = load_disp_volume(root_path=root_path, file_name='disp_range.bin',
                                         volume_size=[self.H, self.W, self.D], batch_size=self.N)

        # Pad layers:
        self.rep_pad_2 = nn.ReplicationPad2d(2)
        self.ref_pad_2 = nn.ReflectionPad2d(2)
        self.rep_pad_1 = nn.ReplicationPad2d(1)
        self.ref_pad_1 = nn.ReflectionPad2d(1)

        # Down sample layers
        self.dn_convs = []
        self.dn_self_convs = []
        dn_conv_planes = [3, 32, 32, 32, 32, 32]
        for i in range(0, self.K):
            tmp_conv = nn.Sequential(
                nn.Conv2d(dn_conv_planes[i], dn_conv_planes[i + 1], kernel_size=5, stride=2, padding=0),
                nn.ReLU(inplace=True)
            )
            tmp_self_conv = nn.Sequential(
                nn.Conv2d(dn_conv_planes[i + 1], dn_conv_planes[i + 1], kernel_size=3, padding=0),
                nn.ReLU(inplace=True)
            )
            self.dn_convs.append(tmp_conv)
            self.dn_self_convs.append(tmp_self_conv)
        self.dn_convs = nn.ModuleList(self.dn_convs)
        self.dn_self_convs = nn.ModuleList(self.dn_self_convs)
        # self.image_dn_convs = []
        # dn_conv_planes = [3, 32, 32, 32, 32]
        # for i in range(0, 4):
        #     tmp_conv = nn.Sequential(
        #         nn.Conv2d(dn_conv_planes[i], dn_conv_planes[i + 1], kernel_size=5, stride=2, padding=2),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(dn_conv_planes[i + 1], dn_conv_planes[i + 1], kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.image_dn_convs.append(tmp_conv)
        # self.image_dn_convs = nn.ModuleList(self.image_dn_convs)

        # Pattern ReflectionPad
        # self.pattern_ref = []
        # self.pattern_self_ref = []
        # for i in range(0, 4):
        #     self.pattern_ref.append(nn.ReflectionPad2d(2))
        #     self.pattern_self_ref.append(nn.ReflectionPad2d(1))
        # self.pattern_ref = nn.ModuleList(self.pattern_ref)
        # self.pattern_self_ref = nn.ModuleList(self.pattern_self_ref)
        # self.pattern_dn_convs = []
        # for i in range(0, 4):
        #     tmp_conv = nn.Sequential(
        #         nn.ReflectionPad2d(2),
        #         nn.Conv2d(dn_conv_planes[i], dn_conv_planes[i + 1], kernel_size=5, stride=2, padding=0),
        #         nn.ReLU(inplace=True),
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(dn_conv_planes[i + 1], dn_conv_planes[i + 1], kernel_size=3, padding=0),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.pattern_dn_convs.append(tmp_conv)
        # self.pattern_dn_convs = nn.ModuleList(self.pattern_dn_convs)

        # self_conv res
        self.res_convs = []
        res_conv_planes = [32, 32, 32, 32, 32, 32, 32]
        for i in range(0, 6):
            tmp_conv = nn.Sequential(
                nn.Conv2d(res_conv_planes[i], res_conv_planes[i + 1], kernel_size=3, padding=0),
                nn.BatchNorm2d(res_conv_planes[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.res_convs.append(tmp_conv)
        self.res_convs = nn.ModuleList(self.res_convs)
        self.self_out = nn.Conv2d(res_conv_planes[-1], 32, kernel_size=3, padding=0)

        #  Image self conv layers
        # self.image_self_convs = []
        # self_conv_planes = [32, 32, 32, 32, 32, 32, 32]
        # for i in range(0, 6):
        #     tmp_conv = nn.Sequential(
        #         nn.Conv2d(self_conv_planes[i], self_conv_planes[i + 1], kernel_size=3, padding=1),
        #         nn.BatchNorm2d(self_conv_planes[i + 1]),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.image_self_convs.append(tmp_conv)
        # self.image_self_convs = nn.ModuleList(self.image_self_convs)
        # self.image_self_out = nn.Conv2d(self_conv_planes[6], 32, kernel_size=3, padding=1)

        # Pattern self conv layers
        # self.pattern_self_convs = []
        # for i in range(0, 6):
        #     tmp_conv = nn.Sequential(
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(self_conv_planes[i], self_conv_planes[i + 1], kernel_size=3, padding=0),
        #         nn.BatchNorm2d(self_conv_planes[i + 1]),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.pattern_self_convs.append(tmp_conv)
        # self.pattern_self_convs = nn.ModuleList(self.pattern_self_convs)
        # self.pattern_self_out = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(self_conv_planes[6], 32, kernel_size=3, padding=0)
        # )

        # Volume conv3d layers
        # self.volume_convs = []
        # for i in range(0, 4):
        #     tmp_conv = nn.Sequential(
        #         nn.Conv3d(32, 32, kernel_size=3, padding=1),
        #         nn.BatchNorm3d(32),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.volume_convs.append(tmp_conv)
        # self.volume_convs = nn.ModuleList(self.volume_convs)
        # self.volume_out = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        # Image residual self-conv
        # self.image_res_conv = nn.ModuleList()
        # res_conv_plane = [4, 32, 32, 32, 32]
        # res_conv_dilation = [1, 2, 4, 1]
        # for i in range(0, 4):
        #     tmp_conv = nn.Sequential(
        #         nn.Conv2d(res_conv_plane[i], res_conv_plane[i + 1], kernel_size=3, padding=res_conv_dilation[i],
        #                   dilation=res_conv_dilation[i]),
        #         nn.BatchNorm2d(res_conv_plane[i + 1]),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.image_res_conv.append(tmp_conv)
        # self.disp_res_out = nn.Conv2d(res_conv_plane[-1], 1, kernel_size=3, padding=1)
        # self.final_disp_out = nn.ReLU(inplace=True)  # Make sure output is positive

        # # Up_conv layers
        # self.image_up_convs = []
        # self.image_up_self_convs = []
        # self.predict_disp = []
        # up_self_planes = [32 + 1, 32 + 32 + 1, 32 + 32 + 1, 32 + 32 + 1, 32 + 1]
        # for i in range(0, 4):
        #     tmp_conv = nn.Sequential(
        #         nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.image_up_convs.append(tmp_conv)
        # self.image_up_convs = nn.ModuleList(self.image_up_convs)
        # for i in range(0, 5):
        #     tmp_conv = nn.Sequential(
        #         nn.Conv2d(up_self_planes[i], 32, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.image_up_self_convs.append(tmp_conv)
        # self.image_up_self_convs = nn.ModuleList(self.image_up_self_convs)
        # for i in range(0, 4):
        #     tmp_out = nn.Sequential(
        #         nn.Conv2d(32, 1, kernel_size=3, padding=1),
        #         nn.Sigmoid()
        #     )
        #     self.predict_disp.append(tmp_out)
        # self.predict_disp = nn.ModuleList(self.predict_disp)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):

        # Image, pattern down sample:
        image_dn_conv_out = [x[0]]
        pattern_dn_conv_out = [x[1]]
        for i in range(0, self.K):
            img_padding = self.rep_pad_2(image_dn_conv_out[i])
            img_out_val = self.dn_convs[i](img_padding)
            img_padding = self.rep_pad_1(img_out_val)
            img_out_val = self.dn_self_convs[i](img_padding)
            image_dn_conv_out.append(img_out_val)
            # print('image_dn_conv_out[%d]: ' % i, torch.isnan(img_out_val).any().item())
            # print('image_dn_conv_out[%d]: ' % i, img_out_val.shape)

            pat_padding = self.ref_pad_2(pattern_dn_conv_out[i])
            pat_out_val = self.dn_convs[i](pat_padding)
            pat_padding = self.ref_pad_1(pat_out_val)
            pat_out_val = self.dn_self_convs[i](pat_padding)
            pattern_dn_conv_out.append(pat_out_val)
            # print('pattern_dn_conv_out[%d]: ' % i, torch.isnan(pat_out_val).any().item())
            # print('pattern_dn_conv_out[%d]: ' % i, pat_out_val.shape)

        # Image, pattern residual self conv
        img_feature_mat = image_dn_conv_out[-1]
        pat_feature_mat = pattern_dn_conv_out[-1]
        for i in range(0, 6):
            img_padding = self.rep_pad_1(img_feature_mat)
            img_feature_mat = self.res_convs[i](img_padding)
            # print('img_feature_mat[%d]: ' % i, torch.isnan(img_feature_mat).any().item())
            # print('img_feature_mat[%d]: ' % i, img_feature_mat.shape)

            pat_padding = self.ref_pad_1(pat_feature_mat)
            pat_feature_mat = self.res_convs[i](pat_padding)
            # print('pat_feature_mat[%d]: ' % i, torch.isnan(pat_feature_mat).any().item())
            # print('pat_feature_mat[%d]: ' % i, pat_feature_mat.shape)

        # print('img_feature_mat: ', img_feature_mat[0, :, 32, 32])

        # Final conv
        img_padding = self.rep_pad_1(img_feature_mat)
        img_feature_mat = self.self_out(img_padding)  # [N, 32, H, W]
        # print('img_feature_mat: ', img_feature_mat[0, :, 32, 32])
        # print('img_feature_mat_out: ', img_feature_mat.shape)
        pat_padding = self.ref_pad_1(pat_feature_mat)
        pat_feature_mat = self.self_out(pat_padding)  # [N, 32, H_p, W_p]
        # print('pat_feature_mat_out: ', pat_feature_mat.shape)

        # # Pattern down sample:
        # pattern_dn_conv_out = [x[1]]
        # for i in range(0, 4):
        #     out_val = self.pattern_dn_convs[i](pattern_dn_conv_out[i])
        #     pattern_dn_conv_out.append(out_val)
        #     # print('pattern_dn_conv_out[%d]: ' % i, out_val.shape)
        # # Pattern residual self conv
        # pattern_feature_mat = pattern_dn_conv_out[4]
        # for i in range(0, 6):
        #     pattern_feature_mat = self.pattern_self_convs[i](pattern_feature_mat)
        #     # print('pattern_feature_mat[%d]: ' % i, out_val.shape)
        # pattern_feature_mat = self.pattern_self_out(pattern_feature_mat)

        # Use gather to generate 3D cost volume
        image_stack = torch.stack([img_feature_mat]*self.D, 4)
        # print('image_stack: ', image_stack.shape)
        pattern_shape = pat_feature_mat.shape
        # print('pat_feature_mat(pattern_shape): ', pattern_shape)
        pattern_search = pat_feature_mat.reshape(pattern_shape[0], pattern_shape[1],
                                                 pattern_shape[2] * pattern_shape[3])
        # print('pattern_search: ', pattern_search.shape)
        pattern_align_vec = torch.index_select(input=pattern_search, dim=2, index=self.idx_pro_vec)
        # print('pattern_align_vec: ', pattern_align_vec.shape)
        pattern_align = pattern_align_vec.reshape(pattern_shape[0], pattern_shape[1], self.H, self.W, self.D)
        # print('pattern_align: ', pattern_align.shape)
        cost_volume = image_stack - pattern_align  # [N, C, H, W, D]

        # h = 15
        # w = 18
        # print('pat_feature_mat[2:4, 24]: ', pat_feature_mat[0, :, 2:4, 24])
        # print('pattern_align[16, 18, 10:12]: ', pattern_align[0, :, 16, 18, 10:12])

        # self conv on cost_volume
        # for i in range(0, 4):
        #     cost_volume = self.volume_convs[i](cost_volume)
        # cost_volume = self.volume_out(cost_volume)

        # Select
        volume_cost = torch.norm(input=cost_volume, p=2, dim=1, keepdim=True)  # [N, 1, H, W, D]
        exp_volume_cost = torch.exp(-volume_cost) + self.epsilon  # [N, 1, H, W, D]
        sum_exp_volume_cost = torch.sum(input=exp_volume_cost, dim=4, keepdim=False)  # [N, 1, H, W]
        sum_exp_mul = torch.sum(input=exp_volume_cost * self.disp_mat, dim=4, keepdim=False)  # [N, 1, H, W]
        selected_disp = sum_exp_mul / sum_exp_volume_cost  # [N, 1, H, W]
        volume_prob = exp_volume_cost / sum_exp_volume_cost.unsqueeze(4)  # [N, 1, H, W, D]
        volume_prob = volume_prob.permute(0, 4, 2, 3, 1).squeeze(4)  # [N, D, H, W]

        # print(selected_disp[0, 0, 0:2, 0:2])
        # selected_disp = torch.argmin(volume_cost, dim=4, keepdim=False)
        # print(selected_disp[0, 0, 0:2, 0:2])
        # selected_disp = self.alpha * (selected_disp.type(torch.cuda.FloatTensor) / self.D) + self.beta
        # print(selected_disp.requires_grad)

        # h = 15
        # w = 18
        # print('img_feature_mat: ', img_feature_mat[0, :, h, w])
        # print('pat_feature_mat: ', pat_feature_mat[0, :, h, w])
        # print('img_feature_mat: ', image_dn_conv_out[-1])
        # print('image_stack: ', image_stack[0, :, h, w, 0:6])
        # print('pattern_align: ', pattern_align[0, :, h, w, 0:6])
        # print('cost_volume: ', cost_volume[0, :, h, w, 0:6])
        # print('volume_cost: ', volume_cost[0, 0, h, w, :])
        # print('exp_volume_cost: ', exp_volume_cost[0, 0, h, w, :])
        # print('sum_exp_volume_cost: ', sum_exp_volume_cost[0, 0, h, w])
        # print('disp_mat: ', self.disp_mat[0, 0, h, w, :])
        # print('sum_exp_mul: ', sum_exp_mul[0, 0, h, w])
        # print('selected_disp: ', selected_disp[0, 0, h, w])

        # selected_disp[torch.isnan(selected_disp)] = self.disp_mat[0, 0, 0, 0, int(self.D / 2)].item()
        # print('select_disp: ', selected_disp.shape)
        # sparse_disp = crop_like(Func.interpolate(input=selected_disp, scale_factor=16, mode='bilinear',
        #                                          align_corners=False), x[0])
        # # print('sparse_disp: ', sparse_disp.shape)
        #
        # # Calculate disp residual
        # res_input = torch.cat((x[0], sparse_disp), 1)
        # # print('res_input: ', res_input.shape)
        # for i in range(0, 4):
        #     res_input = self.image_res_conv[i](res_input)
        #     # print('res_input: ', res_input.shape)
        # res_output = self.disp_res_out(res_input)
        # # print('selected_disp: ', selected_disp)
        #
        # # Output disp residual
        # dense_disp = sparse_disp + res_output
        # norm_dense_disp = self.final_disp_out(dense_disp)

        return volume_prob
        # return selected_disp
        # return selected_disp, norm_dense_disp

        # Up sample
        # disp_output = [selected_disp]
        # up_conv_input = []
        # # print(disp_output[0].shape)
        # # print(image_dn_conv_out[4].shape)
        # concat = torch.cat((disp_output[0], image_dn_conv_out[4]), 1)
        # up_conv_input.append(self.image_up_self_convs[0](concat))
        # up_conv_output = []
        # for i in range(0, 3):
        #     # print('up_conv_input[%d]: ' % i, up_conv_input[i].shape)
        #     up_out = self.image_up_convs[i](up_conv_input[i])
        #     up_conv_output.append(up_out)
        #     # print('up_conv_output[%d]: ' % i, up_conv_output[i].shape)
        #     disp_up = crop_like(Func.interpolate(input=disp_output[i], scale_factor=2, mode='bilinear',
        #                                          align_corners=False), up_conv_output[i])
        #     concat = torch.cat((up_conv_output[i], image_dn_conv_out[3 - i], disp_up), 1)
        #     self_out = self.image_up_self_convs[i + 1](concat)
        #     up_conv_input.append(self_out)
        #     disp_out = self.predict_disp[i](up_conv_input[i + 1])
        #     disp_output.append(disp_out)
        # up_out = self.image_up_convs[3](up_conv_input[3])
        # up_conv_output.append(up_out)
        # disp_up = crop_like(Func.interpolate(input=disp_output[3], scale_factor=2, mode='bilinear',
        #                                      align_corners=False), up_conv_output[3])
        # concat = torch.cat((up_conv_output[3], disp_up), 1)
        # self_out = self.image_up_self_convs[4](concat)
        # up_conv_input.append(self_out)
        # disp_out = self.predict_disp[3](up_conv_input[4])
        # disp_output.append(disp_out)
        #
        # # print('disp_output[-1]: ', disp_output[-1].shape)
        # return disp_output[-1]
