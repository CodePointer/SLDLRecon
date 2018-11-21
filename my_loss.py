import torch
import numpy as np
import torch.nn.functional as F


def _load_epipolar_info(img_size, file_path, cam_name, pro_name, rot_name, trans_name):
    cam_matrix = np.loadtxt(file_path + cam_name)
    pro_matrix = np.loadtxt(file_path + pro_name)
    rot_matrix = np.loadtxt(file_path + rot_name)
    trans_matrix = np.loadtxt(file_path + trans_name).reshape((3, 1))
    cam_mat = np.hstack((cam_matrix, np.zeros((3, 1))))
    pro_mat = np.dot(pro_matrix, np.hstack((rot_matrix, trans_matrix)))

    m1 = torch.zeros((1, img_size[0], img_size[1]))
    m3 = torch.zeros((1, img_size[0], img_size[1]))
    for h in range(0, img_size[0]):
        for w in range(0, img_size[1]):
            tmp_vec = np.array([(w - cam_mat[0, 2]) / cam_mat[0, 0],
                                (h - cam_mat[1, 2]) / cam_mat[1, 1],
                                1]).reshape((3, 1))
            m1[0, h, w] = np.dot(pro_mat[0, :3], tmp_vec)[0]
            m3[0, h, w] = np.dot(pro_mat[2, :3], tmp_vec)[0]
    d1 = torch.tensor(pro_mat[0, 3])
    d3 = torch.tensor(pro_mat[2, 3])
    return m1.cuda(), m3.cuda(), d1.cuda(), d3.cuda()


def _load_pattern_info(seq_size, file_path, pattern_name, color_name):
    pattern_set = torch.tensor(np.loadtxt(file_path + pattern_name)).type(torch.cuda.LongTensor)
    assert (pattern_set.shape == (seq_size,))
    color_name = torch.tensor(np.loadtxt(file_path + color_name)).type(torch.cuda.FloatTensor)
    assert (color_name.shape == (7, 3))
    return pattern_set, color_name


class MyLoss(torch.nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='sum'):  # elementwise_mean
        super(MyLoss, self).__init__(size_average, reduce, reduction)
        self.seq_size = 130
        self.root_path = './SLDataSet/20181112/'
        self.calib_path = 'virtual/'
        self.pattern_path = ''
        self.threshold = 0.3
        self.reduction = 'elementwise_mean'

        # self.m1, self.m3, self.d1, self.d3 = _load_epipolar_info(
        #     (1024, 1280), self.root_path + self.calib_path, 'cam_mat0.txt', 'pro_mat0.txt', 'rot_mat0.txt',
        #     'trans_vec0.txt'
        # )
        self.pattern, self.color = _load_pattern_info(
            self.seq_size, self.root_path + self.pattern_path, 'rand_seq.txt', 'ColorSet.txt'
        )

    def forward(self, output_data, input_data):
        assert (output_data.shape[-2:] == input_data.shape[-2:])
        # # x_pro_data = (self.m1 * output_data + self.d1) / (self.m3 * output_data + self.d3)
        # idx_data = output_data * (self.seq_size - 1)  # Range: (0, 205)
        # # print(output_data.max().item(), output_data.min().item())
        # idx_left = torch.floor(idx_data).clamp(0, self.seq_size - 2).type(torch.cuda.LongTensor)
        # idx_right = idx_left + 1
        #
        # pattern_left = self.pattern[idx_left]
        # pattern_right = self.pattern[idx_right]
        #
        # color_left = self.color[pattern_left.squeeze(1)].permute((0, 3, 1, 2))
        # color_right = self.color[pattern_right.squeeze(1)].permute((0, 3, 1, 2))
        #
        # weight_data = torch.sigmoid(((idx_data - idx_left.type(torch.cuda.FloatTensor) - 0.5) * 4))
        #
        # refer_data = weight_data * color_right + (1 - weight_data) * color_left
        #
        # valid_tensor = torch.sum((input_data + 1).abs(), 1, keepdim=True) > self.threshold
        # valid_tensor = valid_tensor.type(torch.cuda.FloatTensor)

        # s = 0
        # h = 640
        # w = 522
        # print("Debug Info:")
        # print(idx_left[s, :, h, w])
        # print(idx_right[s, :, h, w])
        # print(pattern_left[s, :, h, w])
        # print(pattern_right[s, :, h, w])
        # print(color_left[s, :, h, w])
        # print(color_right[s, :, h, w])
        # print(weight_data[s, :, h, w])
        # print(refer_data[s, :, h, w])
        # print(input_data[s, :, h, w])
        # print(valid_tensor[s, :, h, w])

        # return (F.mse_loss(input_data * valid_tensor, refer_data *cuda.Dout valid_tensor, reduction=self.reduction)
        #         / valid_tensor.sum())

        return F.mse_loss(input_data.type(torch.cuda.DoubleTensor), output_data.type(torch.cuda.DoubleTensor),
                          reduction=self.reduction)
