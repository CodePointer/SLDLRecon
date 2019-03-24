import torch
import numpy as np


class CalibPara:
    def __init__(self):
        self.L = None       # Oc to Op distance.
        self.Fx = None      # Oc to epi-line distance. Matrix.
        self.Tx = None      # Oc to Op on epi-line direction.
        self.CosF = None    # cos(alpha) of fx and Fx.
        self.Xop = None     # Op to x distance.

    def to_cuda(self):
        self.L = self.L.cuda() if self.L else None
        self.Fx = self.Fx.cuda() if self.Fx else None
        self.Tx = self.Tx.cuda() if self.Tx else None
        self.CosF = self.CosF.cuda() if self.CosF else None
        self.Xop = self.Xop.cuda() if self.Xop else None

    def __str__(self):
        str_list = ['CalibPara: ',
                    'L    : ' + ('None' if self.L is None else str(self.L.shape)),
                    'Fx   : ' + ('None' if self.Fx is None else str(self.Fx.shape)),
                    'Tx   : ' + ('None' if self.Tx is None else str(self.Tx.shape)),
                    'CosF : ' + ('None' if self.CosF is None else str(self.CosF.shape)),
                    'Xop  : ' + ('None' if self.Xop is None else str(self.Xop.shape))]
        return '\n'.join(str_list)


class FlowEstimator:
    def __init__(self, root_dir, disp_range):
        self.M = torch.from_numpy(np.load(root_dir + 'para_M.npy')).float()
        self.D = torch.from_numpy(np.load(root_dir + 'para_D.npy')).float()
        self.H, self.W = self.M.shape[-2:]
        self.f_tvec_mul = 1185.92488
        self.y_bias = 512.0
        self.disp_range = disp_range
        x_cam_ori_mat = torch.arange(0, self.W).reshape(1, -1).repeat(self.H, 1)
        y_cam_ori_mat = torch.arange(0, self.H).reshape(-1, 1).repeat(1, self.W)
        self.cam_ori_mat = torch.stack((x_cam_ori_mat, y_cam_ori_mat), dim=0).reshape((1, 2, self.H, self.W)).float()
        self.M = self.M.unsqueeze(0)
        self.D = self.D.unsqueeze(0)
        self.f = (1179.747380 + 1182.235175) / 2
        self.dx = 640.0
        self.dy = 512.0
        self.op_center = None
        self.para = CalibPara()
        self.is_cuda = False

    def to_cuda(self):
        self.M = self.M.cuda()
        self.D = self.D.cuda()
        self.cam_ori_mat = self.cam_ori_mat.cuda()
        self.para.to_cuda()
        self.is_cuda = True

    def get_flow_value(self, disp_cam, mask_cam, cor_xc_t, cor_yc_t, mask_pro, cor_xc=None, cor_yc=None):

        # disp -> xy_mat
        tmp_mat = disp_cam * self.D[:, 2] + self.M[:, 2, :, :] * self.f_tvec_mul
        xp_mat = (disp_cam * self.D[:, 0] + self.M[:, 0, :, :] * self.f_tvec_mul) / tmp_mat
        yp_mat = (disp_cam * self.D[:, 1] + self.M[:, 1, :, :] * self.f_tvec_mul) / tmp_mat + self.y_bias
        # print('xp_mat:', torch.max(xp_mat), torch.min(xp_mat))
        # print('yp_mat:', torch.max(yp_mat), torch.min(yp_mat))
        # print('xp_mat[%d, %d]: ' % (h, w), xp_mat[:, :, h, w])
        # print('yp_mat[%d, %d]: ' % (h, w), yp_mat[:, :, h, w])

        # xy_mat to [-1, 1]
        Hp, Wp = cor_xc_t.shape[-2:]
        xp_mat = xp_mat / (Wp - 1) * 2.0 - 1.0
        yp_mat = yp_mat / (Hp - 1) * 2.0 - 1.0
        xp_mat[mask_cam == 0] = -1
        yp_mat[mask_cam == 0] = -1
        grid_pro_mat = torch.cat((xp_mat, yp_mat), dim=1).permute(0, 2, 3, 1)

        # sample from cor_xy
        # cor_xy = torch.cat((cor_xc, cor_yc), dim=1)
        cor_xy_t = torch.cat((cor_xc_t, cor_yc_t), dim=1)
        cam_dst_mat = torch.nn.functional.grid_sample(input=cor_xy_t, grid=grid_pro_mat)
        # cam_ori_mat = torch.nn.functional.grid_sample(input=cor_xy, grid=grid_pro_mat)
        mask_flow = torch.nn.functional.grid_sample(input=mask_pro.float(), grid=grid_pro_mat)
        # print('mask_flow:', torch.max(mask_flow), torch.min(mask_flow))
        mask_flow[mask_flow < 0.999] = 0
        mask_flow[mask_flow > 0] = 1

        # flow mat
        flow_mat = cam_dst_mat - self.cam_ori_mat
        mask_flow[flow_mat[:, 0:1, :, :] > 8.0] = 0
        mask_flow[flow_mat[:, 0:1, :, :] < -8.0] = 0
        mask_flow[flow_mat[:, 1:, :, :] > 8.0] = 0
        mask_flow[flow_mat[:, 1:, :, :] < -8.0] = 0
        flow_mat[mask_flow.repeat(1, 2, 1, 1) == 0] = 0

        # print('grid_mat[%d, %d]: ' %(h, w), grid_pro_mat[:, h, w, :])
        # print('cam_dst_mat[%d, %d]: ' % (h, w), cam_dst_mat[:, :, h, w])
        # print('cam_ori_mat[%d, %d]: ' % (h, w), cam_ori_mat[:, :, h, w])
        # print('mask_flow[%d, %d]: ' % (h, w), mask_flow[:, :, h, w])
        # print('others: ', cor_xy[:, :, 379, 744])

        # op_center
        # self.op_center = self.get_op_center(flow_mat, mask_flow)
        # self.set_cam_info(self.op_center)

        return flow_mat, mask_flow

    def get_op_center(self, flow_mat, mask_flow):
        self.op_center = np.array([1427.295, 173.2341])
        # delta_xc, delta_yc = flow_mat[:, 0, :, :], flow_mat[:, 1, :, :]
        # xc_t0, yc_t0 = self.cam_ori_mat[:, 0, :, :], self.cam_ori_mat[:, 1, :, :]
        # b_mat = delta_yc * xc_t0 - delta_xc * yc_t0
        # b = b_mat.masked_select(mask_flow.byte()).reshape(-1, 1)
        # # print('b', torch.max(b), torch.min(b))
        #
        # A1 = delta_xc.masked_select(mask_flow.byte())
        # A2 = delta_yc.masked_select(mask_flow.byte())
        # A = torch.stack((A2, -A1), dim=1)
        # # print('A:', torch.max(A), torch.min(A))
        #
        # center = np.linalg.lstsq(a=A.cpu().numpy(), b=b.cpu().numpy(), rcond=-1)[0]
        # self.op_center = center
        # print('center:', center)
        return self.op_center

    def set_cam_info(self, center):
        x = np.sqrt((center[0] - self.dx)**2 + (center[1] - self.dy) ** 2 + self.f ** 2)
        self.para.L = torch.tensor([x])
        if self.is_cuda:
            self.para.L = self.para.L.cuda()

        cam_ori_mat_norm = self.cam_ori_mat.clone()
        cam_ori_mat_norm[:, 0, :, :] = cam_ori_mat_norm[:, 0, :, :] - self.dx
        cam_ori_mat_norm[:, 1, :, :] = cam_ori_mat_norm[:, 1, :, :] - self.dy
        tmp_square = cam_ori_mat_norm[:, 0, :, :]**2 + cam_ori_mat_norm[:, 1, :, :]**2
        self.para.Fx = torch.sqrt(tmp_square + self.f ** 2).unsqueeze(0)
        if self.is_cuda:
            self.para.Fx = self.para.Fx.cuda()

        self.para.Tx = torch.sqrt(self.para.L ** 2 - self.para.Fx ** 2)
        if self.is_cuda:
            self.para.Tx = self.para.Tx.cuda()

        self.para.CosF = self.f / self.para.Fx
        if self.is_cuda:
            self.para.CosF = self.para.CosF.cuda()

        cam_ori_mat_op = self.cam_ori_mat.clone()
        cam_ori_mat_op[:, 0, :, :] = cam_ori_mat_op[:, 0, :, :] - center[0].item()
        cam_ori_mat_op[:, 1, :, :] = cam_ori_mat_op[:, 1, :, :] - center[1].item()
        sign_mat = cam_ori_mat_op[:, 0, :, :].unsqueeze(1).sign()
        self.para.Xop = torch.sqrt(cam_ori_mat_op[:, 0, :, :]**2 + cam_ori_mat_op[:, 1, :, :]**2).unsqueeze(0)
        self.para.Xop = self.para.Xop * sign_mat
        if self.is_cuda:
            self.para.Xop = self.para.Xop.cuda()

    def alpha2disp(self, cos_alpha, sin_alpha, mask_flow):
        # cos_alpha = torch.cos(alpha_mat)
        # sin_alpha = torch.sin(alpha_mat)
        r_mat = (self.para.Xop * self.para.Fx) / (cos_alpha * self.para.Fx
                                                  + sin_alpha * (self.para.Tx - self.para.Xop))
        depth_x = r_mat * sin_alpha + self.para.Fx
        # depth_mat = depth_x * self.para.CosF
        disp_mat = (self.para.Fx * self.para.L) / depth_x
        # print('disp_mat: ', disp_mat.shape)
        # print('mask_flow: ', mask_flow.shape)
        disp_mat[mask_flow == 0] = 0
        # h, w = 600, 700
        # print('cos_alpha[%d, %d]: ' % (h, w), cos_alpha[:, :, h, w])
        # print('sin_alpha[%d, %d]: ' % (h, w), sin_alpha[:, :, h, w])
        # print('mask_flow[%d, %d]: ' % (h, w), mask_flow[:, :, h, w])
        # print('disp_mat [%d, %d]: ' % (h, w), disp_mat[:, :, h, w])
        return disp_mat

    def alpha2disps(self, alpha_mat, flow_mat, mask_flow):
        # print('alpha_mat: ', torch.min(alpha_mat), torch.max(alpha_mat))
        # print('flow_mat: ', torch.min(flow_mat), torch.max(flow_mat))
        # print('mask_flow: ', torch.min(mask_flow), torch.max(mask_flow))
        if isinstance(alpha_mat, (tuple, list)):
            cos_alpha, sin_alpha = alpha_mat
        else:
            cos_alpha = torch.cos(alpha_mat)
            sin_alpha = torch.sin(alpha_mat)
        # t-1 depth:
        disp_mat = self.alpha2disp(cos_alpha, sin_alpha, mask_flow)
        # Create t alpha
        cam_dst_mat = (self.cam_ori_mat + flow_mat).long()
        cam_dst_idx = (cam_dst_mat[:, 1, :, :] * self.W + cam_dst_mat[:, 0, :, :]).unsqueeze(1).reshape(
            flow_mat.shape[0], 1, self.H * self.W)
        cam_dst_idx = torch.clamp(cam_dst_idx, 0, self.H * self.W - 1)
        # print('cam_dst_idx: ', torch.min(cam_dst_idx), torch.max(cam_dst_idx))
        # print('alpha_mat: ', torch.min(alpha_mat), torch.max(alpha_mat))
        cos_vec = cos_alpha.reshape(cos_alpha.shape[0], cos_alpha.shape[1], self.H * self.W)
        sin_vec = sin_alpha.reshape(sin_alpha.shape[0], sin_alpha.shape[1], self.H * self.W)
        # print('alpha_vec: ', torch.min(alpha_vec), torch.max(alpha_vec))
        cos_vec_t = torch.zeros_like(cos_vec).scatter_(dim=2, index=cam_dst_idx, src=cos_vec)
        sin_vec_t = torch.zeros_like(sin_vec).scatter_(dim=2, index=cam_dst_idx, src=sin_vec)
        # alpha_vec_t = torch.zeros_like(alpha_vec).scatter_(dim=2, index=cam_dst_idx, src=alpha_vec)
        # print('alpha_vec_t: ', torch.max(alpha_vec_t))
        # alpha_mat_t = alpha_vec_t.reshape(alpha_mat.shape)
        cos_alpha_t = cos_vec_t.reshape(cos_alpha.shape)
        sin_alpha_t = sin_vec_t.reshape(sin_alpha.shape)
        mask_flow_vec = mask_flow.reshape(mask_flow.shape[0], mask_flow.shape[1], self.H * self.W)
        mask_flow_vec_t = torch.zeros_like(mask_flow_vec).scatter_(dim=2, index=cam_dst_idx, src=mask_flow_vec)
        mask_flow_t = mask_flow_vec_t.reshape(mask_flow.shape)
        # print('alpha_mat_t: ', alpha_mat_t.shape)
        # print('mask_flow_t: ', mask_flow_t.shape)
        # print(torch.min(mask_flow_t), torch.max(mask_flow_t))
        # disp_mat_t = None
        disp_mat_t = self.alpha2disp(cos_alpha_t, sin_alpha_t, mask_flow_t)
        return disp_mat, disp_mat_t, mask_flow_t

    def disp2alpha(self, disp_cam, mask_cam):  # From ground-truth to alpha estimation
        # disp -> depth
        depth_cam = (self.para.Fx * self.para.L) / disp_cam
        # depth_cam -> depth_cam_x
        # depth_cam_x = depth_cam / self.para.CosF
        # Calculate sin, cos
        sin_l = depth_cam - self.para.Fx
        cos_l = self.para.Tx + depth_cam / self.para.Fx * (self.para.Xop - self.para.Tx)
        length = torch.sqrt(cos_l**2 + sin_l**2)
        cos_alpha = cos_l / length
        sin_alpha = sin_l / length
        cos_alpha[mask_cam == 0] = 0
        sin_alpha[mask_cam == 0] = 0
        # h, w = 600, 700
        # print('disp_cam [%d, %d]: ' % (h, w), disp_cam[:, :, h, w])
        # print('mask_cam [%d, %d]: ' % (h, w), mask_cam[:, :, h, w])
        # print('cos_alpha[%d, %d]: ' % (h, w), cos_alpha[:, :, h, w])
        # print('sin_alpha[%d, %d]: ' % (h, w), sin_alpha[:, :, h, w])
        return cos_alpha, sin_alpha

    def disp2depths(self, disp_cam, disp_cam_t, mask_flow, mask_flow_t):
        depth_mat = self.f_tvec_mul / disp_cam
        depth_mat[mask_flow == 0] = 0
        depth_mat_t = self.f_tvec_mul / disp_cam_t
        depth_mat_t[mask_flow_t == 0] = 0
        # Scale here

        return depth_mat, depth_mat_t
