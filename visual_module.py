import torch
import visdom


def volume_visual(gt_v, res_v, mask, vis, win_imgs, win_fig, nrow=4):
    # Detach & cpu
    # gt_v = gt_v.detach().cpu()
    # res_v = res_v.detach().cpu()
    # mask = mask.detach().cpu()

    # Input size
    vol_shape = gt_v.shape  # [N, D, H, W]
    N = vol_shape[0]
    D = vol_shape[1]
    H = vol_shape[2]
    W = vol_shape[3]

    # Make disp for 64
    disp_range = (torch.Tensor(range(D, 0, -1)) - 1) / D  # [D]
    # disp_vol = torch.stack([disp_range] * N, 0)  # [N, D]
    # disp_vol = torch.stack([disp_vol] * H, 2)
    # disp_vol = torch.stack([disp_vol] * W, 3)  # [N, D, H, W]

    # Calculate disp for N set
    # gt_disp_set = torch.sum(input=gt_v * disp_vol, dim=1, keepdim=True)  # [N, 1, H, W]
    # res_disp_set = torch.sum(input=res_v * disp_vol, dim=1, keepdim=True)  # [N, 1, H, W]
    # res_disp_set = res_disp_set * mask.float()
    # show_disp_set = torch.cat((gt_disp_set, res_disp_set), dim=3)
    # show_disp_set = torch.nn.functional.interpolate(input=show_disp_set, scale_factor=4.0, mode='nearest')
    # vis.images(show_disp_set, nrow=nrow, padding=2, win=win_imgs)

    # Choose one part as plot show
    opt = {'width': 512, 'height': 256}
    for idx in range(0, N, 2):
        mask_part = mask[idx, :, :, :]  # [1, H, W]
        idx_list = mask_part.nonzero()  # [num, 3]
        pt_num = idx_list.shape[0]
        mid_num = int(pt_num / 2)
        gt_vec = gt_v[idx, :, idx_list[mid_num, 1], idx_list[mid_num, 2]]
        res_vec = res_v[idx, :, idx_list[mid_num, 1], idx_list[mid_num, 2]]
        show_vec = torch.stack((gt_vec, res_vec), dim=1)
        vis.line(Y=show_vec, X=torch.stack((disp_range, disp_range), dim=1), opts=opt, win=win_imgs + str(idx))


def disp_visual(gt_c, res_c, mask_c, vis, win_imgs, nrow=4):
    # Detach & cpu
    gt_v = gt_c.detach().cpu()  # [N, 1, H, W]
    res_v = res_c.detach().cpu()
    mask = mask_c.detach().cpu()
    beta = 716
    alpha = 16.0 * 63

    # Calculate disp for N set
    gt_disp_set = (gt_v - beta) / alpha * mask.float()
    res_disp_set = (res_v - beta) / alpha * mask.float()
    show_disp_set = torch.cat((gt_disp_set, res_disp_set), dim=3)
    show_disp_set = torch.nn.functional.interpolate(input=show_disp_set, scale_factor=4.0, mode='nearest')
    vis.images(show_disp_set, nrow=nrow, padding=2, win=win_imgs)


def dense_visual(input_set, output_set, vis, win_img, win_disp):
    """
    :param input_set: (image_obs, image_est, disp_in, mask). [N, 3, H, W] for image. [N, 1, H, W] for disp.
    :param output_set: (disp_gt, disp_res). [N, 1, H, W] for them.
    :param vis: visdom handle
    :param win_img: window for image
    :param win_disp: window for disp
    :return: None
    """
    img_obs = input_set[0][0, :, :, :]
    img_est = input_set[1][0, :, :, :]
    show_img = torch.stack((img_obs, img_est), dim=0)
    show_img = (show_img + 1) / 2
    show_img = torch.nn.functional.interpolate(input=show_img, scale_factor=0.5, mode='nearest')
    vis.images(show_img * 255.0, nrow=2, padding=2, win=win_img)

    mask = input_set[3][0, :, :, :]
    disp_in = input_set[2][0, :, :, :]
    disp_in[mask == 0] = 0
    disp_gt = output_set[0][0, :, :, :]
    disp_gt[mask == 0] = 0
    disp_res = output_set[1][0, :, :, :]
    disp_res[mask == 0] = 0
    show_disp = torch.stack((disp_in, disp_res, disp_gt), dim=0)
    show_disp = torch.nn.functional.interpolate(input=show_disp, scale_factor=0.5, mode='nearest')
    vis.images(show_disp * 255.0, nrow=3, padding=4, win=win_disp)
