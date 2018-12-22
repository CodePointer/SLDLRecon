import torch
import visdom


def volume_visual(gt_v, res_v, mask, vis, win_imgs, win_fig, nrow=4):
    # Detach & cpu
    gt_v = gt_v.detach().cpu()
    res_v = res_v.detach().cpu()
    mask = mask.detach().cpu()

    # Input size
    vol_shape = gt_v.shape  # [N, D, H, W]
    N = vol_shape[0]
    D = vol_shape[1]
    H = vol_shape[2]
    W = vol_shape[3]

    # Make disp for 64
    disp_range = (torch.Tensor(range(D, 0, -1)) - 1) / D  # [D]
    disp_vol = torch.stack([disp_range] * N, 0)  # [N, D]
    disp_vol = torch.stack([disp_vol] * H, 2)
    disp_vol = torch.stack([disp_vol] * W, 3)  # [N, D, H, W]

    # Calculate disp for N set
    gt_disp_set = torch.sum(input=gt_v * disp_vol, dim=1, keepdim=True)  # [N, 1, H, W]
    res_disp_set = torch.sum(input=res_v * disp_vol, dim=1, keepdim=True)  # [N, 1, H, W]
    res_disp_set = res_disp_set * mask.float()
    show_disp_set = torch.cat((gt_disp_set, res_disp_set), dim=3)
    show_disp_set = torch.nn.functional.interpolate(input=show_disp_set, scale_factor=2.0, mode='nearest')
    vis.images(show_disp_set, nrow=nrow, padding=2, win=win_imgs)

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
