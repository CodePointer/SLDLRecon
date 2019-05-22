# -*- coding: utf-8 -*-

"""
Some useful functional process.

Import when using.

This file is created by QIAO RUKUN on Feb.21, 2019.
"""

import torch
import numpy as np


def check_nan(network, save_path=None):
    """Check if network has nan parameters. Save if needed.

    Param:
        :param network: The model need to be checked.
        :param save_path: If path was given, the error network will be saved.

    Return:
        :return: None

    Raise:
        AssertionError
    """

    flag = True
    param_list = list(network.parameters())
    for idx in range(0, len(param_list)):
        param = param_list[idx]
        if torch.isnan(param).any().item():
            flag = False
            break
        if param.grad is not None and torch.isnan(param.grad).any().item():
            flag = False
            break
    try:
        assert flag
    except AssertionError as inst:
        if save_path:
            torch.save(network.state_dict(), save_path)
        print(inst)
        raise


class AverageMeter(object):
    """Computes and stores the average and current value

    Usage:
        losses = AverageMeter()
        losses.update(val_0)
        ...
        losses.update(val_n)
        print(losses.avg)
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def depth2rgb(depth_mat, mask_mat=None):
    """Convert torch depth result to rgb result

    :param depth_mat:   The input depth mat. Normalized to 0, 1
    :param mask_mat:    The mask mat for invalid depth points. (Optional)
    :return: rgb_mat, valid rgb_mat for visualization.
    """
    depth_mat_cpu = depth_mat.cpu()
    if mask_mat is not None:
        depth_mat_cpu[mask_mat == 0] = 0
    max_val = torch.max(depth_mat_cpu.masked_select(depth_mat_cpu > 0))
    min_val = torch.min(depth_mat_cpu.masked_select(depth_mat_cpu > 0))
    rgb_mat = depth_mat_cpu.repeat(3, 1, 1).clamp(0, 1)
    rgb_mat = (rgb_mat - min_val) / (max_val - min_val)
    return rgb_mat


def flow2rgb(flow_map, max_value, mask_mat=None):
    """Convet flow mat to rgb result for tensorboard.

    :param flow_map:    The input flow map. Shape is [2, H, W]
    :param max_value:   Max value for visualization.
    :param mask_mat:    Valid mask for flow map. (Optional)
    :return:
    """
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    if mask_mat is not None:
        flow_map_np[mask_mat.reshape(1, h, w) == 0] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    normalized_flow_map = flow_map_np / max_value
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    final_rgb_map = rgb_map.clip(0, 1).transpose(1, 2, 0)
    return final_rgb_map
