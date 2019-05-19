import sys

sys.path.append('../')

from History.data_set import CameraDataSet
from torch.utils.data import DataLoader
import torch.nn.functional as func
import numpy as np
from matplotlib import pyplot as plt


def generate_sparse_gt(root_path, down_k):
    # Step 1: Set data_loader
    batch_size = 1
    workers = 8
    opt_header = ('mask_mat', 'disp_mat')
    opt = {'header': opt_header}
    camera_dataset = CameraDataSet(root_path, 'TestDataList' + str(down_k), down_k=down_k, opts=opt)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    print('Step 0: DataLoader size: %d.' % len(data_loader))

    # vis_env = 'K' + str(down_k) + '_Network_Disp_Gen'
    # vis = visdom.Visdom(env=vis_env)
    print('Step 1: Initialize finished.')

    # Step 2: Process all data by looping
    report_period = 40
    for i, data in enumerate(data_loader, 0):
        if i < 1000:
            continue
        # Get data:
        data_idx = data['idx']
        dense_mask = data['mask_mat']
        dense_disp = data['disp_mat']
        dense_mask = dense_mask.cuda().float()
        dense_disp = dense_disp.cuda()

        # Pooling
        kernel = pow(2, down_k)
        sparse_disp = func.avg_pool2d(input=dense_disp, kernel_size=kernel)
        sparse_mask = func.avg_pool2d(input=dense_mask, kernel_size=kernel)
        sparse_disp = sparse_disp / sparse_mask
        sparse_disp[sparse_mask == 0] = 0
        sparse_mask[sparse_mask > 0] = 1
        sparse_mask = sparse_mask.byte()

        # Save disp_c
        sparse_disp_name = camera_dataset.get_path_by_name('disp_c', data_idx)
        sparse_disp_mat = sparse_disp.cpu().squeeze().numpy()
        np.save(sparse_disp_name, sparse_disp_mat)  # disp_c: [H_c, W_c]

        # Save mask_c
        sparse_mask_name = camera_dataset.get_path_by_name('mask_c', data_idx)
        sparse_mask_mat = sparse_mask.cpu().squeeze().numpy()
        plt.imsave(sparse_mask_name, sparse_mask_mat, cmap='Greys_r')

        # Visualization and report
        train_num = '.'
        if i % report_period == report_period - 1:
            # vis.image(sparse_disp, win='Disp', env=vis_env)
            report_info = '[%4d/%d]' % (i + 1, len(data_loader))
            print(train_num, report_info)
        else:
            print(train_num, end='', flush=True)
    print('Step 3: Finish evaluating.')


def main(argv):
    # Input parameters
    assert len(argv) >= 2

    down_k = int(argv[1])

    generate_sparse_gt(root_path='../SLDataSet/20190209/', down_k=down_k)


if __name__ == '__main__':
    main(sys.argv)
