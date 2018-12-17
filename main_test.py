from image_data_set import CameraDataSet, imshow
from torch.utils.data import DataLoader
from my_net import MyNet
# from patch_net import PatchNet
from my_loss import MyLoss
import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    print('Test process.')

    # Step 1: Set data_loader, create net
    root_path = './SLDataSet/20181204/'
    batch_size = 1
    camera_dataset = CameraDataSet(root_path, 'DataNameList.csv')
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Step 0: DataLoader size: %d.' % len(data_loader))
    network = MyNet(root_path=root_path, batch_size=batch_size, alpha=1024.0, beta=700)
    network = network.cuda()
    if os.path.exists('./model_volume.pt'):
        print('Model found. Import parameters.')
        network.load_state_dict(torch.load('./model_volume.pt'))
        network.eval()
    else:
        print('Model not found. Train from start.')

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-7, momentum=0.9)
    # device = torch.device('cuda:0')
    print('Step 1: Initialize finished.')

    # Step 2: Generate pattern and Set Loss for that
    criterion = criterion.cuda()
    pattern = camera_dataset.GetPatternTensor()
    pattern = torch.stack([pattern] * batch_size, 0)  # BatchSize
    pattern = pattern.cuda()

    # Step 3: Test one image
    num = 30
    output_dir = './test_1pic_1210'
    # os.mkdir(output_dir)
    data_iter = iter(data_loader)
    for i in range(0, num):
        data = data_iter.next()
        image = data['image'].cuda()
        disp_c = data['disp_c'].cuda()
        mask_c = data['mask_c'].cuda()
        est_disp_c = network((image, pattern))  # [N, 1, H, W]
        loss = criterion(est_disp_c.masked_select(mask_c), disp_c.masked_select(mask_c))
        print('[%d]Training Loss: ' % i, loss.item())
        # Save
        np_est = (est_disp_c * mask_c.float()).detach().cpu().squeeze().numpy()
        np.savetxt(''.join([output_dir + '/est_disp_c', str(i), '.txt']), np_est)
        np_disp = disp_c.detach().cpu().squeeze().numpy()
        np.savetxt(''.join([output_dir + '/obs_disp_c', str(i), '.txt']), np_disp)
        print('[%d]Finish saving.' % i)


if __name__ == '__main__':
    main()
