from image_data_set import CameraDataSet
from torch.utils.data import DataLoader
from my_net import MyNet
# from patch_net import PatchNet
import torch
import visdom


def main():
    print('Test process.')

    # Step 1: Set data_loader, create net
    # Step 1: Set data_loader, create net, visual
    root_path = './SLDataSet/20181204/'
    batch_size = 1
    down_k = 5
    camera_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k) + 'v.csv', K=down_k)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Step 0: DataLoader size: %d.' % len(data_loader))
    network = MyNet(root_path=root_path, batch_size=batch_size, K=down_k)
    network = network.cuda()
    network.load_state_dict(torch.load('./model_volume.pt'))
    network.train()
    vis = visdom.Visdom(env='K5_Network_Test')

    print('Step 1: Initialize finished.')

    # Step 2: Generate pattern and Set Loss for that
    pattern = camera_dataset.GetPatternTensor()
    pattern = torch.stack([pattern] * batch_size, 0)  # BatchSize
    pattern = pattern.cuda()

    # Step 3: Test <num> image
    num = 30
    output_dir = './test_1pic_1210'
    # os.mkdir(output_dir)
    data_iter = iter(data_loader)
    for i in range(0, num):
        data = data_iter.next()
        image = data['image'].cuda()
        coarse_vol = data['disp_v'].cuda()
        coarse_mask = data['mask_c'].cuda()
        volume_prob = network((image, pattern))  # [N=1, D=64, H=32, W=40]

        # Save
        np_mask = (coarse_mask.byte()).detach().cpu().squeeze().numpy()
        np_mask.tofile(''.join([output_dir + '/mask_c', str(i), '.bin']))
        np_gt = (coarse_vol.float()).detach().cpu().squeeze().numpy()
        np_gt.tofile(''.join([output_dir + '/gt_c', str(i), '.bin']))
        np_prob = (volume_prob.float()).detach().cpu().squeeze().numpy()
        np_prob.tofile(''.join([output_dir + '/vol_c', str(i), '.bin']))

        print('[%d]Finish saving.' % i)


if __name__ == '__main__':
    main()
