from image_data_set import CameraDataSet, imshow
from torch.utils.data import DataLoader
from patch_net import PatchNet
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def main():
    # Step 1: Set net && Load model
    network = PatchNet(alpha=1.0, beta=0)
    network.load_state_dict(torch.load('./model.pt'))
    network.eval()
    network.cuda()
    print('Step 1: Network initialize finished.')

    # Step 2: Load image
    image_name = './SLDataSet/20181112/DataSet1/cam_img15.png'
    x_pro_name = './SLDataSet/20181112/DataSet1/x_pro15.txt'
    test_image = plt.imread(image_name)
    test_xpro = np.loadtxt(x_pro_name)
    output_xpro = np.array(test_xpro)
    print('Data loading finished.')

    # Step 3: Evaluate
    for h in range(0, 1024):
        for w in range(0, 1280):
            if test_xpro[h, w] > 0:
                part_img = test_image[h - 10:h + 11, w - 10:w + 11, :].copy()
                norm_part_img = torch.from_numpy((part_img.transpose((2, 0, 1)) - 0.5) * 2)
                output_label = network(norm_part_img.unsqueeze(0).cuda())
                output_xpro[h, w] = output_label[0]

    # imshow(output.detach() / 2 + 0.5)
    # np_image = output.detach().cpu().squeeze(0).numpy()
    np.savetxt('output_mat.txt', output_xpro)
    print('Step 3: Evaluation finished.')


if __name__ == '__main__':
    main()
