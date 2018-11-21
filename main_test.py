from image_data_set import CameraDataSet, imshow
from torch.utils.data import DataLoader
from my_net import MyNet
import matplotlib.pyplot as plt
import numpy as np
from my_loss import MyLoss
import torch
import torchvision


def main():
    # Step 1: Set data_loader, create net
    camera_dataset = CameraDataSet("./SLDataSet/20181112/ImageName.csv", "./SLDataSet/20181112/")
    # data_loader = DataLoader(camera_dataset, batch_size=4, shuffle=True, num_workers=2)
    network = MyNet(alpha=1.0, beta=0)
    print('Step 1: Initialize finished.')

    # Step 2: Load model
    network.load_state_dict(torch.load('./model.pt'))
    network.eval()
    network.cuda()

    # Step 3: Evaluate
    data_loader = DataLoader(camera_dataset, batch_size=1, shuffle=True, num_workers=1)
    data_iter = iter(data_loader)
    data = data_iter.next()
    input_image = data['image']
    x_pro_mat = data['x_pro']
    imshow(input_image.squeeze(0) / 2 + 0.5)

    output = network(input_image.cuda()).squeeze(0)
    print(output.shape)
    print(output.max(), output.min())
    # imshow(output.detach() / 2 + 0.5)
    np_image = output.detach().cpu().squeeze(0).numpy()
    np.savetxt('output_mat.txt', np_image)

    plt.imshow(np_image)
    plt.show()

if __name__ == '__main__':
    main()
