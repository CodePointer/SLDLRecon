from image_data_set import CameraDataSet, imshow
from torch.utils.data import DataLoader
# from my_net import MyNet
from patch_net import PatchNet
from my_loss import MyLoss
import torch
import torchvision
import math


def main():
    # Step 1: Set data_loader, create net
    camera_dataset = CameraDataSet("./SLDataSet/20181112/", "patch_list.csv")
    data_loader = DataLoader(camera_dataset, batch_size=2048, shuffle=True, num_workers=4)
    # data_iter = iter(data_loader)
    # data = data_iter.next()
    # imshow(torchvision.utils.make_grid(image))
    network = PatchNet(alpha=1.0, beta=0)
    network.load_state_dict(torch.load('./model.pt'))
    network.eval()
    # criterion = MyLoss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    # device = torch.device('cuda:0')
    print('Step 1: Initialize finished.')

    # Step 2: Train and Set Loss for that
    network = network.cuda()
    criterion = criterion.cuda()
    report_period = 100
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # input_image, x_pro_mat = data
            # print(input_value.max(), input_value.min())
            input_image = data['image'].cuda()
            x_label = data['x_pro'].cuda()

            optimizer.zero_grad()
            outputs = network(input_image)
            # print("%d loss begin." % i)

            # loss = criterion(output_data=outputs, input_data=x_pro_mat)
            loss = criterion(outputs, x_label)
            # print("%d loss finished." % i)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % report_period == report_period - 1:
                average = math.sqrt(running_loss / report_period)
                print('[%d, %4d] pixel-loss: %.2f' % (epoch + 1, i + 1, average * 1024.0))
                # print('Max-Min: ', outputs.max(), outputs.min())
                running_loss = 0.0
        torch.save(network.state_dict(), './model.pt')
        print('Save model after epoch %d.' % (epoch + 1))
    print('Step 2: Finish training.')

    # Step 3: Save the model
    torch.save(network.state_dict(), './model.pt')
    print('Step 3: Finish saving.')


if __name__ == '__main__':
    main()
