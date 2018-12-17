from image_data_set import CameraDataSet, imshow
from torch.utils.data import DataLoader
from my_net import MyNet
# from patch_net import PatchNet
from my_loss import MyLoss
import torch
import torchvision
import math
import numpy as np
import os
import visdom


def main():
    # Step 1: Set data_loader, create net, visual
    root_path = './SLDataSet/20181204/'
    batch_size = 16
    down_k = 5
    camera_dataset = CameraDataSet(root_path, 'DataNameList' + str(down_k) + 'v.csv', K=down_k)
    data_loader = DataLoader(camera_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Step 0: DataLoader size: %d.' % len(data_loader))
    # data_iter = iter(data_loader)
    # data = data_iter.next()
    # imshow(torchvision.utils.make_grid(image))
    network = MyNet(root_path=root_path, batch_size=batch_size, K=down_k)

    # for param_tensor in network.state_dict():
    #     tmp = network.state_dict()[param_tensor]
    #     print(param_tensor, '\t', tmp.size())
    # print(network.state_dict()['self_out.weight'])
    # return

    # criterion = MyLoss()
    criterion = torch.nn.MSELoss()
    # cos_criterion = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, momentum=0.9)
    # device = torch.device('cuda:0')
    vis = visdom.Visdom(env='K5_Network')
    win_epoch = 'epoch_loss'
    win_report = 'report_loss'
    print('Step 1: Initialize finished.')

    # Step 2: Train and Set Loss for that
    criterion = criterion.cuda()
    report_period = 10
    save_period = 125
    pattern = camera_dataset.GetPatternTensor()
    pattern = torch.stack([pattern] * batch_size, 0)  # BatchSize
    pattern = pattern.cuda()
    loss_list = []

    network = network.cuda()
    if os.path.exists('./model_volume.pt'):
        print('Model found. Import parameters.')
        network.load_state_dict(torch.load('./model_volume.pt'))
        network.train()  # For BN
        # print(network.state_dict()['dn_convs.0.0.bias'])
    else:
        print('Model not found. Train from start.')

    for epoch in range(25, 100):
        running_loss = 0.0
        epoch_loss = 0.0
        loss_times = 0
        epoch_hit_times = 0
        for i, data in enumerate(data_loader, 0):
            # input_image, x_pro_mat = data
            # print(input_value.max(), input_value.min())
            img_have_nan = np.any(np.isnan(data['image']))
            # disp_have_nan = np.any(np.isnan(data['disp_c']))
            disp_have_nan = np.any(np.isnan(data['disp_v']))
            mask_have_nan = np.any(np.isnan(data['mask_c']))
            if img_have_nan or disp_have_nan or mask_have_nan:
                print('Nan detect: img, disp, mask -> ', img_have_nan, disp_have_nan, mask_have_nan)
            image = data['image'].cuda()
            # disparity = data['disp'].cuda()
            # mask = data['mask'].cuda()
            # coarse_disp = data['disp_c'].cuda()
            coarse_disp = data['disp_v'].cuda()
            coarse_mask = data['mask_c'].cuda()

            optimizer.zero_grad()
            volume_prob = network((image, pattern))
            # print(coarse_disp[0, :, 15, 18])
            # print(volume_prob[0, :, 15, 18])
            # print(volume_prob.shape)
            # sparse_disp = network((image, pattern))
            # sparse_disp, dense_disp = network((image, pattern))
            # print("%d loss begin." % i)

            # return

            # Check validation
            # sparse_disp_nan = np.any(np.isnan((sparse_disp * coarse_mask).detach().cpu().numpy()))
            # dense_disp_nan = np.any(np.isnan((dense_disp * mask).detach().cpu().numpy()))
            # if sparse_disp_nan or dense_disp_nan:
            #     print('Nan Error[%i]: ', sparse_disp_nan, dense_disp_nan)

            # loss = criterion(output_data=outputs, input_data=x_pro_mat)
            # loss_dense = criterion(dense_disp * mask, disparity * mask)
            # print("%d loss finished." % i)
            # print(disparity)
            # print(sparse_disp[0, 0, 32, 32])
            # print(feature_mat.requires_grad)
            # print(sparse_disp.requires_grad)
            loss_coarse = criterion(volume_prob.masked_select(coarse_mask), coarse_disp.masked_select(coarse_mask))
            # loss_coarse = criterion(sparse_disp.masked_select(coarse_mask), coarse_disp.masked_select(coarse_mask))
            loss_coarse.backward()
            # for param_tensor in network.state_dict():
            #     tmp = network.state_dict()[param_tensor]
            #     print(param_tensor, '\t', tmp.size())
            # print(network.state_dict()['self_out.weight'])
            # print(network.state_dict()['self_out.bias'])
            # param_list = list(network.self_out.parameters())
            # print(param_list[1])
            param_list = list(network.parameters())
            for idx in range(0, len(param_list)):
                param = param_list[idx]
                if torch.isnan(param).any().item():
                    print('Found nan parameter.', i, '->', idx)
                    print(param.shape)
                    return
                if param.grad is not None and torch.isnan(param.grad).any().item():
                    print('Found nan grad.', i, '->', idx)
                    print(param.grad)
                    return

            # loss_dense.backward()
            loss_add = loss_coarse.item()
            if not np.isnan(loss_add):
                optimizer.step()
                running_loss += loss_add
                epoch_loss += loss_add
                loss_times += 1
                epoch_hit_times += 1
            else:
                print('Warning: Nan detected at set [%d].' % i)
                return

            # loss_add = loss_coarse.item()  # + loss_dense.item()
            # print('Finish [%d] iteration: loss=' % i, loss_add)
            if i % report_period == report_period - 1:
                average = -1.0
                if loss_times > 0:
                    average = running_loss / loss_times
                print('[%d, %4d/%d] MSE-loss: %f' % (epoch + 1, i + 1, len(data_loader), average))
                # Draw:
                vis.line(X=torch.FloatTensor([epoch + i / len(data_loader)]), Y=torch.FloatTensor([average]),
                         win=win_report, update='append')
                running_loss = 0.0
                loss_times = 0
            if i % save_period == save_period - 1:
                torch.save(network.state_dict(), './model_volume.pt')
                print(network.state_dict()['dn_convs.0.0.bias'])
                print('Save model at epoch %d after %3d dataset.' % (epoch + 1, i + 1))
        epoch_average = epoch_loss / epoch_hit_times
        loss_list.append(epoch_average)
        # Draw:
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([epoch_average]), win=win_epoch, update='append')
        print('Average loss for epoch %d: %.2f' % (epoch + 1, epoch_average))
        vis.text('Average loss for epoch %d: %f<br>' % (epoch + 1, epoch_average), win='log', append=True)
        print('History: ', loss_list)
    print('Step 2: Finish training.')

    # Step 3: Save the model
    torch.save(network.state_dict(), './model_volume.pt')
    print('Step 3: Finish saving.')


if __name__ == '__main__':
    main()
