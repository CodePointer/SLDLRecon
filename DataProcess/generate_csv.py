# -*- coding: utf-8 -*-

"""
This Module is used for generate csv file for dataset.
    Dataset: Thing10K
    Type: model with 100 depth image set. Temporal series.
        Has been processed with 'coord_generator'.
"""

import csv
import sys
import numpy as np


def generate_csv_list(sub_main_path, model_num, frame_num, data_types):
    """
    Generate csv files given model & frame number.

    Param:
        :param main_path: The local folder path for file reading.
        :param model_num: Model number. Range: [1, model_num]
        :param frame_num: Frame number for each model. Range: [0, frame_num)
        :param data_types: data name for every type of data.
    Return:
        :return: data_list
    """
    data_list = []
    for m_idx in range(1, model_num + 1):
        for f_idx in range(0, frame_num):
            prefix = 'm%02df%03d_' % (m_idx, f_idx)
            line = [sub_main_path]
            for name, suffix in data_types:
                line.append(''.join([prefix, name, suffix]))
            data_list.append(line)
    return data_list


def main(out_file_name):

    # data_types = [('depth_cam', '.npy'),
    #               ('depth_pro1', '.npy'),
    #               ('depth_pro2', '.npy'),
    #               ('mask_cam', '.png'),
    #               ('mask_pro1', '.png'),
    #               ('mask_pro2', '.png'),
    #               ('flow1_cv', '.npy'),
    #               ('flow2_cv', '.npy'),
    #               ('mask_flow1', '.png'),
    #               ('mask_flow2', '.png'),
    #               ('xy_pro1_cv', '.npy'),
    #               ('xy_pro2_cv', '.npy'),
    #               ('xy_cam_p1v', '.npy'),
    #               ('xy_cam_p2v', '.npy')]
    data_types = [('depth_cam', '.npy'),
                  ('mask_cam', '.png'),
                  ('flow1_cv', '.npy'),
                  ('mask_flow1', '.png'),
                  ('flow1_cv_est', '.npy')]

    # Save header file
    header_dict = {}
    for i in range(0, len(data_types)):
        header_dict[data_types[i][0]] = i + 1  # 0 pos is local_path
    np.save(out_file_name + '_header.npy', header_dict)

    # Save train_list file
    train_list = generate_csv_list('train_dataset/', model_num=50, frame_num=100, data_types=data_types)
    with open(out_file_name + '_train.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(train_list)

    # Save test_list file
    test_list = generate_csv_list('test_dataset/', model_num=2, frame_num=100, data_types=data_types)
    with open(out_file_name + '_test.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(test_list)

    # Finished print.
    print('Input out_file_name:', out_file_name)
    print('Train dataset: %d' % len(train_list))
    print('Test dataset: %d' % len(test_list))
    print('Header result:')
    for key in header_dict:
        print('    ', key, ':', header_dict[key])

    return


if __name__ == '__main__':
    assert len(sys.argv) >= 1

    csv_name = 'data_list'
    if len(sys.argv) >= 2:
        csv_name = str(sys.argv[1])

    root_path = ''
    if len(sys.argv) >= 3:
        root_path = str(sys.argv[2])

    main(out_file_name=root_path + csv_name)
