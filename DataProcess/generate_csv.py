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


def main(out_file):

    # Some parameters:
    main_path = 'E:/SLDataSet/Thing10K/flow_dataset'
    model_num = 50      # Start from 1
    frame_num = 100     # Start from 0

    prefix_set = []
    for m_idx in range(1, model_num + 1):
        for f_idx in range(0, frame_num):
            prefix_set.append('m%02df%03d' % (m_idx, f_idx))
    data_types = [('depth_cam', '.npy'),
                  ('depth_pro1', '.npy'),
                  ('depth_pro2', '.npy'),
                  ('mask_cam', '.png'),
                  ('mask_pro1', '.png'),
                  ('mask_pro2', '.png'),
                  ('flow1_cv', '.npy'),
                  ('flow2_cv', '.npy'),
                  ('mask_flow1', '.png'),
                  ('mask_flow2', '.png'),
                  ('xy_pro1_cv', '.npy'),
                  ('xy_pro2_cv', '.npy'),
                  ('xy_cam_p1v', '.npy'),
                  ('xy_cam_p2v', '.npy')]

    total_list = []
    for prefix in prefix_set:
        line = [main_path + '/', prefix + '_']
        for data_type in data_types:
            line.append(data_type[0] + data_type[1])
        total_list.append(line)

    with open(out_file + '.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(total_list)

    header_dict = {}
    for i in range(0, len(data_types)):
        header_dict[data_types[i][0]] = i + 2
    np.save(out_file + '.npy', header_dict)
    print(header_dict)

    return


if __name__ == '__main__':
    assert len(sys.argv) >= 1

    csv_name = 'DataNameList'
    if len(sys.argv) >= 2:
        csv_name = str(sys.argv[1])

    main(out_file=csv_name)
