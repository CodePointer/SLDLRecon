import csv
import sys
import numpy as np


def main(out_file, k):
    total_list = []
    data_sets = [('DataSetW' + str(i), 1000) for i in range(2, 10, 2)]
    data_types = [('cam_img', 'cam_img', '.png'),
                  ('disp_mat', 'disp_mat', '.bin'),
                  ('disp_cam', 'disp_cam', '.npy'),
                  ('mask_mat', 'mask_mat', '.png'),
                  ('mask_cam', 'mask_cam', '.png'),
                  ('cor_xc', 'cor_xc', '.npy'),
                  ('cor_yc', 'cor_yc', '.npy'),
                  ('mask_pro', 'mask_pro', '.png'),
                  ('shade_mat', 'shade_mat', '.png'),
                  ('disp_c' + str(k), 'disp_c', '.npy'),
                  ('disp_v' + str(k), 'disp_v', '.bin'),
                  ('mask_c' + str(k), 'mask_c', '.png'),
                  ('disp_out' + str(k), 'disp_out', '.npy'),
                  ('est_img' + str(k), 'est_img', '.png')]

    for data_set in data_sets:
        set_num = data_set[1]
        for idx in range(0, set_num):
            tmp_list = list()
            tmp_list.append(data_set[0] + '/')
            for data_type in data_types:
                file_name = ''.join((data_type[0], '/', data_type[1], str(idx), data_type[2]))
                tmp_list.append(file_name)
            total_list.append(tmp_list)

    with open(out_file + '.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(total_list)

    header_dict = {}
    for i in range(0, len(data_types)):
        header_dict[data_types[i][1]] = i + 1
    np.save(out_file + '.npy', header_dict)
    print(header_dict)

    return


if __name__ == '__main__':
    assert len(sys.argv) >= 2

    down_k = int(sys.argv[1])

    csv_name = 'DataNameList'
    if len(sys.argv) >= 3:
        csv_name = sys.argv[2]
    csv_name += str(down_k)

    main(out_file=csv_name, k=down_k)
