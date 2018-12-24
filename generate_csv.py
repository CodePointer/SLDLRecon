import csv
import sys


def main(out_file, k):
    total_list = []
    data_sets = [('DataSetW1', 1000), ('DataSetW2', 1000), ('DataSetW3', 1000), ('DataSetW4', 1000)]
    data_types = [('cam_img', 'cam_img', '.png'),
                  ('disp_mat', 'disp_mat', '.bin'),
                  ('mask_mat', 'mask_mat', '.png'),
                  ('disp_c' + str(k), 'disp_c', '.bin'),
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

    with open(out_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(total_list)

    return


if __name__ == '__main__':
    assert len(sys.argv) >= 2

    down_k = int(sys.argv[1])

    csv_name = 'DataNameList'
    if len(sys.argv) >= 3:
        csv_name = sys.argv[2]
    csv_name += str(down_k) + '.csv'

    main(out_file=csv_name, k=down_k)
