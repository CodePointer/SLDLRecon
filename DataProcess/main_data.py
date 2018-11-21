import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Set several parameters
    csv_path = '../SLDataSet/20181112/'
    folder_path = 'DataSet1/'
    img_name = 'cam_img'
    pro_name = 'x_pro'
    csv_name = 'patch_list.csv'
    start_frm_idx = 0
    step_size = 10
    frm_size = 1000

    # Create data_list
    data_list = []
    for frm_idx in range(start_frm_idx, start_frm_idx + frm_size, step_size):
        image = plt.imread(''.join([csv_path, folder_path, img_name, str(frm_idx), '.png']))
        x_pro = np.loadtxt(''.join([csv_path, folder_path, pro_name, str(frm_idx), '.txt']))
        image_height, image_width = image.shape[:2]
        for h in range(0, image_height):
            for w in range(0, image_width):
                if x_pro[h, w] > 0:
                    tmp_list = [str(frm_idx), str(h), str(w), str(x_pro[h, w])]
                    data_list.append(tmp_list)
        print('%d frame finished. (size: %d)' % (frm_idx, len(data_list)))

    # Write to csv_file
    df = pd.DataFrame(data_list)
    df.to_csv(csv_path + csv_name, index=False, header=False)
    print('Write over. Total_num=%d' % df.shape[0])


if __name__ == "__main__":
    main()
