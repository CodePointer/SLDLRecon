import sys
import numpy as np


def main(out_m_file, out_d_file):
    cam_height = 1024
    cam_width = 1280
    cam_mat = np.loadtxt('cam_mat.txt')
    pro_mat = np.loadtxt('pro_mat.txt')
    rot_mat = np.loadtxt('rot_mat.txt')
    trans_vec = np.loadtxt('trans_vec.txt').reshape((3, 1))

    pro_matrix = np.matmul(pro_mat, np.hstack((rot_mat, trans_vec)))
    f_x = cam_mat[0, 0]
    f_y = cam_mat[1, 1]
    d_x = cam_mat[0, 2]
    d_y = cam_mat[1, 2]

    para_m = np.zeros(shape=(3, cam_height, cam_width))
    for h in range(0, cam_height):
        for w in range(0, cam_width):
            tmp_vec = np.array([(w - d_x) / f_x, (h - d_y) / f_y, 1]).transpose()
            res_vec = np.matmul(pro_matrix[:, :3], tmp_vec)
            para_m[:, h, w] = res_vec.copy()
        if (h + 1) % 64 == 0:
            print('(%d/%d) finished.' % (h + 1, cam_height))
    para_d = pro_matrix[:, 3].copy()

    np.save(out_m_file + '.npy', para_m)
    np.save(out_d_file + '.npy', para_d)


if __name__ == '__main__':

    out_m_file = 'para_M'
    if len(sys.argv) >= 2:
        out_m_file = sys.argv[1]

    out_d_file = 'para_D'
    if len(sys.argv) >= 3:
        out_d_file = sys.argv[2]

    main(out_m_file, out_d_file)
