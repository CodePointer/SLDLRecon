import os

input_folder = '/media/qiao/数据文档/SLDataSet/20190516/2/dyna/'
output_folder = '/media/qiao/数据文档/SLDataSet/20190516/2/img_pair/'
start_num = 60
stride = 5
end_num = 1000

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for i in range(start_num, end_num, stride):
    first_in = 'dyna_mat%d.png' % i
    first_out = 'cam_img%d_1.png' % i
    if i + stride >= end_num:
        continue
    second_in = 'dyna_mat%d.png' % (i + stride)
    second_out = 'cam_img%d_2.png' % i
    os.system('cp %s %s' % (input_folder + first_in, output_folder + first_out))
    os.system('cp %s %s' % (input_folder + second_in, output_folder + second_out))
