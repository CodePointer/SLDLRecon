# Data Process Part

## File illustration

### generate_csv.py

Used for generate csv file. File is used for `dataloader`.

Need:
- Nothing.

Output:
- `DataNameList.csv`
- `DataNameList.npy` (header)

### generate_epiinfo.py

Generate `para_M, para_D` mat for epipolar match at original resolution.

Need:
- `cam_mat.txt`, `pro_mat.txt`, `rot_mat.txt`, `trans_vec.txt`

Output:
- `para_M.npy`, `para_D.npy`

### sparse_groundtruth.py

Generate `disp_c` matrix for sparse version supervised training.

Need:
- Generated csv file from `generate_csv.py`
- disp mat at origin resolution.

Output:
- `disp_c.npy`

### generate_shading.py

Generate `shade` matrix for image rendering.

Need:
- Generated csv file from `generate_csv.py`
- `cam_mat.txt`
- disp mat at origin resolution.

Output:
- `shade.png`

## Run step

`generate_epiinfo.py` \
`generate_csv.py` \
`sparse_groundtruth.py` \
`generate_shading.py`