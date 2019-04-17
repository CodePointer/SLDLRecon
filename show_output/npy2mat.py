from scipy import io
import numpy as np
import sys

assert len(sys.argv) >= 2

mat = np.load(sys.argv[1] + '.npy')
io.savemat(sys.argv[1], {sys.argv[1]: mat})
