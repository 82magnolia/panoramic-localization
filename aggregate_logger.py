import glob
import sys
from log_utils import load_logger
import os
import numpy as np

root_str = sys.argv[1]  # Specify the prefix to search upon
t_thres = float(sys.argv[2])
r_thres = float(sys.argv[3])
log_dirs = glob.glob(root_str + "*")

t_list = []
r_list = []
for ld in log_dirs:
    lg = load_logger(os.path.join(ld, 'result.pkl'))
    t_list += lg.total_error_dict['t_error_list']
    r_list += lg.total_error_dict['r_error_list']

t = np.array(t_list)
r = np.array(r_list)

print("Median (t): ", np.median(t))
print("Median (r): ", np.median(r))
print("Accuracy : ", ((t < t_thres) & (r < r_thres)).sum() / t.shape[0])
