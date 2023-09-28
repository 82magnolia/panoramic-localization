from log_utils import load_logger
import sys
import math
import numpy as np


log_name = sys.argv[1]
lg = load_logger(log_name)

if 'result' in log_name:
    lg.t_thres = float(sys.argv[2])
    lg.r_thres = float(sys.argv[3])

    if sys.argv[-1] == 'acc':
        lg.calc_statistics('total')
        print(round(lg.stat_dict['total']['accuracy'], 2))
    elif sys.argv[-1] == 'trans':
        lg.calc_statistics('total')
        print(round(lg.stat_dict['total']['median_t_error'], 2))
    elif sys.argv[-1] == 'rot':
        lg.calc_statistics('total')
        print(round(lg.stat_dict['total']['median_r_error'], 2))
    else:
        lg.calc_statistics('total')
        lg.print_stat('total')

elif 'error_dict' in log_name:
    for k in lg.keys():
        if 'avg' in k:
            if math.isnan(lg[k]):
                orig_data = np.array(lg[k.replace('avg_', '')])
                fixed_val = orig_data[np.bitwise_not(np.isnan(orig_data))].mean()
                print(f"{k}: {fixed_val}")
            else:
                print(f"{k}: {lg[k]}")

