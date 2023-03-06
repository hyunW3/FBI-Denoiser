from cmath import nan
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import numpy as np
import pandas as pd
from glob import glob
import json
import sys

CD_info = {}
for path in glob("../CD_info*.txt"):
    print(path)
    with open(path, 'r') as f:
        data = json.load(f)
    CD_info["_".join(path.split("_")[2:])[:-4]] = data
    # break

target = 'F16_v2_150'
source = 'median_filter'
total_num = 0
success = 0 
diff_max_list = []
diff_min_list = []


orig_stdout = sys.stdout
orig_stderr = sys.stderr
f = open(f"./output_diff_{target}_vs_{source}.txt",'w')
sys.stderr = f
sys.stdout = f


print(f"=== info {target} vs {source} ====")
for set_num in CD_info[target].keys():
    dst_dict = CD_info[target][set_num]
    src_dict = CD_info[source][set_num]
    for f_num in dst_dict.keys():
        dst_dict_f = dst_dict[f_num]
        src_dict_f = src_dict[f_num]
        for idx in dst_dict_f.keys():
            dst_info = dst_dict_f[idx]
            src_info = src_dict_f[idx]

            print(f"=== {set_num} {f_num} {idx}th image ====")
            print(f"=== info {target} vs {source} ====")
            print(f"avg_min_CD : {dst_info['avg_min_CD']:.4f} vs {src_info['avg_min_CD']:.4f}")
            diff_avg_min = dst_info['avg_min_CD'] - src_info['avg_min_CD']
            print(f"-> Difference : {diff_avg_min:.4f}")
            print(f"avg_max_CD : {dst_info['avg_max_CD']:.4f} vs {src_info['avg_max_CD']:.4f}")
            diff_avg_max = dst_info['avg_max_CD'] - src_info['avg_max_CD']
            print(f"-> Difference : {diff_avg_max:.4f}")
            if (not np.isnan(diff_avg_max) ) and (not np.isnan(diff_avg_min) is False):
                print("True")
                diff_max_list.append(diff_avg_max)
                diff_min_list.append(diff_avg_min)
                total_num += 1
                if diff_avg_max <= 1 and diff_avg_min <= 1 :
                    success +=1 
            print("")
        print("-------------------------------")
print(f"success rate : {success}/{total_num} ({100*success/total_num}%)")
print(f"avg max diff : {np.mean(diff_max_list):.4f} ")
print(f"avg max diff range: {np.min(diff_max_list):.4f} ~ {np.max(diff_max_list):.4f} ")
print(f"avg min diff : {np.mean(diff_min_list):.4f} ")
print(f"avg min diff range: {np.min(diff_min_list):.4f} ~ {np.max(diff_min_list):.4f} ")

CD_diff_info = {'min_CD_diff' : diff_min_list, 'max_CD_diff' : diff_max_list}
with open(f'./CD_diff_info_{target}vs{source}.txt', 'w') as f:
    f.write(json.dumps(CD_diff_info,indent="\t"))
print("save complete",target)

sys.stdout = orig_stdout
sys.stderr = orig_stderr

print("== measuring finish ===")
