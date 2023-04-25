import numpy as np
import h5py
import cv2
import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0,1,2,3" # "0"
import matplotlib.pyplot as plt
from copy import deepcopy
from core.median_filter import apply_median_filter_gpu_simple
from core.CD_measure import *
from skimage.metrics import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json

pad = 150
noisy_f_num_list = ['F01', 'F02', 'F04', 'F08','F16','F32']
CD_dict ={}
target = 'F16_v2'
with open(f'./result_data/hole_info_{target}.txt', 'r') as f:
    hole_info = json.load(f)
def return_img(img_dict): 
    img_info = {}
    for set_num in img_dict.keys():
        for f_num in img_dict[set_num].keys():
            if f_num not in noisy_f_num_list:
                continue
            for idx,img in enumerate(img_dict[set_num][f_num]):
                img_info = {'set_num' : set_num, 'f_num' : f_num, 'idx' : idx}
                yield img_info, img
img_dict = np.load("./intermediate_result/full_img_dict.npy",allow_pickle=True).item()
CD_dict = {}
collect_metric_list = ["avg_min_CD", "avg_max_CD"]

method_list = ["median"]
print_img,save_img = False, True


for img_info, img in return_img(img_dict):

    print(img_info, img.shape) # (1,x,y)
    set_num, f_num, idx = img_info["set_num"], img_info["f_num"], img_info["idx"]
    key = f'{img_info["set_num"]}_{img_info["f_num"]}_{img_info["idx"]:02d}'
    img_hole_info = hole_info[set_num]['F08'][idx]
    CD_dict[key] = {}
    for method in method_list:
        target_model = apply_median_filter_gpu_simple
        img_info['method'] = 'median_baseline'
        print(f"===== {method}  CD measure =====")
        CD_info_ours = CD_process(target_model, img, img_info, img_hole_info, pad,print_img,save_img)
        CD_dict = get_metric(CD_dict, CD_info_ours,key, collect_metric_list, method)

CD_dict = pd.DataFrame(CD_dict)
CD_dict.to_csv(f"./result_data/CD_dict_median_filter_F1~F32.csv")

# post processing
def f_num_generator(first_fnumber,last_fnumber):
    f_num_list = []
    val = first_fnumber
    while True:
        f_num_list.append(f"F{val:02d}")
        if val == last_fnumber:
            break
        val *=2
    return f_num_list
def return_f_num_list(set_number):
    set_number = int(set_number)
    if set_number <= 4 :
        number_of_img = 16
        return number_of_img,f_num_generator(8,32)
    else :
        number_of_img = 2
        return number_of_img,f_num_generator(1,32)

arranged_CD_info = {}
metric_list = ['avg_min_CD', 'avg_max_CD']
for metric in metric_list:
    arranged_CD_info[metric] = {}
set_num_list = []
for i in range(1,11):
    set_num_list.append(f"SET{i:02d}")
print(set_num_list)

for idx,set_num in enumerate(set_num_list):
    number_of_img, f_num_list = return_f_num_list(set_num[3:])
    print(f_num_list)
    for img_idx in range(number_of_img):
        key_list = []
        for f_num in f_num_list:
            key = f"{set_num}_{f_num}_{img_idx:02d}"
            # key_list.append(key)
            # print(key, CD_info[key])
            # print(arranged_CD_info[metric_list[0]])
            for i in range(2):
                arranged_CD_info[metric_list[i]][key] = CD_dict.iloc[i][key]
        # print(key_list,"\n",CD_info.iloc[0][key_list])
        # print(key_list,"\n",CD_info.iloc[1][key_list])
pd.DataFrame(arranged_CD_info).to_csv('./result_data/CD_dict_median_filter_F1~F32_arranged.csv')