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