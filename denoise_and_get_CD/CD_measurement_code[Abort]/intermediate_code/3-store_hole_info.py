import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # "0,1,2,3" # "0"
import numpy as np
import h5py
import cv2
import os, sys
import json
import argparse
from glob import glob
from core.get_args import get_args
from core.produce_denoised_img import produce_denoised_img_no_crop
from core.utils import TedataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from core.watershed import *
from core.watershed import watershed,watershed_per_img,watershed_original
from core.median_filter import apply_median_filter_cpu, apply_median_filter_gpu, apply_median_filter_gpu_simple


# remember hole info

target_list = []#[f"median_filter"]
for target in [['F16','v2']]:
    target = f"{target[0]}_{target[1]}"
    target_list.append(target)

for target in target_list:
    if target != 'median_filter':
        target_fnum = target[:3]
    # print(target)
    data_path = f"./intermediate_result/denoised_img_{target}_img_dict.npy"
    print(data_path.split("/")[2:])
    img_dict = np.load(data_path,allow_pickle=True).item()
    hole_info_dict = {}
    frame = []
    for set_num in img_dict.keys():
        # print(img_dict[set_num].keys())
        hole_info_dict[set_num] = {}
        folder_name = f"../tmp/segmentation_img/{target}"
        os.makedirs(f"{folder_name}/{set_num}", exist_ok=True)
        for f_num in img_dict[set_num].keys():
            if target != 'median_filter' and f_num == target_fnum:
                break
            
            # print(denoised_img_dict[set_num][f_num].shape
            hole_info_dict[set_num][f_num] = []
            f_num = f_num[:3]
            for idx,img in enumerate(img_dict[set_num][f_num]):
                img_uint8 = (img[0]*255).astype('uint8') # [:-20]

                hole_info = find_possible_section(img_uint8)
                
                # print(hole_info.shape) 
                
                hole_info_dict[set_num][f_num].append(hole_info)
                    
            print(f"==== {set_num} {f_num} segmentation complete ====")
        
    with open(f'./hole_info_{target}.txt', 'w') as f:
        f.write(json.dumps(hole_info_dict,indent="\t"))
    del hole_info_dict
    print("save complete",target)