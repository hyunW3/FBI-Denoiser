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
import imageio

import parmap
from functools import partial

target = 'median_filter'
img_dict = np.load(f"./intermediate_result/segmentation_img_{target}.npy",allow_pickle=True).item()
target_fnum_list = ['F01','F02','F04','F08','F16',"F32"]
with open(f'./hole_info_{target}.txt', 'r') as f:
    hole_info = json.load(f)
    
print_img = False
pad = 150
CD_dict = {}
for set_num in img_dict.keys():
    CD_dict[set_num] = {}
    
    for f_num in img_dict[set_num].keys():
        if f_num not in target_fnum_list:
            continue
        
        CD_dict[set_num][f_num] = {}
        f_num = f_num[:3]
        for idx,img in enumerate(img_dict[set_num][f_num]):
            CD_dict[set_num][f_num][idx] = {"min_CD" : [], "max_CD" : []}
            img_hole_info = hole_info[set_num][f_num][idx]
            
            # make image patch
            for i,j in img_hole_info:
                # crop image patch
                pos = {'r_start' : i-pad, 'r_end' : i+pad, 'c_start' : j-pad, 'c_end' : j+pad}
                for key,item in pos.items():
                    if item < 0:
                        pos[key] = 0
                img_patch = img[pos['r_start']:pos['r_end'],pos['c_start']:pos['c_end']].copy()
                if img_patch.shape != (pad*2,pad*2):
                    continue
                
                # leave only circle imagein patch 
                circle_val = img[i,j]
                img_patch[img_patch != circle_val] = 0
                
                (cX,cY), circle_with_marker = find_center(img_patch)
                
                if print_img is True:
                    print("center : ",cX,cY)
                    plt.title(f"pos : {i},{j}")
                    plt.imshow(img_patch)
                    plt.scatter(cX,cY,color='r',marker='.')
                    plt.pause(0.01)
                search_angle = list(range(0,180))# (map(lambda x : int(x),range(0,180)))
                
                # find min/max CD
                partial_func = partial(measure_CD,img=img_patch,cX=cX,cY=cY,debug=False)
                measured_CD = parmap.map(partial_func,range(0,180), pm_processes=12)
                if None in measured_CD:
                    print(f"{set_num}, {f_num} {idx} {i},{j} hole cannot get CD measurement due to None")
                    continue
                min_CD,max_CD = min(measured_CD),max(measured_CD)
                # print(min_CD,max_CD, len(measured_CD))
                
                CD_dict[set_num][f_num][idx]['min_CD'].append(min_CD)
                CD_dict[set_num][f_num][idx]['max_CD'].append(max_CD)
            CD_dict[set_num][f_num][idx]['avg_min_CD'] = np.mean(CD_dict[set_num][f_num][idx]['min_CD'])
            CD_dict[set_num][f_num][idx]['avg_max_CD'] = np.mean(CD_dict[set_num][f_num][idx]['max_CD'])
            # print(f"{set_num}, {f_num} {idx} - Min/Max CD : {CD_dict[set_num][f_num][idx]['avg_min_CD']} ~ {CD_dict[set_num][f_num][idx]['avg_max_CD']}")
            len_CD_measurement = len(CD_dict[set_num][f_num][idx]['min_CD'])
            CD_dict[set_num][f_num][idx]['length'] = len_CD_measurement
            print(f"{set_num}, {f_num} {idx}th image - Min/Max (len : {len_CD_measurement}) CD : \
                {CD_dict[set_num][f_num][idx]['avg_min_CD']:.4f} ~ {CD_dict[set_num][f_num][idx]['avg_max_CD']:.4f}")

with open(f'./CD_info_{target}_{pad}.txt', 'w') as f:
    f.write(json.dumps(CD_dict,indent="\t"))
print("save complete",target)