#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # "0,1,2,3" # "0"


# In[2]:


import numpy as np
import h5py
import cv2
from PIL import Image
import os, sys, gc
import argparse
import json
from glob import glob
from core.get_args import get_args
from core.produce_denoised_img import produce_denoised_img_no_crop
from core.utils import TedataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

import parmap
from functools import partial
from core.watershed import *
from core.watershed import watershed,watershed_per_img,watershed_original,segmentation_with_masking
from core.median_filter import apply_median_filter_cpu, apply_median_filter_gpu, apply_median_filter_gpu_simple

def get_CD(img,img_info, img_hole_info, pad, print_img=False):
    img = np.squeeze(img.copy())
    CD_info = {'min_CD' : [],'max_CD' : [], 'avg_min_CD': None , 'avg_max_CD' : None }

    for i,j in img_hole_info:
        # crop image patch
        pos = {'r_start' : i-pad, 'r_end' : i+pad, 'c_start' : j-pad, 'c_end' : j+pad}
        need_arrange = []
        for key,item in pos.items():
            if item < 0:
                pos[key] = 0
                need_arrange.append(key)
        for key in need_arrange:
            correspond_key = f"{key[0]}_end"
            pos[correspond_key] = pad*2
        img_patch = img[pos['r_start']:pos['r_end'],pos['c_start']:pos['c_end']].copy()
        # print(img_patch.shape)
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
        # find min/max CD for jupyter
        # measured_CD = []
        # for angle in range(180):
        #     measured_CD.append(measure_CD(angle,img_patch,cX,cY,debug=False))
        # print(measured_CD)
        if None in measured_CD:
            print(f"{img_info['set_num']}, {img_info['f_num']} {img_info['idx']} {i},{j} hole cannot get CD measurement due to None")
            continue
        min_CD,max_CD = min(measured_CD),max(measured_CD)
        CD_info['min_CD'].append(min_CD)
        CD_info['max_CD'].append(max_CD)
    
    CD_info['avg_min_CD'] = np.mean(CD_info['min_CD'])
    CD_info['avg_max_CD'] = np.mean(CD_info['max_CD'])
    return CD_info


# In[7]:


def return_img(img_dict): 
    img_info = {}
    for set_num in img_dict.keys():
        for f_num in img_dict[set_num].keys():
            for idx,img in enumerate(img_dict[set_num][f_num]):
                img_info = {'set_num' : set_num, 'f_num' : f_num, 'idx' : idx}
                yield img_info, img
def denoise_and_segment_img_uint8(model, img,print_img=False):
    if img.shape != (1,1,1474,3010):
        while len(img.shape) < 4:
            img = np.expand_dims(img,axis=0)
        if len(img.shape) > 4:
            img = np.squeeze(img)
            img = np.expand_dims(img,axis=0)
            img = np.expand_dims(img,axis=0) # (1,1,1474,3010)
    # print(img.shape)
    if type(model) is type(apply_median_filter_gpu_simple) :
        denoised_img = model(img)
    else :
        denoised_img = model.eval(img)
    denoised_img_uint8 = (denoised_img[0][0]*255).astype('uint8')
    segmentataion_img_uint8 = segmentation_with_masking(denoised_img_uint8)
    if print_img is True:
        plt.imshow(segmentataion_img_uint8)
        plt.pause(0.01)
    segmentataion_img_uint8 = np.expand_dims(segmentataion_img_uint8, axis=0)
    # print(segmentataion_img_uint8.shape)
    return segmentataion_img_uint8 # (1,1474,3010)



def CD_process(model,img,img_info, img_hole_info,pad) -> dict:
    # print(img_info)
    segmentataion_img_uint8 = denoise_and_segment_img_uint8(model,img,print_img=False)
    CD_info = get_CD(segmentataion_img_uint8,img_info, img_hole_info, pad,False)
    print(f"{img_info['set_num']}_{img_info['f_num']}_{img_info['idx']:02d}th image - Min/Max (len : {len(CD_info['min_CD'])}) CD : \
    {CD_info['avg_min_CD']:.4f} ~ {CD_info['avg_max_CD']:.4f}")
    return CD_info



# In[10]:

parser = argparse.ArgumentParser(description='CD_info')
parser.add_argument('--target', default='F16_v2', type=str,
                choices = ['F16_v2','F16_v1','F08_v2'])
args=parser.parse_args()
    
pad = 150
# if args.target[:3] == 'F08':
#     noisy_f_num_list = ['F01', 'F02', 'F04', 'F08']
# else :
#     noisy_f_num_list = ['F01', 'F02', 'F04', 'F08']
noisy_f_num_list = ['F32']
CD_dict ={}
target = 'F16_v2'
with open(f'./hole_info_{target}.txt', 'r') as f:
    hole_info = json.load(f)

# args


img_dict = np.load("./intermediate_result/full_img_dict.npy",allow_pickle=True).item()
# fbi_weight_dir = glob("../weights/230207_FBI_Net*")[0]# '../weights/230207_FBI_Net_Grayscale_Samsung_SET050607080910_mixed_x_as_F#_y_as_F08_MSE_Affine_layers_x17_filters_x64_cropsize_256.w'
# fbi_weight_dir = f'../weights/230207_FBI_Net_Grayscale_Samsung_SET050607080910_mixed_x_as_F#_y_as_{target.split("_")[0]}_MSE_Affine_layers_x17_filters_x64_cropsize_256.w'

# model = produce_denoised_img_no_crop(_pge_weight_dir=None,_fbi_weight_dir = fbi_weight_dir,_args = args)

"""
3. median
"""
# 
# 
success_key = []
collect_metric_list = ["avg_min_CD", "avg_max_CD"]
method_list = [ 'median']

def get_metric(CD_dict,local_CD_info,key, collect_metric_list, method_list, method_idx):
    for metric in collect_metric_list:
        CD_dict[key][f"{method_list[method_idx]}_{metric}"] = local_CD_info[metric]
    method_idx+=1
    return method_idx, CD_dict
for img_info, img in return_img(img_dict):
    if img_info['f_num'] not in noisy_f_num_list :
        continue
    print(img_info, img.shape) # (1,x,y)
    # continue
    set_num, f_num, idx = img_info["set_num"], img_info["f_num"], img_info["idx"]
    key = f'{img_info["set_num"]}_{img_info["f_num"]}_{img_info["idx"]:02d}'
    CD_dict[key] = {}
    method_idx = 0
    
    img_hole_info = hole_info[set_num]['F08'][idx]
    
    print("===== median filter CD measure ===== ")
    CD_info_median = CD_process(apply_median_filter_gpu_simple, img, img_info, img_hole_info, pad)
    method_idx, CD_dict = get_metric(CD_dict, CD_info_median,key, collect_metric_list, method_list, method_idx)    

CD_dict = pd.DataFrame(CD_dict)
method_str = "_".join(method_list)
CD_dict.to_csv(f"CD_dict_{method_str}.csv")




