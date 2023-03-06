#!/usr/bin/env python
# coding: utf-8
import os 
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
from core.CD_measure import *



# In[10]:

parser = argparse.ArgumentParser(description='CD_info')
parser.add_argument('--target', default='F16_v2', type=str,
                choices = ['F16_v2','F16_v1','F08_v2'])
parser.add_argument('--weight', default=None, type=str, help='For samsung SEM image, set denoised fbi-net weight')
parser.add_argument('--gpu', default='0', type=str, help='set GPU')
args=parser.parse_args()

if args.weight is None:
    raise ValueError("weight is not set")
fbi_weight_dir = args.weight # 
target = args.target #'F16_v2'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

pad = 150
if args.target[:3] == 'F08':
    noisy_f_num_list = ['F01', 'F02', 'F04']
elif args.target[:3] == 'F16' :
    noisy_f_num_list = ['F01', 'F02', 'F04', 'F08']
else :
    raise ValueError("invalid args.target",args.target)
CD_dict ={}

with open(f'./result_data/hole_info_F16_v2.txt', 'r') as f:
    hole_info = json.load(f)


# In[11]:


## Denoising argument
args = get_args()
args.pge_weight_dir = None #pge_weight_dir
args.loss_function = "MSE_Affine"
args.noise_type = 'Poisson-Gaussian'
args.model_type = 'FBI_Net'
# args.set_num = '1'
args.data_name = 'Samsung'
args.data_type = 'Grayscale'
args.lr = 0.001
args.num_layers = 17
args.num_filters = 64
# args.crop_size = 200
args.debug = False
# args


img_dict = np.load("./intermediate_result/full_img_dict.npy",allow_pickle=True).item()


model = produce_denoised_img_no_crop(_pge_weight_dir=None,_fbi_weight_dir = fbi_weight_dir,_args = args)

"""
1. Denoise with FBI-denoiser
2. Denoise with FBI-denoiser + median
3. median + Denoise with FBI-denoiser
4. median # done once enough
"""
# 
# 
success_key = []
collect_metric_list = ["avg_min_CD", "avg_max_CD"]

method_list = [target,f"median_{target}",f"{target}_median"]

if target == 'F16_v2' or os.path.isfile("./result_data/CD_dict_F16_v2&F16_v2&median_median.csv") is False:
    method_list.append("median")

method_str = "-".join(method_list)
save_weight_name = fbi_weight_dir.split("layers")[0].split("Samsung_")[1]
print(f"save file name : CD_dict_{method_str}_{save_weight_name}")

def get_metric(CD_dict,local_CD_info,key, collect_metric_list, method):
    for metric in collect_metric_list:
        CD_dict[key][f"{method}_{metric}"] = local_CD_info[metric]
    return CD_dict

def denoise_and_median(img):
    denoised_img = model.eval(img)
    denoised_img = apply_median_filter_gpu_simple(denoised_img)
    return denoised_img 

def median_and_denoise(img):
    denoised_img = apply_median_filter_gpu_simple(img)
    denoised_img = model.eval(denoised_img)
    return denoised_img 

print("method list :", method_list)

for img_info, img in return_img(img_dict):
    if img_info['f_num'] not in noisy_f_num_list :
        continue
    print(img_info, img.shape) # (1,x,y)
    # continue
    set_num, f_num, idx = img_info["set_num"], img_info["f_num"], img_info["idx"]
    key = f'{img_info["set_num"]}_{img_info["f_num"]}_{img_info["idx"]:02d}'
    CD_dict[key] = {}
    
    img_hole_info = hole_info[set_num][f_num][idx]
    for method in method_list:
        img_info['method'] = method
        target_model = model
        if f"{target}_median" == method:
            target_model = denoise_and_median
        elif f"median_{target}":
            target_model = median_and_denoise
        
        print(f"===== {method}  CD measure =====")
        if method != 'median':
            CD_info_ours = CD_process(target_model, img, img_info, img_hole_info, pad)
            CD_dict = get_metric(CD_dict, CD_info_ours,key, collect_metric_list, method)
        else :
            target_model = apply_median_filter_gpu_simple
            img = img_dict[set_num]['F32'][idx]
            img_info['f_num'] = 'F32'
            CD_info_ours = CD_process(target_model, img, img_info, img_hole_info, pad)
            CD_dict = get_metric(CD_dict, CD_info_ours,key, collect_metric_list, method)
            

    print("")


CD_dict = pd.DataFrame(CD_dict)
CD_dict.to_csv(f"./result_data/CD_dict_{method_str}_{save_weight_name}.csv")





