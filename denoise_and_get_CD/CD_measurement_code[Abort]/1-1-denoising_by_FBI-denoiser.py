#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0,1,2,3" # "0"


# In[2]:


import numpy as np
import h5py
import cv2
import os, sys, gc
import argparse
from glob import glob
from core.get_args import get_args
from core.produce_denoised_img import produce_denoised_img_no_crop
from core.utils import TedataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from core.watershed import watershed,watershed_per_img,watershed_original
from core.median_filter import apply_median_filter_cpu, apply_median_filter_gpu, apply_median_filter_gpu_simple



# Denoising SET1~10 image with FBI-denoiser


## Load original img
# with 0~1 scale

# In[3]:

img_dict = {}
debug = False
# data_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_SNU_1474x3010_aligned_ordered/" 
path_list = ["/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_tmp_dataset/Samsung_SNU/","/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_tmp_dataset/Samsung+SNU+dataset+221115/"]
for data_path in path_list:
    print("=====",data_path, "=====")
    for set_num in sorted(os.listdir(data_path)):
        if set_num[0] == '.' :
            continue
        set_path = os.path.join(data_path,set_num)
        set_number = int(set_num.split(" ")[-1][:-1])
        set_num = f"{set_num[1:4]}{set_number:02d}"
        print("=====",set_num, "=====")
        img_dict[set_num] = {}
        # if debug is True:
        #     print(set_num)
        f_num_list = os.listdir(set_path)
        print(f_num_list)
        for f_num in f_num_list:
            f_path = os.path.join(set_path,f_num)
            f_number = int(f_num[1:])
            f_num = f"F{f_number:02d}"
            # print(set_num,f_num)
            img_dict[set_num][f_num] = None
            img_list = sorted(os.listdir(f_path))
            img_list = list(filter(lambda x : ".ipynb_checkpoints" not in x,img_list))
            
            for img_path in sorted(img_list):

                img_path = os.path.join(f_path,img_path)
                if debug is True:
                    print(f_num," & ",img_path)
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) 
                # img_dict[set_num][f_num].append(img)
                img = np.expand_dims(img,axis=0)
                img = np.expand_dims(img,axis=0) / 255.

                if img_dict[set_num][f_num] is None:  
                    img_dict[set_num][f_num] = img
                else :
                    img_dict[set_num][f_num] = np.append(img_dict[set_num][f_num],img,axis=0)
                if debug is True:
                    print(set_num,f_num,img.shape)
if os.path.isfile('filename.txt') is False:
    filename = f"./intermediate_result/full_img_dict.npy"
    np.save(filename,img_dict)
    print(f"saving {filename}")
if debug is True:
    print(img_dict.keys())
    for set_num in img_dict.keys():
        print(img_dict[set_num].keys())
        for f_num in img_dict[set_num].keys():
            print(img_dict[set_num][f_num].shape)

# In[5]:


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



# In[7]:


debug= True
# for fbi_weight_dir in sorted(glob("../weights/230207_FBI_Net*")):
for fbi_weight_dir in sorted(glob("../weights/230303_FBI_Net*")):
    target_y = fbi_weight_dir.split("_y_as_")[1][:3]
    if "x_as_" in fbi_weight_dir:
        target_x = fbi_weight_dir.split("_x_as_")[1][:3]
    else :
        target_x = "F#"
    dataset_version = 'v1' if "with" in fbi_weight_dir else 'v2'
    median_filter_input = True if "median_filter" in fbi_weight_dir else False

    if target_y not in ['F08','F16']:
        continue
    print("======",target_x,"vs",target_y,dataset_version,f" median_filter : {median_filter_input} ======")

    if debug is True:
        print(fbi_weight_dir)
        # print(median_filter_input,"median_filter_input")
    folder_name = f"./denoised_img_{target_x}_vs_{target_y}_{dataset_version}"
    if median_filter_input is True:
        folder_name +="_median_filter_input"
    os.makedirs(folder_name,exist_ok=True)
    median_folder_name = f"{folder_name}_and_median"
    if median_filter_input is False:
        os.makedirs(median_folder_name,exist_ok=True)
    model = produce_denoised_img_no_crop(_pge_weight_dir=None,_fbi_weight_dir = fbi_weight_dir,_args = args)
    
    # DENOISE IMAGE
    denoised_img_dict = {}
    for set_num in img_dict.keys():
        denoised_img_dict[set_num] = {}
        for f_num in img_dict[set_num].keys():
            denoised_img_dict[set_num][f_num] = None
            for idx,img in enumerate(img_dict[set_num][f_num]):
                img = np.expand_dims(img, axis=0) # 1,1536x3074
                print(set_num,f_num,idx,img.shape)
                
                if median_filter_input is True:
                    img = apply_median_filter_gpu_simple(img)
                denoised_img = model.eval(img)[0]
                # print("after denoised",denoised_img.shape)
                if denoised_img_dict[set_num][f_num] is None:
                    denoised_img_dict[set_num][f_num] = denoised_img
                else :
                    denoised_img_dict[set_num][f_num] = np.append(denoised_img_dict[set_num][f_num], denoised_img,axis=0)
                filename = f"{set_num}_{f_num}_{idx}.png"
                if os.path.isfile(f"{folder_name}/{filename}") is False:
                    print("imwrite : ",f"{folder_name}/{filename}")
                    cv2.imwrite(f"{folder_name}/{filename}",denoised_img[0]*255)
                # apply median filter after denoising
                if median_filter_input is False:
                    print("median imwrite : ",f"{median_folder_name}/{filename}")
                    median_denoised_img = apply_median_filter_gpu_simple(denoised_img)[0][0]*255
                    cv2.imwrite(f"{median_folder_name}/{filename}",median_denoised_img)
            print(f"Denoising SET : {set_num}, f_num : {f_num} End")
            
    np.save(f"./intermediate_result/{folder_name}_denoised_img_dict.npy",denoised_img_dict)

    gc.collect()




