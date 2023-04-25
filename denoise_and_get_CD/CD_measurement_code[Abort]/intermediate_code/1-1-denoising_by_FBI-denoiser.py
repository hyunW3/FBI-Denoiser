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



# # Denoising SET1~10 image with FBI-denoiser
# 

# ## Load original img
# with 0~1 scale

# In[3]:

img_dict = {}
debug = False
# data_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_SNU_1474x3010_aligned_ordered/" 
for data_path in glob(".//Samsung_SNU_dataset_1536x3072/"):
    print("=====",data_path, "=====")
    for set_num in sorted(os.listdir(data_path)):
        if set_num[0] == '.' :
            continue
        print("=====",set_num, "=====")
        img_dict[set_num] = {}
        # if debug is True:
        #     print(set_num)
        set_path = os.path.join(data_path,set_num)
        img_list = sorted(os.listdir(set_path))
        img_list = list(filter(lambda x : ".ipynb_checkpoints" not in x,img_list))
        f_num_list = np.array(list(map(lambda x : x.split('_')[0],img_list)))
        f_num_list = np.unique(f_num_list)
        for f_num in f_num_list:
            img_dict[set_num][f_num] = None
        # if debug is True:
        #     print(img_list)
        #     print(f_num_list)


        for img_path in sorted(img_list):
            f_num = img_path.split("_")[0]

            img_path = os.path.join(set_path,img_path)
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
args


# In[7]:


debug= True
for fbi_weight_dir in glob("../weights/230207_FBI_Net*"):
    target_y = fbi_weight_dir.split("_y_as_")[1][:3]
    dataset_version = 'v1' if "with" in fbi_weight_dir else 'v2'
    if debug is True:
        print(fbi_weight_dir)
    print("======",target_y,dataset_version,"======")
    folder_name = f"./denoised_img_{target_y}_{dataset_version}"
    os.makedirs(folder_name,exist_ok=True)
    model = produce_denoised_img_no_crop(_pge_weight_dir=None,_fbi_weight_dir = fbi_weight_dir,_args = args)
    
    # DENOISE IMAGE
    denoised_img_dict = {}
    for set_num in img_dict.keys():
        denoised_img_dict[set_num] = {}
        for f_num in img_dict[set_num].keys():
            denoised_img_dict[set_num][f_num] = None
            for img in img_dict[set_num][f_num]:
                img = np.expand_dims(img, axis=0) 
                print(img.shape)
                if denoised_img_dict[set_num][f_num] is None:
                    denoised_img_dict[set_num][f_num] = model.eval(img)
                else :
                    denoised_img_dict[set_num][f_num] = np.append(denoised_img_dict[set_num][f_num], model.eval(img),axis=0)

            print(f"Denoising SET : {set_num}, f_num : {f_num} End")
    np.save(f"./intermediate_result/{folder_name}_denoised_img_dict.npy",denoised_img_dict)
    # SAVE IMG
    print(denoised_img_dict.keys())
    for set_num in denoised_img_dict.keys():
        print(denoised_img_dict[set_num].keys())
        os.makedirs(f"{folder_name}/{set_num}", exist_ok=True)
        for f_num in denoised_img_dict[set_num].keys():
            # print(denoised_img_dict[set_num][f_num].shape)
            filename = f"{folder_name}/{set_num}/{f_num}.npy"
            np.save(filename, denoised_img_dict[set_num][f_num])
            # for idx,img in enumerate(denoised_img_dict[set_num][f_num]):
            #     filename = f"{folder_name}/{set_num}/{f_num}_{idx:02d}.png"
            #     img = (img*255).astype('uint8')
            #     cv2.imwrite(filename, img[0])
            print(f"Saving SET : {set_num}, f_num : {f_num} End")
    del model,denoised_img_dict
    gc.collect()




