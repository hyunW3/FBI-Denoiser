#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.backends.cudnn as cudnn
import torchvision
import cv2 
from matplotlib import pyplot as plt
import os,glob,sys
import numpy as np
import PIL
import random
from PIL import Image
import h5py
from typing import Generator
import gc,json
from tqdm import tqdm
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
from core.data_loader import sem_generator,patch_generator, load_whole_image
from core.crop_image import scale_f_num_0to3,crop_image, make_image_crop
from core.utils import seed_everything
from core.patch_generate import *
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--num-crop', type=int, default=1000)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3" 
seed = 0
# seed_everything(seed) # Seed 고정


# In[3]:

#data_path = "Samsung+SNU+dataset+221115_727x1495"
data_path_list = ["Samsung_SNU_1474x3010_aligned_ordered",
        "Samsung+SNU+dataset+221115_1454x2990_aligned_ordered"]


#random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256
num_crop= args.num_crop#500
num_cores = 16
if args.test is True:
    num_crop = 4
 
output_folder = './output_log_make_patch_whole.txt'
f = None
if args.test is False:
    f = open(f"./{output_folder}",'w')
orig_stdout = sys.stdout
orig_stderr = sys.stderr
if args.test is False:
    sys.stderr = f
    sys.stdout = f
val_off = True
patch_for_dataset = make_intial_patch_for_whole_dataset(['F01','F02','F04','F08','F16','F32','F64'],val_off)

for data_path in data_path_list:
    set_list = sorted(os.listdir(data_path))

    for set_num in set_list: # 'SET1'
        if set_num[0] == '.': #hidden file
            continue
        set_number = int(set_num[3:]) # '1'
        print(f"==={set_num}===")
        set_path = os.path.join(data_path,set_num)
        image_list = sorted(glob.glob(set_path+"/F64_*.png")) # select clean image
        
        noisy_f_num = ['F08','F16','F32']
        image_cnt = 16
        val_index = test_index = 0
        if set_number >= 5:
            noisy_f_num = ['F01','F02','F04'] + noisy_f_num
            image_cnt = 2
            val_index, test_index = return_val_test_index(length=image_cnt)
        else :
            val_index, test_index = return_val_test_index(length=image_cnt)
        # print(set_number,val_index,test_index)
        dataset_index = ['train'] * image_cnt
        
        dataset_index[test_index] = 'test'
        if args.test is True:
            print(set_number,dataset_index)
        for image_idx, image_path in enumerate(image_list):
            dataset_type = dataset_index[image_idx]
            # img = cv2.imread(image_path,0)
            patches_dict = make_patch_of_f_num_image(image_path,data_path,set_num,noisy_f_num,num_crop,num_cores,args.test)
            if args.test is True:
                print(image_idx,image_path)
                print(dataset_type)
            patch_for_dataset = append_numpy_from_dict_whole_dataset(dataset_type,set_num,patch_for_dataset,patches_dict)
        # if args.test is True:
        #     break
print(patch_for_dataset.keys())
print(patch_for_dataset['train'].keys())
print('SET01 keys : ',patch_for_dataset['train']['SET01'].keys())
print('SET05 keys : ',patch_for_dataset['train']['SET05'].keys())
# print(patch_for_dataset['train']['SET01']['F64'].shape)
save_whole_dataset_to_hdf5(patch_for_dataset,args.test)
# for dataset_type in patch_for_dataset.keys():
#     with h5py.File(f"{dataset_type}_Samsung_SNU_patches_whole_set10to1_divided_by_fnum_new.hdf5","w") as f:
#         for f_num in patch_for_dataset[dataset_type].keys():
#             print(dataset_type ,f_num)
#             print(patch_for_dataset[dataset_type][f_num].shape)
#             if args.test is False :
#                 f.create_dataset(f_num,patch_for_dataset[dataset_type][f_num].shape, dtype='f', data=patch_for_dataset[dataset_type][f_num])
sys.stdout = orig_stdout
sys.stderr = orig_stderr
if args.test is False:
    f.close()   
print("All Complete")
#print(f"shift info : {len(shift_info['SET1'].keys())}")





