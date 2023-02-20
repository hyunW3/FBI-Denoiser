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
data_path_list = [# "Samsung_SNU_1474x3010_aligned_ordered",
        "Samsung+SNU+dataset+221115_1454x2990_aligned_ordered"]


#random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256
num_crop= args.num_crop#500
num_cores = 32
val_off = True

 
output_folder = './output_log_make_patch_whole.txt'
f = None
if args.test is False:
    f = open(f"./{output_folder}",'w')
orig_stdout = sys.stdout
orig_stderr = sys.stderr
if args.test is False:
    sys.stderr = f
    sys.stdout = f



for data_path in data_path_list:
    set_list = sorted(os.listdir(data_path))
    set_list = list(filter(lambda x : x[0] != '.',set_list))
    ## dataset init
    set_number_list = map(lambda x : x[3:],set_list) # ['01','02','03','04']
    set_number_list_str = "".join(set_number_list)
    save_filename = f"Samsung_SNU_patches_SET{set_number_list_str}_divided_by_fnum_setnum"

    if data_path == "Samsung_SNU_1474x3010_aligned_ordered":
        patch_for_dataset = make_intial_patch_for_whole_dataset(set_num_list=['SET01','SET02','SET03','SET04'],
                                                                f_num_list=['F08','F16','F32','F64'],val_off=val_off)
        num_crop = 480
    else :
        patch_for_dataset = make_intial_patch_for_whole_dataset(set_num_list=['SET05', 'SET06', 'SET07', 'SET08', 'SET09', 'SET10'],
                                                        f_num_list=['F01','F02','F04','F08','F16','F32','F64'],val_off=val_off)
        init_dataset(patch_for_dataset,save_filename)
        num_crop = 4320
    if args.test is True:
        print(data_path)
        if data_path == "Samsung_SNU_1474x3010_aligned_ordered":
            print(patch_for_dataset['train'].keys())
            print(patch_for_dataset['train'][list(patch_for_dataset['train'].keys())[0]])
    if args.test is True:
        num_crop = 4
    total_num_crop = num_crop
    test_num_crop = 480
    
    for set_num in set_list: # 'SET1'
        if set_num[0] == '.': #hidden file
            continue
        
        gc.collect()
        #if data_path == "Samsung+SNU+dataset+221115_1454x2990_aligned_ordered":
            
        set_number = int(set_num[3:]) # '1'
        print(f"==={set_num}=== crop {num_crop} patches")
        set_path = os.path.join(data_path,set_num)
        image_list = sorted(glob.glob(set_path+"/F64_*.png")) # select clean image
        
        noisy_f_num = ['F08','F16','F32']
        image_cnt = 16
        test_index = 0
        if set_number >= 5:
            noisy_f_num = ['F01','F02','F04'] + noisy_f_num
            image_cnt = 2
            _, test_index = return_val_test_index(length=image_cnt)
        else :
            _, test_index = return_val_test_index(length=image_cnt)
        dataset_index = ['train'] * image_cnt
        dataset_index[test_index] = 'test'
        if args.test is True:
            print(set_number,dataset_index)

        split_num = 2
        print(total_num_crop,split_num)
        for idx in range(split_num):
            num_crop = total_num_crop//split_num
            test_num_crop_each = test_num_crop//split_num
            if idx == split_num-1 and total_num_crop%split_num != 0:
                num_crop += total_num_crop%split_num
                test_num_crop_each += test_num_crop%split_num
            print("=== num_crop {}, test_crop ===".format(num_crop,test_num_crop_each))
            patch_for_dataset = make_intial_patch_for_whole_dataset(
                                set_num_list=[set_num],
                                f_num_list=['F01','F02','F04','F08','F16','F32','F64'],
                                val_off=val_off)
            for image_idx, image_path in enumerate(image_list):
                dataset_type = dataset_index[image_idx]
                # img = cv2.imread(image_path,0)
                if dataset_type == 'test':
                    num_crop_each = test_num_crop_each
                elif dataset_type == 'train' :
                    num_crop_each = num_crop
                patches_dict = make_patch_of_f_num_image(image_path,data_path,set_num,noisy_f_num,num_crop_each,num_cores,args.test)
                if args.test is True:
                    print(image_idx,image_path)
                    print(dataset_type)
                patch_for_dataset = append_numpy_from_dict_whole_dataset(dataset_type,set_num,patch_for_dataset,patches_dict)
            print(patch_for_dataset.keys())
            print(patch_for_dataset['train'].keys(),patch_for_dataset['test'].keys())
            print(f'{set_num} keys : {patch_for_dataset["train"][set_num].keys()}')
            print(f'{set_num} F08 keys : {patch_for_dataset["train"][set_num]["F08"].shape}')
            
            print(f"saving file : {set_num}",save_filename)
            save_partial_dataset_to_hdf5(patch_for_dataset,save_filename,test=args.test)
            print(f"saving complete {idx+1}/{split_num}")
            del patch_for_dataset
        # if args.test is True:
        #     break
    
    # print("saving file : ",save_filename)
    # save_whole_dataset_to_hdf5(patch_for_dataset,save_filename,args.test)
    # print("saving complete")
    
    gc.collect()
sys.stdout = orig_stdout
sys.stderr = orig_stderr
if args.test is False:
    f.close()   
print("All Complete")
#print(f"shift info : {len(shift_info['SET1'].keys())}")





