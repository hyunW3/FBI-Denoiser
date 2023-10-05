#!/usr/bin/env python
# coding: utf-8

from ast import parse
import torch
import os,glob,sys
import h5py
import argparse
#from multiprocessing import Process, Lock
import torch.multiprocessing as mp
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
from core.utils import seed_everything
from core.patch_generate import make_dataset_iterative

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--crop-size', type=int, default=256)
parser.add_argument('--pad', type=int, default=128)
parser.add_argument('--len-training-patch', type=int, default=21600)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
seed = 0
seed_everything(seed) # Seed 고정

#data_path = "Samsung+SNU+dataset+221115_727x1495"
# data_path = "../dataset"
data_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/training_code/dataset"

img_size = (1710,2990)
len_training_patch = args.len_training_patch#21600

# read image list & check image length
crop_size = 256
image_list = []
image_len = None
for f_num in os.listdir(data_path):
   if f_num.startswith('F'):
      print(f_num)
      image_list.append(sorted(glob.glob(f"{data_path}/{f_num}/*.png")))
for idx in range(len(image_list)-1):
   f_image_list = image_list[idx]
   image_len = len(f_image_list)
   f_image_list_next = image_list[idx+1]
   assert len(f_image_list) == len(f_image_list_next), f"len(f_image_list) != len(f_image_list_next) :\
      {len(f_image_list)} != {len(f_image_list_next)}"
# calculated num_crop 
num_crop = int(len_training_patch/image_len)
print(f"num_crop : {num_crop}, image_len : {image_len}\nexpected training len : {num_crop * image_len}")

   

num_crop= args.num_crop#500

if args.test is True:
   num_crop = 4
    

m = mp.Manager()
write_lock = m.Lock()

make_dataset_iterative(data_path,img_size, write_lock,num_crop=num_crop,crop_size=args.crop_size,
                       clean_f_num = 'F32',
                       is_test=args.test)
   
print("All Complete")
#print(f"shift info : {len(shift_info['SET1'].keys())}")





