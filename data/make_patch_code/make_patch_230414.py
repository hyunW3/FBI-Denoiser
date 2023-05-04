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
#from multiprocessing import Process, Lock
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pool, Lock, Manager
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
from core.data_loader import sem_generator,patch_generator, load_whole_image
from core.crop_image import scale_f_num_0to3,crop_image, make_image_crop
from core.utils import seed_everything
from core.patch_generate import *
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--num-crop', type=int, default=530)
parser.add_argument('--crop-size', type=int, default=256)
parser.add_argument('--pad', type=int, default=128)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
seed = 0
seed_everything(seed) # Seed 고정


# In[3]:

#data_path = "Samsung+SNU+dataset+221115_727x1495"
data_path = "/home/hyunwoong/FBI-Denoiser/data/Samsung_SNU_dataset_230414_1710x2990_aligned"
img_size = (1710,2990)


#random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256
num_crop= args.num_crop#500

if args.test is True:
   num_crop = 4
    


## function define

# In[4]:


write_lock = Lock()
device_id =0 
# make_dataset(data_path,device_id,print_lock,num_crop,args.test)
make_dataset_iterative(data_path,img_size, write_lock,num_crop=num_crop,crop_size=args.crop_size, is_test=args.test)
# process = []
# for device_id,set_num in enumerate(sorted(os.listdir(data_path))):
#     # singale process version
#     #make_dataset_per_set(data_path, set_num ,device_id)
#     # make patch & split train/val/test set
#     #p = Process(target=make_dataset_per_set, args=(data_path,set_num,device_id%4,print_lock,num_crop,args.test))
#     p = Process(target=make_dataset, args=(data_path,set_num,device_id%4,print_lock,num_crop,args.test))
#     p.start()
#     process.append(p)
#     if args.test is True:
#         break
# for p in process:
#     p.join()
    
print("All Complete")
#print(f"shift info : {len(shift_info['SET1'].keys())}")





