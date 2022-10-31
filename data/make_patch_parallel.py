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
parser.add_argument('--search-width', type=int, default=30)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1" 
seed = 0
seed_everything(seed) # Seed 고정


# In[3]:


data_path="./Samsung_SNU_1474x3010_aligned"
crop_data_path="./Samsung_SNU_cropped"
os.makedirs(crop_data_path,exist_ok=True)
#random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256
num_crop= 500
num_cores = 16
if args.test is True:
    num_crop = 4
    


## function define

# In[4]:


print_lock = Lock()
process = []
for device_id,set_num in enumerate(sorted(os.listdir(data_path))):
    # singale process version
    #make_dataset_per_set(data_path, set_num ,device_id)
    # make patch & split train/val/test set
    #p = Process(target=make_dataset_per_set, args=(data_path,set_num,device_id,print_lock,num_crop,args.test))
    p = Process(target=make_f_num_dataset_per_set, args=(data_path,set_num,device_id,print_lock,num_crop,args.test))
    p.start()
    process.append(p)
for p in process:
    p.join()
    
print("All Complete")
#print(f"shift info : {len(shift_info['SET1'].keys())}")





