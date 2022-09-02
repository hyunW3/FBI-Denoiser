#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.backends.cudnn as cudnn
import torchvision
import cv2 
from matplotlib import pyplot as plt
import os
import numpy as np
import PIL
import random
from PIL import Image
import h5py
from typing import Generator
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPU 1 to use
from core.data_loader import sem_generator,patch_generator, load_whole_image
from core.crop_image import scale_f_num_0to3,crop_image, make_image_crop
from core.utils import seed_everything
import h5py
import gc

# In[2]:


seed_everything(0) # Seed 고정


# In[8]:


data_path="./Samsung_SNU_1536x3072"
crop_data_path="./Samsung_SNU_cropped"
os.makedirs(crop_data_path,exist_ok=True)
random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256
num_crop= 500
"""
gen1 = patch_generator(data_path)
for im in gen1:
    print(im)
    break
"""
def show_patch(crop_im):
    crop_im = crop_im.permute(1,2,0).cpu().numpy()
    crop_im = crop_im.astype('uint8')
    PIL.Image.fromarray(crop_im)
def read_image_and_crop(image : np.array, target : np.array):
    im_t,target_im_t = torch.Tensor(image).cuda(), torch.Tensor(target).cuda()
    top = random.randrange(0,image.shape[0]-crop_size)
    left = random.randrange(0,image.shape[1]-crop_size)
    crop_im = torchvision.transforms.functional.crop(im_t,top,left,crop_size,crop_size)
    crop_target_im = torchvision.transforms.functional.crop(target_im_t,top,left,crop_size,crop_size)
    
    return crop_im,crop_target_im


# In[10]:


for set_num in sorted(os.listdir(data_path)):
    print(set_num)
    set_path = os.path.join(data_path,set_num)
    image_list = sorted(os.listdir(set_path)) # 16
    image_list = list(filter(lambda x : "F64" not in x,image_list))
    image_list = list(filter(lambda x : "checkpoints" not in x,image_list))
    noisy_patch = torch.Tensor(num_crop*len(image_list),256,256).cuda() 
    target_patch = torch.Tensor(num_crop*len(image_list),256,256).cuda()
    # print(len(image_list))
    gc.collect()
    for image_idx,image_name in enumerate(image_list):
        image_path = os.path.join(data_path,set_num,image_name)
        image_num = image_name.split("_")[1]
        clean_path = os.path.join(data_path,set_num,f"F64_{image_num}")
        im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        target_im = cv2.imread(clean_path,cv2.IMREAD_GRAYSCALE)
        patch_save_path = os.path.join(crop_data_path,set_num,image_name)
        
        for i in range(num_crop):
            im_cropped,target_im_cropped = read_image_and_crop(im,target_im)
            #print(im_cropped.shape)
            index = i + image_idx*num_crop
            noisy_patch[i],target_patch[i] = im_cropped,target_im_cropped
            #print(im_cropped.shape)
    noisy_patch, target_patch = noisy_patch.cpu().numpy(), target_patch.cpu().numpy()
    print(f"create dataset of {set_num}")
    print(f"noisy & target(clean) patch shape")
    print(f"{noisy_patch.shape} and {target_patch.shape}")
    with h5py.File(f"Samsung_SNU_patches_{set_num}.hdf5","w") as f:
        f.create_dataset("noisy_images", noisy_patch.shape, dtype='f', data=noisy_patch)
        f.create_dataset("clean_images", target_patch.shape, dtype='f', data=target_patch)
