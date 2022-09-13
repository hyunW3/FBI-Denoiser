#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.backends.cudnn as cudnn
import torchvision
import cv2 
from matplotlib import pyplot as plt
import os,sys, glob
import numpy as np
import PIL
import random
from PIL import Image
import gc
import h5py
from sklearn.model_selection import train_test_split
from typing import Generator
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"  # Set the GPU 1 to use
from core.data_loader import sem_generator,patch_generator, load_whole_image
from core.crop_image import scale_f_num_0to3,crop_image, make_image_crop
from core.utils import seed_everything


# In[2]:


seed = 0
seed_everything(seed) # Seed 고정


# In[3]:


data_path="./Samsung_SNU_1536x3072"
crop_data_path="./Samsung_SNU_cropped"
os.makedirs(crop_data_path,exist_ok=True)
random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256
num_crop= 500

## function define
def show_patch(crop_im):
    crop_im = crop_im.permute(1,2,0).cpu().numpy()
    crop_im = crop_im.astype('uint8')
    PIL.Image.fromarray(crop_im)
def read_image_and_crop(image : np.array, target : np.array):
    im_t,target_im_t = torch.Tensor(image).cuda(), torch.Tensor(target).cuda()
    top = random.randrange(0,image.shape[0]-crop_size)
    left = random.randrange(0,image.shape[1]-crop_size)
    crop_im = torchvision.transforms.functional.crop(im_t,top,left, crop_size,crop_size)
    crop_target_im = torchvision.transforms.functional.crop(target_im_t,top,left, crop_size,crop_size)
    
    return crop_im,crop_target_im
def return_val_test_index():
    random_seed = random.randint(0,2**32-1)
    seed_everything(random_seed)
    val_image_index = random.randint(0,15)
    test_image_index = random.randint(0,15)
    while val_image_index == test_image_index:
        test_image_index = random.randint(0,15)
    return (val_image_index, test_image_index)
def make_patch_of_image(image_path):
    noisy_patch = torch.Tensor(num_crop,256,256).cuda() 
    target_patch = torch.Tensor(num_crop,256,256).cuda() 
    
    image_num = image_path.split("_")[-1]
    clean_path = os.path.join(data_path,set_num,f"F64_{image_num}")
    #print(image_num, clean_path)
    im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    target_im = cv2.imread(clean_path,cv2.IMREAD_GRAYSCALE)
    im, target_im = im/255. ,target_im / 255.
    #patch_save_path = os.path.join(crop_data_path,set_num,image_name)

    for index in range(num_crop):
        im_cropped, target_im_cropped = read_image_and_crop(im,target_im)
        noisy_patch[index],target_patch[index] = im_cropped,target_im_cropped
    return noisy_patch,target_patch
def append_tensor(original,new):
    if original is None:
        original = new
    else :
        original = torch.cat((original, new),axis=0)
    return original
def save_to_hdf5(noisy_patch, target_patch,set_num :int ,type_name : str):
    if type_name not in ['train','val','test']:
        print("error in type_name")
        sys.exit(-1)
    noisy_patch, target_patch = noisy_patch.cpu().numpy(), target_patch.cpu().numpy()
    with h5py.File(f"{type_name}_Samsung_SNU_patches_{set_num}.hdf5","w") as f:
        f.create_dataset("noisy_images", noisy_patch.shape, dtype='f', data=noisy_patch)
        f.create_dataset("clean_images", target_patch.shape, dtype='f', data=target_patch)
    gc.collect()
# In[4]:

# make patch & split train/val/test set

num_test_image = 3
for set_num in sorted(os.listdir(data_path)):
    if "1" in set_num:
        continue
    if "2" in set_num:
        continue
    set_path = os.path.join(data_path,set_num)
    print(f"====== {set_path} ======")
    image_list = sorted(os.listdir(set_path)) # 16
    image_list = list(filter(lambda x : "F64" not in x,image_list))
    image_list = list(filter(lambda x : "checkpoints" not in x,image_list))
    len_train = len(image_list)-num_test_image*2
    
    train_noisy_patch, train_target_patch = None, None
    val_noisy_patch, val_target_patch = None, None
    test_noisy_patch, test_target_patch = None, None
    gc.collect()
    noisy_f_num = ['F8','F16','F32']
    for i,f_num in enumerate(noisy_f_num): # f_num 마다 구분
        val_index, test_index = return_val_test_index()
        #print("======",i, f_num, set_path)
        #print("======",val_index,test_index)
        for image_idx, image_path in enumerate(glob.glob(f"{set_path}/{f_num}*.png")):
            # print(image_idx,image_path)
            start_idx = i * num_crop
            end_idx = (i+1) * num_crop
            noisy, target = make_patch_of_image(image_path)
            if image_idx == val_index:
                #print(f"This is val image")
                val_noisy_patch = append_tensor(val_noisy_patch,noisy)
                val_target_patch = append_tensor(val_target_patch,target)
            elif image_idx == test_index:
                #print(f"This is test image")
                test_noisy_patch = append_tensor(test_noisy_patch,noisy)
                test_target_patch = append_tensor(test_target_patch,target)
            else :
                train_noisy_patch = append_tensor(train_noisy_patch,noisy)
                train_target_patch = append_tensor(train_target_patch,target)
            noisy, target = None, None
            gc.collect()
        
    print(f"create dataset of {set_num}")
    print(f"[train] noisy & target(clean) patch shape")
    print(f"{train_noisy_patch.shape} and {train_target_patch.shape}")
    print(train_noisy_patch[0].max(), train_target_patch[1].min())
    print(f"[val] noisy & target(clean) patch shape")
    print(f"{val_noisy_patch.shape} and {val_target_patch.shape}")
    print(val_noisy_patch[0].max(), val_target_patch[1].min())
    print(f"[test] noisy & target(clean) patch shape")
    print(f"{test_noisy_patch.shape} and {test_target_patch.shape}")
    print(test_noisy_patch[0].max(), test_target_patch[1].min())
    save_to_hdf5(train_noisy_patch, train_target_patch,set_num,'train')
    save_to_hdf5(val_noisy_patch, val_target_patch,set_num,'val')
    save_to_hdf5(test_noisy_patch, test_target_patch,set_num,'test')
    #x_train, x_test, y_train, y_test = train_test_split(noisy_patch,target_patch,test_size=0.08,shuffle=True)
    #print(x_train.shape, x_test.shape
    del train_noisy_patch, train_target_patch, val_noisy_patch, val_target_patch,test_noisy_patch, test_target_patch
    torch.cuda.empty_cache()
print("complete")





