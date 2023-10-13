import torch
import torch.backends.cudnn as cudnn
import torchvision
import cv2 
from matplotlib import pyplot as plt
import os,glob,sys
from tqdm import tqdm
import numpy as np
import PIL
import random
from PIL import Image
import h5py
from typing import Generator
import gc,json 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 1 to use
from core.utils import seed_everything
from core.patch_generate import *

seed_everything(0) # Seed 고정

data_path = "../dataset_1536x3072"
crop_size = 256

show_log = True
clean_f_num = 'F64'
f_num_list = ['F01','F02','F04','F08','F16','F32']


## alignment correction

aligned_data_path=f"{data_path}_aligned"
os.makedirs(aligned_data_path,exist_ok=True)
for f_num in f_num_list:
    os.makedirs(os.path.join(aligned_data_path,f_num),exist_ok=True)
search_range = 40
pad = search_range+1 # 41 -> 82
import json

shift_info = {}
save_file_name = "./shift_info.txt"
with open(save_file_name,'r') as f:
    shift_info = json.load(f)
print(shift_info.keys())
print("==== align image =====")

    #print(shift_info[set_path+"/F8_14.png"])
# for file_name in sorted(os.listdir(data_path)):
for img_path, value in shift_info.items():
    if "checkpoints" in img_path :
        continue
    file_name = img_path.split('/')[-1]
    f_num = img_path.split('/')[-2]
    save_path = os.path.join(aligned_data_path,f_num,file_name)
    print(img_path)
    if os.path.exists(img_path) is False:
        raise ValueError(f"{img_path} is not exist")
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    
    print(f"{file_name} save path : {save_path}")
    # if clean_f_num in file_name :
    #     padded_img = img[pad:-pad,pad:-pad]
    #     assert padded_img.shape == (1454,2990), f"{padded_img.shape}"
    #     cv2.imwrite(save_path,padded_img)
    #     print(file_name, padded_img.shape)
    # else :
    v_shift, h_shift = shift_info[img_path].values()
    padded_img = img[pad+v_shift:-pad+v_shift:,pad+h_shift:-pad+h_shift]
    
    assert padded_img.shape == (1454,2990), f"{padded_img.shape}"
    print(file_name,v_shift, h_shift,padded_img.shape)
    
    cv2.imwrite(save_path,padded_img)
import shutil
shutil.copytree(f"{data_path}/F64",f"{aligned_data_path}/F64")
print("complete align")