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

# load whole image[size : (2048, 3072)] in "whole_images" 

data_path = "../dataset_1536x3072"
crop_size = 256

print("==== make align information =====")
## make align information
# per set, takes 12 min
#data_path="./Samsung_SNU_1536x3072"
search_range = 40
shift_info = {}
show_log = True

noisy_f_num = ['F01','F02','F04','F08','F16','F32']
clean_f_num = 'F64'
for i,f_num in enumerate(noisy_f_num): # f_num 마다 구분
    for image_idx, image_path in tqdm(enumerate(sorted(glob.glob(f"{data_path}/{f_num}/*.png")))):
        #print(image_path)
        file_name = image_path.split('/')[-1]
        img_info = file_name.split('.')[0].split('_')
        f_num, image_idx = img_info
        clean_path = os.path.join(data_path,f"{clean_f_num}/{clean_f_num}_{image_idx}.png")
        if show_log is True : 
            print(image_path, clean_path)
        
        im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        target_im = cv2.imread(clean_path,cv2.IMREAD_GRAYSCALE)
        print(im.shape, target_im.shape)
        v_shift, h_shift = get_shift_info(im, target_im,v_width=search_range,h_width=search_range)
        shift_info[image_path] = {'v_shift' : v_shift, 'h_shift' : h_shift}
        if show_log is True:
            print( v_shift,h_shift)
save_file_name = "./shift_info.txt"
with open(save_file_name, 'w') as f:
    f.write(json.dumps(shift_info,indent="\t"))
print(f"write to {save_file_name} complete")
