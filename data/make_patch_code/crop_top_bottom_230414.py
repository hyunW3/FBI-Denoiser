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
from core.data_loader import sem_generator,patch_generator, load_whole_image
from core.crop_image import scale_f_num_0to3,crop_image, make_image_crop
from core.utils import seed_everything
from core.patch_generate import *

seed_everything(0) # Seed 고정

# load whole image[size : (2048, 3072)] in "whole_images" 
data_path = "../Samsung_SNU_dataset_230414"

def sem_generator(data_path : str):
    for f_num in sorted(os.listdir(data_path)):
        f_path = os.path.join(data_path,f_num)
        f_number = int(f_num[1:])
        for file_name in sorted(os.listdir(f_path)):
            file_path = os.path.join(f_path,file_name)
            # print(f_num)
            img_info = file_name.split(f"_{f_num}")[0]
            # print(img_info)
            if f_number == 1:
                image_num,image_idx = img_info.split("-")[0], img_info.split("-")[1] # 1-2
            else :
                image_num = img_info
                image_idx = 0
            img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
            
            yield [f"F{f_number:02d}", int(image_num), int(image_idx),img]

def generate_top_bottom_crop_image(data_path : str,generator : Generator[str,str,np.array],save=True):
    save_dir = data_path+"_1792x3072"
    os.makedirs(save_dir,exist_ok=True)
    for img in generator:
        # print(img[:3])
        f_num, image_num, image_idx = img[:3]
    
        img = img[-1][128:-128,:]
        #print(img.shape) # (1792, 3072)
        #plt.imshow(img,cmap='gray')
        file_name = f"{f_num}_{image_num:02d}-{image_idx:02d}.png"
        file_path = os.path.join(save_dir,file_name)
        # print(file_name)
        if save is True:
            cv2.imwrite(file_path,img)
        
    return save_dir

print("==== load original image =====")
gen = sem_generator(data_path)
print("==== crop top&bottom of original image =====")
data_path = generate_top_bottom_crop_image(data_path,gen) # "../Samsung+SNU+dataset+230216_1792x3072"


# crop_data_path="Samsung+SNU+dataset+221115_1792x3072_cropped" #"./Samsung_SNU_cropped"
# os.makedirs(crop_data_path,exist_ok=True)
#random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256

print("==== make align information =====")
## make align information
# per set, takes 12 min
#data_path="./Samsung_SNU_1792x3072"
search_range = 40
shift_info = {}
show_log = True

noisy_f_num = ['F01'] #['F1','F2','F4','F8','F16','F32']
for i,f_num in enumerate(noisy_f_num): # f_num 마다 구분
    for image_idx, image_path in tqdm(enumerate(sorted(glob.glob(f"{data_path}/{f_num}*.png")))):
        #print(image_path)
        img_info = image_path.split("_")[-1]
        image_num,image_idx = img_info.split("-")[0], img_info.split("-")[1] # 1-2
        clean_path = os.path.join(data_path,f"F32_{image_num}-00.png")
        if show_log is True : 
            print(image_path, clean_path)
        
        im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        target_im = cv2.imread(clean_path,cv2.IMREAD_GRAYSCALE)
        
        v_shift, h_shift = get_shift_info(im, target_im,v_width=search_range,h_width=search_range)
        shift_info[image_path] = {'v_shift' : v_shift, 'h_shift' : h_shift}
        if show_log is True:
            print( v_shift,h_shift)
save_file_name = "./shift_info_230414.txt"
with open(save_file_name, 'w') as f:
    f.write(json.dumps(shift_info,indent="\t"))
print(f"write to {save_file_name} complete")

## alignment correction

# data_path="../Samsung+SNU+dataset+230216_1792x3072"
aligned_data_path=f"{data_path}_aligned"
os.makedirs(aligned_data_path,exist_ok=True)
search_range = 40
pad = search_range+1 # 41 -> 82
import json

# shift_info = {}
# with open(f"shift_info_5~10.txt",'r') as f:
#     shift_info = json.load(f)
print(shift_info.keys())
print("==== align image =====")

    #print(shift_info[set_path+"/F8_14.png"])
for file_name in sorted(os.listdir(data_path)):
    if "checkpoints" in file_name :
        continue
    img_path = os.path.join(data_path,file_name)
    save_path = os.path.join(aligned_data_path,file_name)
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    
    print(f"{file_name} save path : {save_path}")
    if "F32" in file_name :
        padded_img = img[pad:-pad,pad:-pad]
        assert padded_img.shape == (1710,2990)
        cv2.imwrite(save_path,padded_img)
        print(file_name, padded_img.shape)
        continue
    
    v_shift, h_shift = shift_info[img_path].values()
    padded_img = img[pad+v_shift:-pad+v_shift:,pad+h_shift:-pad+h_shift]
    assert padded_img.shape == (1710,2990)
    print(file_name,v_shift, h_shift,padded_img.shape)
    
    cv2.imwrite(save_path,padded_img)
        
print("complete align")