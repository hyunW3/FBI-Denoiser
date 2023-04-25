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
data_path = "../Samsung+SNU+dataset+230216"
num_per_Ffolder = 42
# whole_images = load_whole_image(data_path,num_per_Ffolder)

def generate_top_bottom_crop_image(data_path : str,generator : Generator[str,str,np.array],save=True):
    new_path = data_path+"_1536x3072"
    for img in generator:
        print(img[:3])
        set_num, f_num, image_num = img[:3]
        save_dir = os.path.join(new_path,set_num)
        os.makedirs(save_dir,exist_ok=True)
        #print(save_dir)
    
        img = img[-1][256:-256,:]
        #print(img.shape) # (1536, 3072)
        #plt.imshow(img,cmap='gray')
        file_name = f"{f_num}_{image_num:02d}.png"
        file_path = os.path.join(save_dir,file_name)
        if save is True:
            cv2.imwrite(file_path,img)
    return new_path

print("==== load original image =====")
gen = sem_generator(data_path)
print("==== crop top&bottom of original image =====")
data_path = generate_top_bottom_crop_image(data_path,gen) # "../Samsung+SNU+dataset+230216_1536x3072"


# crop_data_path="Samsung+SNU+dataset+221115_1536x3072_cropped" #"./Samsung_SNU_cropped"
# os.makedirs(crop_data_path,exist_ok=True)
#random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256

print("==== make align information =====")
## make align information
# per set, takes 12 min
#data_path="./Samsung_SNU_1536x3072"
search_range = 40
shift_info = {}
show_log = True
for set_num in sorted(os.listdir(data_path)):
    print(set_num)
    shift_info[f"{set_num}"] = {}

    set_path = os.path.join(data_path,set_num)
    noisy_f_num = ['F08','F16','F32'] #['F1','F2','F4','F8','F16','F32']
    
    for i,f_num in enumerate(noisy_f_num): # f_num 마다 구분
        for image_idx, image_path in tqdm(enumerate(sorted(glob.glob(f"{set_path}/{f_num}*.png")))):
            #print(image_path)
            image_num = image_path.split("_")[-1]
            clean_path = os.path.join(data_path,set_num,f"F64_{image_num}")
            if show_log is True : 
                print(image_path, clean_path)
            
            im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            target_im = cv2.imread(clean_path,cv2.IMREAD_GRAYSCALE)
            
            v_shift, h_shift = get_shift_info(im, target_im,v_width=search_range,h_width=search_range)
            shift_info[set_num][image_path] = {'v_shift' : v_shift, 'h_shift' : h_shift}
            if show_log is True:
                print( v_shift,h_shift)


with open(f' d_F08.txt', 'w') as f:
    f.write(json.dumps(shift_info,indent="\t"))
print(f"write to shift_info_230216.txt complete")

sys.exit(0)
## alignment correction

# data_path="../Samsung+SNU+dataset+230216_1536x3072"
aligned_data_path="../Samsung+SNU+dataset+230216_1536x3072_aligned"
os.makedirs(aligned_data_path,exist_ok=True)
search_range = 40
pad = search_range+1 # 41 -> 82
import json

# shift_info = {}
# with open(f"shift_info_5~10.txt",'r') as f:
#     shift_info = json.load(f)
print(shift_info.keys())
print("==== align image =====")
for set_num in sorted(os.listdir(data_path)):
    print(f"====== {set_num}======")
    keys = list(shift_info.keys())
    print(f"True ? : {keys[1] == set_num}")
    for i in range(4):
        print(f"{set_num[i]} vs {keys[1][i]}",set_num[i] == keys[1][i])
        #print(shift_info[set_num].keys())
    set_path = os.path.join(data_path,set_num)
    aligned_set_path = os.path.join(aligned_data_path,set_num)
    os.makedirs(aligned_set_path,exist_ok=True)
    shift_info_set = shift_info[set_num]
    #print(shift_info[set_path+"/F8_14.png"])
    for file_name in sorted(os.listdir(set_path)):
        if "checkpoints" in file_name :
            continue
        img_path = os.path.join(set_path,file_name)
        save_path = os.path.join(aligned_data_path,set_num,file_name)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        
        print(f"{file_name} save path : {save_path}")
        if "F64" in file_name :
            padded_img = img[pad:-pad,pad:-pad]
            assert padded_img.shape == (1454,2990)
            cv2.imwrite(save_path,padded_img)
            print(file_name, padded_img.shape)
            continue
        file_path = os.path.join(set_path,file_name)
        v_shift, h_shift = shift_info_set[file_path].values()
        file_num = file_name.split("_")[-1].split(".")[0]
        padded_img = img[pad+v_shift:-pad+v_shift:,pad+h_shift:-pad+h_shift]
        assert padded_img.shape == (1454,2990)
        print(file_name,v_shift, h_shift,padded_img.shape)
        #print(shift_info[file_path])
        # plt.imshow(clean_img,cmap='gray')
        cv2.imwrite(save_path,padded_img)
        
        #break
        
    #break
print("complete align")