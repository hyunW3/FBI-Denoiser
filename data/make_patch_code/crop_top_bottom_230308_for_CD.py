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
from core.data_loader import sem_generator_path,patch_generator, load_whole_image
from core.crop_image import scale_f_num_0to3,crop_image, make_image_crop
from core.utils import seed_everything
from core.patch_generate import *
import SimpleITK as sitk

seed_everything(0) # Seed 고정
path_list = ["/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_tmp_dataset/Samsung_SNU/","/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_tmp_dataset/Samsung+SNU+dataset+221115/"]

# load whole image[size : (2048, 3072)] in "whole_images" 
# data_path = "../Samsung_tmp_dataset/Samsung+SNU+dataset+230"
num_per_Ffolder = 42
# whole_images = load_whole_image(data_path,num_per_Ffolder)

def generate_top_bottom_crop_image(data_path : str,generator : Generator[str,str,np.array],save=True):
    new_path = "../../denoise_and_get_CD/Samsung_SNU_dataset_1536x3072"
    ref_img_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_tmp_dataset/Samsung_SNU/[SET 1]/F64/1_F64.png"
    for img in generator:
        file_name = img[-1].split("/")[-1]
        img_path = img[-1]
        if file_name[0] == '.':
            continue
        set_num, f_num, image_num = img[:3]
        set_number = int(set_num[3:])
        save_dir = os.path.join(new_path,f"F{set_number:02d}")
        f_number = int(f_num[1:])
        file_name = f"F{f_number:02d}_{int(image_num):02d}.png"
        file_path = os.path.join(save_dir,file_name)
        # if os.path.isfile(file_path):
        #     continue
        os.makedirs(save_dir,exist_ok=True)
        print(set_num,file_name)
        img_info = {'set_num' : set_num,'f_num' : f_num, 'image_idx' : image_idx}
        
        
        # print(resultImage.shape) # (1536, 3072)
        
        #plt.imshow(img,cmap='gray')
        if save is True:
            cv2.imwrite(file_path,resultImage)

    return new_path
for data_path in path_list :
    print(f"==== load original image {data_path}=====")
    gen = sem_generator_path(data_path)
    print("==== crop top&bottom of original image =====")
    data_path = generate_top_bottom_crop_image(data_path,gen) # "../Samsung+SNU+dataset+230216_1536x3072"
print("==== load cropped image =====")

print("==== make align information =====")
## make align information
# per set, takes 12 min
#data_path="./Samsung_SNU_1536x3072"
search_range = 40
shift_info = {}
def align_image(img_info,im,target_im,search_range):
    v_shift, h_shift = get_shift_info(im[512:-512], target_im[512:-512],v_width=search_range,h_width=search_range)
    shift_info[set_num][image_path] = {'v_shift' : v_shift, 'h_shift' : h_shift}

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
            
            if show_log is True:
                print( _shift,h_shift)


search_range = 40
pad = search_range+1 # 41 -> 82
import json

# shift_info = {}
# with open(f"shift_info_5~10.txt",'r') as f:
#     shift_info = json.load(f)
# print(shift_info.keys())
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