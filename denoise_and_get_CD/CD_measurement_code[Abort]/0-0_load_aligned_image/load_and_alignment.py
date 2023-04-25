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
from core.utils_data import seed_everything
from core.patch_generate import *
import pandas as pd
import wandb

wandb.init(project="CD_process",name="Samsung_SNU_dataset_1536x3072_fixed",tags=['make alignment image'])


seed_everything(0) # Seed 고정
path_list = ["/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_tmp_dataset/Samsung_SNU/","/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_tmp_dataset/Samsung+SNU+dataset+221115/"]

# load whole image[size : (2048, 3072)] in "whole_images" 
# data_path = "../Samsung_tmp_dataset/Samsung+SNU+dataset+230"
num_per_Ffolder = 42
# whole_images = load_whole_image(data_path,num_per_Ffolder)

search_range = 40
shift_info = {}
shift_info_exist = False
if os.path.isfile('shift_info.json'):
    with open('shift_info.json', 'r') as f:
        shift_info = json.load(f)
    shift_info_exist = True
def align_image(img_info,im,target_im,v_width=40,h_width=40,print_log=False):
    v_shift, h_shift = get_shift_info(im[512:-512,256:-256], target_im[512:-512,256:-256],v_width=v_width,h_width=h_width)
    # v_shift, h_shift = -1, -58
    shift_info[img_info] = {'v_shift' : v_shift, 'h_shift' : h_shift}
    if print_log is True:
        print(img_info,shift_info[img_info])
    return v_shift, h_shift
def generate_top_bottom_crop_image(data_path : str,generator : Generator[str,str,np.array],save=True):
    new_path = "../../denoise_and_get_CD/Samsung_SNU_dataset_1536x3072"
    ref_img_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/Samsung_tmp_dataset/Samsung_SNU/[SET 1]/F64/1_F64.png"
    ref_img = cv2.imread(ref_img_path,cv2.IMREAD_GRAYSCALE)
    for img in generator:
        file_name = img[-1].split("/")[-1]
        img_path = img[-1]
        if file_name[0] == '.' or img_path == ref_img_path:
            continue
        set_num, f_num, image_num = img[:3]
        set_number = int(set_num[3:])
        save_dir = os.path.join(new_path,f"SET{set_number:02d}")
        f_number = int(f_num[1:])
        file_name = f"F{f_number:02d}_{int(image_num):02d}.png"
        file_path = os.path.join(save_dir,file_name)
        # if os.path.isfile(file_path):
        #     continue
        os.makedirs(save_dir,exist_ok=True)
        print(set_num,file_name)
        image_arr = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        key = f"SET{set_number:02d}_F{f_number:02d}_{int(image_num):02d}"

        print("==== make align information =====")
        if shift_info_exist is False:
            v_shift, h_shift = align_image(key,image_arr,ref_img,30,90,print_log=True)
        else :
            print("shift_info exist")
            v_shift, h_shift = shift_info[key]['v_shift'], shift_info[key]['h_shift']
        # print(resultImage.shape) # (1536, 3072)
        print(key,v_shift,h_shift)
        if h_shift > 0:
            ## want to make same size
            padded_img = np.zeros((1536,3072),dtype=np.uint8)
            padded_img[:,h_shift:] = image_arr[256+v_shift:-256+v_shift, :-h_shift]
        else :
            padded_img = np.zeros((1536,3072),dtype=np.uint8)
            # print(padded_img[:,-h_shift:].shape)
            # print(image_arr[256+v_shift:-256+v_shift, -h_shift:].shape)
            padded_img[:,:h_shift] = image_arr[256+v_shift:-256+v_shift, -h_shift:]
            
    
        if save is True:
            cv2.imwrite(file_path,padded_img)
        
        with open('shift_info.json', 'w') as f:
            json.dump(shift_info, f)
    cv2.imwrite(f"{new_path}/SET01/F64_01.png" ,ref_img[256:-256])
for data_path in path_list :
    print(f"==== load original image {data_path}=====")
    gen = sem_generator_path(data_path)
    print("==== crop top&bottom of original image =====")
    data_path = generate_top_bottom_crop_image(data_path,gen) # "../Samsung+SNU+dataset+230216_1536x3072"
print("==== load cropped image =====")


wandb.log({"shift_info" : wandb.Table(dataframe=pd.DataFrame(data=shift_info,index=['v_shift','h_shift']))})

            

