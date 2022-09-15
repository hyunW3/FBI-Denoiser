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
import gc
import json
from itertools import repeat,product
import argparse
from multiprocessing import Process, Pool, Lock
import multiprocessing
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
from core.data_loader import sem_generator,patch_generator, load_whole_image
from core.crop_image import scale_f_num_0to3,crop_image, make_image_crop
from core.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3" 
seed = 0
seed_everything(seed) # Seed 고정


# In[3]:


data_path="./Samsung_SNU_1536x3072"
crop_data_path="./Samsung_SNU_cropped"
os.makedirs(crop_data_path,exist_ok=True)
#random_crop = torchvision.transforms.RandomCrop(size=256)
crop_size = 256
num_crop= 500
if args.test is True:
    num_crop = 2
shift_info = {}
shift_info_lock = Lock()

from torchvision.transforms import *
transformation = torchvision.transforms.Compose([
    ToTensor(),
    Normalize(0,255)
])
## function define

# make align b/w two img
def get_shift_info(img1, img2, v_width=20,h_width=20, crop_size=256):
    assert(img1.shape == img2.shape)
    top = img1.shape[0]//2
    left = img1.shape[1]//2
    from skimage.metrics import structural_similarity,peak_signal_noise_ratio
    peak_ssim = None
    peak_psnr = None
    for v_shift in range(-v_width,v_width):
        #v_shift *= -1
        for h_shift in range(-h_width,h_width):
            #print(f"===== {v_shift}, {h_shift} ======")
            patch1 = img1[top:top+crop_size,left:left+crop_size]
            patch2 = img2[top+v_shift:top+crop_size+v_shift,left+h_shift:left+crop_size+h_shift]
            ssim, psnr = structural_similarity(patch2,patch1), peak_signal_noise_ratio(patch2,patch1)
            #print(ssim, psnr)
            if peak_ssim is None:
                peak_ssim = {'ssim' : ssim, 'psnr' : psnr, 'v_shift' : v_shift, 'h_shift' : h_shift}
            elif peak_ssim['ssim'] < ssim:
                peak_ssim = {'ssim' : ssim, 'psnr' : psnr, 'v_shift' : v_shift, 'h_shift' : h_shift}

            if peak_psnr is None:
                peak_psnr = {'ssim' : ssim, 'psnr' : psnr, 'v_shift' : v_shift, 'h_shift' : h_shift}
            elif peak_psnr['psnr'] < psnr:
                peak_psnr = {'ssim' : ssim, 'psnr' : psnr, 'v_shift' : v_shift, 'h_shift' : h_shift}
    #print(peak_ssim)
    #print(peak_psnr)
    if peak_ssim['v_shift'] == peak_psnr['v_shift'] and peak_ssim['h_shift'] == peak_psnr['h_shift'] :
        return [peak_psnr['v_shift'],peak_psnr['h_shift']]
    elif (peak_ssim['ssim'] - peak_psnr['ssim']) < (peak_ssim['psnr'] - peak_psnr['psnr']):
        return [peak_psnr['v_shift'],peak_psnr['h_shift']]
    else :
        return [peak_ssim['v_shift'],peak_ssim['h_shift']]
def show_patch(crop_im):
    crop_im = crop_im.permute(1,2,0).cpu().numpy()
    crop_im = crop_im.astype('uint8')
    PIL.Image.fromarray(crop_im)
def read_image_and_crop_unpack(arg):
    return read_image_and_crop(*arg)
def read_image_and_crop(image : np.array, target : np.array,v_shift : int ,h_shift : int,device_id : int,return_list):
    v_pad, h_pad = abs(v_shift), abs(h_shift)
    top = random.randrange(v_pad,image.shape[0]-crop_size-v_pad)
    left = random.randrange(h_pad,image.shape[1]-crop_size-h_pad)
   
    if args.test is True:
        print(top,",",left,",",v_shift,",",h_shift)
    #im_t,target_im_t = transformation(image).cuda(device_id),transformation(target).cuda(device_id)
    im_t,target_im_t = transformation(image),transformation(target)
    crop_im = torchvision.transforms.functional.crop(im_t,top,left, crop_size,crop_size)
    crop_target_im = torchvision.transforms.functional.crop(target_im_t,top+v_shift,left+h_shift, crop_size,crop_size)
    return_list.append([crop_im,crop_target_im])
    if device_id == 0:
        print(return_list)
    #return crop_im,crop_target_im
def make_patch_of_image(image_path,device_id):
    global shift_info
    noisy_patch = torch.Tensor(num_crop,256,256).cuda(device_id) 
    target_patch = torch.Tensor(num_crop,256,256).cuda(device_id) 
    
    image_num = image_path.split("_")[-1]
    clean_path = os.path.join(data_path,set_num,f"F64_{image_num}")
    #print(image_num, clean_path)
    im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    target_im = cv2.imread(clean_path,cv2.IMREAD_GRAYSCALE)
    
    #patch_save_path = os.path.join(crop_data_path,set_num,image_name)
    
    v_shift, h_shift = get_shift_info(im, target_im)
    shift_info_lock.acquire()
    shift_info[f"SET{device_id+1}"][image_path] = {'v_shift' : v_shift, 'h_shift' : h_shift}
    shift_info_lock.release()
    process = []
    manager = multiprocessing.Manager()
    result = manager.list()
    patch_arg = [im,target_im,v_shift,h_shift,device_id,result]
    for i in range(num_crop):
        p = Process(target=read_image_and_crop,args=patch_arg)
        p.start()
        process.append(p)
    for p in process:
        p.join()
    for index, pair in enumerate(result):
        im_cropped,target_im_cropped = pair
        noisy_patch[index],target_patch[index] = im_cropped.cuda(device_id),target_im_cropped.cuda(device_id)
    
    """
    for index in range(num_crop):
        im_cropped, target_im_cropped = read_image_and_crop(image_path,im,target_im,v_shift,h_shift,device_id)
        noisy_patch[index],target_patch[index] = im_cropped,target_im_cropped
    """
    return noisy_patch,target_patch

def return_val_test_index():
    random_seed = random.randint(0,2**32-1)
    seed_everything(random_seed)
    val_image_index = random.randint(0,15)
    test_image_index = random.randint(0,15)
    while val_image_index == test_image_index:
        test_image_index = random.randint(0,15)
    return (val_image_index, test_image_index)

def append_tensor(original,new):
    if original is None:
        original = new
    else :
        original = torch.cat((original, new),axis=0)
    return original
def save_to_hdf5(noisy_patch, target_patch, set_num :int ,type_name : str):
    if type_name not in ['train','val','test']:
        print("error in type_name")
        sys.exit(-1)
    noisy_patch, target_patch = noisy_patch.cpu().numpy(), target_patch.cpu().numpy()
    with h5py.File(f"{type_name}_Samsung_SNU_patches_{set_num}.hdf5","w") as f:
        f.create_dataset("noisy_images", noisy_patch.shape, dtype='f', data=noisy_patch)
        f.create_dataset("clean_images", target_patch.shape, dtype='f', data=target_patch)
    gc.collect()
def make_dataset_per_set(data_path, set_num ,device_id):
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
        #print(set_path)
        for image_idx, image_path in enumerate(glob.glob(f"{set_path}/{f_num}*.png")):
            if args.test is True:
                print(image_idx,image_path)
            noisy, target = make_patch_of_image(image_path,device_id)
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
    print(train_noisy_patch[0].max().item(), train_target_patch[0].min().item())
    print(f"[val] noisy & target(clean) patch shape")
    print(f"{val_noisy_patch.shape} and {val_target_patch.shape}")
    print(val_noisy_patch[0].max().item(), val_target_patch[0].min().item())
    print(f"[test] noisy & target(clean) patch shape")
    print(f"{test_noisy_patch.shape} and {test_target_patch.shape}")
    print(test_noisy_patch[0].max().item(), test_target_patch[0].min().item())
    save_to_hdf5(train_noisy_patch, train_target_patch,set_num,'train')
    save_to_hdf5(val_noisy_patch, val_target_patch,set_num,'val')
    save_to_hdf5(test_noisy_patch, test_target_patch,set_num,'test')
    #x_train, x_test, y_train, y_test = train_test_split(noisy_patch,target_patch,test_size=0.08,shuffle=True)
    #print(x_train.shape, x_test.shape
    del train_noisy_patch, train_target_patch, val_noisy_patch, val_target_patch,test_noisy_patch, test_target_patch
    torch.cuda.empty_cache()
    print(f"complete {set_path}")
# In[4]:

# make patch & split train/val/test set

num_test_image = 3
process = []
for device_id,set_num in enumerate(sorted(os.listdir(data_path))):
    #make_dataset_per_set(data_path, set_num ,device_id)
    shift_info[f"SET{device_id+1}"] = {}
    p = Process(target=make_dataset_per_set, args=(data_path,set_num,device_id))
    p.start()
    process.append(p)
for p in process:
    p.join()
    
print("complete")
with open('shift_info.txt', 'w') as f:
    f.write(json.dumps(shift_info))
print("write to shift_info.txt complete")




