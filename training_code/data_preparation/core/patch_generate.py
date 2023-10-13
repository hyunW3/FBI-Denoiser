from email.mime import image
import torch
import torch.backends.cudnn as cudnn
import torchvision
import cv2 
from matplotlib import pyplot as plt
import os,sys, glob
from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image
import random
import gc
import h5py
from sklearn.model_selection import train_test_split
from typing import Generator
# from torch.multiprocessing import *
import parmap 
from functools import partial
from joblib import Parallel, parallel, delayed
from torch.multiprocessing import Process, Pool, Lock, Manager
    

def get_shift_info(img1, img2, v_width=30,h_width=30, crop_size=256,log=False):
    """
    img1's shift value compared to img2
    img1 : noisy image
    img2 : clean image (reference)
    v_width, h_width : width for search space
    crop_size : default 256
    log : bool, show log if True
    """
    assert(img1.shape == img2.shape)
    top = img1.shape[0]//2
    left = img1.shape[1]//2
    from skimage.metrics import structural_similarity,peak_signal_noise_ratio
    peak_ssim = None
    peak_psnr = None
    if log is True:
        ssim, psnr = structural_similarity(img2,img1), peak_signal_noise_ratio(img2,img1)
        print(f"original ssim (whole) : {ssim}, psnr : {psnr}")
        ssim  = structural_similarity(img2[top:top+crop_size,left:left+crop_size],img1[top:top+crop_size,left:left+crop_size])
        psnr  = peak_signal_noise_ratio(img2[top:top+crop_size,left:left+crop_size],img1[top:top+crop_size,left:left+crop_size])
        print(f"original ssim (patch): {ssim}, psnr : {psnr}")
    for v_shift in range(-v_width,v_width):
        #v_shift *= -1
        for h_shift in range(-h_width,h_width):
            #print(f"===== {v_shift}, {h_shift} ======")
            patch1 = img1[top+v_shift:top+crop_size+v_shift,left+h_shift:left+crop_size+h_shift]
            patch2 = img2[top:top+crop_size,left:left+crop_size]
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
    if log is True:
        print(peak_ssim)
        print(peak_psnr)
    if peak_ssim['v_shift'] == peak_psnr['v_shift'] and peak_ssim['h_shift'] == peak_psnr['h_shift'] :
        return [peak_psnr['v_shift'],peak_psnr['h_shift']]
    elif (peak_ssim['ssim'] - peak_psnr['ssim']) < (peak_psnr['psnr'] - peak_ssim['psnr']):
        return [peak_psnr['v_shift'],peak_psnr['h_shift']]
    else :
        return [peak_ssim['v_shift'],peak_ssim['h_shift']]

def crop_image(data_info, write_lock, num_crop, crop_size=256):
    image_path, key,dataset_type,top_list,left_list = data_info 
    patches_dict = dict()
    patches_dict[key] = np.ones((num_crop,crop_size,crop_size))
    
    im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)   
    try:
        im_t = im/255.
    except Exception as e:
        print(image_path)
        print(e)
        sys.exit(-1)
    for num_iter in tqdm(range(num_crop)):
        top = top_list[num_iter]
        left = left_list[num_iter]
        crop_im = im_t[top:top+crop_size,left:left+crop_size]
        patches_dict[key][num_iter] = crop_im
    write_lock.acquire()
    with h5py.File(f"../{dataset_type}_Samsung_SNU_patches.hdf5",'a') as f:
        
        new_length = f[key].shape[0] + patches_dict[key].shape[0]
        print((new_length,crop_size,crop_size),"<= ",f[key].shape," + ", patches_dict[key].shape)
        f[key].resize((new_length,crop_size,crop_size))
        f[key][-patches_dict[key].shape[0]:] = patches_dict[key]
        print(f"write {key} in {dataset_type}_Samsung_SNU_patches.hdf5")   
        
    write_lock.release()
    
def make_dataset_iterative(data_path,img_size, write_lock,
                           num_crop=500,crop_size = 256,clean_f_num = 'F32',
                           pad = 0, is_test=False):
    num_cores = 16
    if is_test is True :
        num_cores = 2
        print("core to 2 since it is test mode")
    print(f"====== {data_path} ======")
    image_list = sorted(os.listdir(data_path)) # 16
    clean_image_list = sorted(glob.glob(f"{data_path}/{clean_f_num}/*.png"))
    clean_image_list = list(filter(lambda x : "checkpoints" not in x,clean_image_list))
                      
    noisy_f_num = list(filter(lambda x : x not in clean_f_num, image_list))
    noisy_f_num = list(filter(lambda x : "checkpoints" not in x,noisy_f_num))
    print(noisy_f_num)
    for f_num in noisy_f_num:
        assert len(os.listdir(os.path.join(data_path,f_num))) == len(clean_image_list), \
            f"len(noisy {f_num})!= len(clean image {clean_f_num}) : {len(os.listdir(os.path.join(data_path,f_num)))} != {len(clean_image_list)}"
    for dataset_type in ['train','test']:
        with h5py.File(f"../{dataset_type}_Samsung_SNU_patches.hdf5",'w') as f:
            for f_num in noisy_f_num:
                f.create_dataset(f'{f_num}',(0,crop_size,crop_size),maxshape=(None,crop_size,crop_size),dtype='f')
            f.create_dataset(f'{clean_f_num}',(0,crop_size,crop_size),maxshape=(None,crop_size,crop_size),dtype='f')
    
    for img_list in noisy_f_num:
        noisy_path = os.path.join(data_path,img_list)
        noisy_image_list = sorted(glob.glob(noisy_path+"/*.png"))
        noisy_image_list = list(filter(lambda x : "checkpoints" not in x,noisy_image_list))
        image_list += noisy_image_list
        
        
    # f_num_list =  ['F01','F32']
    patch_for_dataset = dict() #make_intial_patch_for_dataset(f_num_list)
    gc.collect()
    print("clean img : ",len(clean_image_list),clean_image_list)
    # val_index, test_index = return_val_test_index(length=16)
    test_index = random.randint(0,len(clean_image_list)-1)
    dataset_index = ['train'] * len(clean_image_list)
    # dataset_index[val_index] = 'val'
    dataset_index[test_index] = 'test'
    data_info = []
    print(dataset_index)
    # for image_idx, image_path in tqdm(enumerate(clean_image_list)):
    for image_idx, image_path in enumerate(clean_image_list):
        dataset_type = dataset_index[image_idx]
        print(image_path,dataset_type)
        image_num = image_path.split('/')[-1].split(f"{clean_f_num}_")[-1].split(".")[0]
        # print(image_path,image_num)
        target_key = f'{clean_f_num}'
        
        top_list = np.random.randint(pad,img_size[0]-pad-crop_size, size=num_crop)
        left_list = np.random.randint(pad,img_size[1]-pad-crop_size, size=num_crop)
        data_info.append([image_path,target_key,dataset_type,top_list,left_list])
        for f_num in noisy_f_num:
            key = f'{f_num}'
            image_path = os.path.join(data_path,f_num,f"{f_num}_{image_num}.png")

            data_info.append([image_path,key,dataset_type,top_list,left_list])
    print("start cropping")
    # Parallel(n_jobs=num_cores, prefer='threads')(delayed(crop_image)\
        # (data_info[i], write_lock, num_crop,crop_size) for i in range(len(data_info)))
    with Pool(num_cores) as p:
        p.starmap(crop_image,[(data_info[i], write_lock, num_crop,crop_size) for i in range(len(data_info))])
        
        
