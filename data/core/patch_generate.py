from re import L
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
from core.utils import seed_everything,poolcontext
# from torch.multiprocessing import *
import parmap 
from functools import partial
from joblib import Parallel, parallel, delayed

    
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
    with h5py.File(f"{dataset_type}_Samsung_SNU_patches_230414.hdf5",'a') as f:
        f.create_dataset(key,patches_dict[key].shape, dtype='f', data=patches_dict[key])
    write_lock.release()
    del patches_dict
    gc.collect()
    print(f"write {key} in {dataset_type}_Samsung_SNU_patches_230414.hdf5 ")
def make_dataset_iterative(data_path,img_size, write_lock,num_crop=500,crop_size = 256,pad = 0, is_test=False):
    num_test_image = 3
    num_cores = 14
    if is_test is True :
        num_cores = 2
        print("core to 2 since it is test mode")
    for dataset_type in ['train','test']:
        with h5py.File(f"{dataset_type}_Samsung_SNU_patches_230414.hdf5",'w') as f:
            pass
    print(f"====== {data_path} ======")
    image_list = sorted(os.listdir(data_path)) # 16
    image_list = list(filter(lambda x : "F32" not in x,image_list))
    image_list = list(filter(lambda x : "checkpoints" not in x,image_list))
    # f_num_list =  ['F01','F32']
    patch_for_dataset = dict() #make_intial_patch_for_dataset(f_num_list)
    gc.collect()
    # val_index, test_index = return_val_test_index(length=16)
    val_index = test_index = random.randint(0,42)
    dataset_index = ['train'] * 42
    # dataset_index[val_index] = 'val'
    dataset_index[test_index] = 'test'
    data_info = []
    for image_idx, image_path in tqdm(enumerate(sorted(glob.glob(f"{data_path}/F32*.png")))):
        dataset_type = dataset_index[image_idx]
        image_num = image_path.split("F32_")[-1].split("-")[0]
        print(image_path,image_num)
        target_key = f'F32_{image_num}'
        
        top_list = np.random.randint(pad,img_size[0]-pad-crop_size, size=num_crop)
        left_list = np.random.randint(0,img_size[1]-crop_size, size=num_crop)
        data_info.append([image_path,target_key,dataset_type,top_list,left_list])
        
        for image_idx in range(1,16+1):
            key = f'F01_{image_num}-{image_idx:02d}'
            image_path = os.path.join(data_path,f"F01_{image_num}-{image_idx:02d}.png")

            data_info.append([image_path,key,dataset_type,top_list,left_list])
    print("start cropping")

    Parallel(n_jobs=num_cores, prefer='threads')(delayed(crop_image)(data_info[i], write_lock, num_crop,crop_size) for i in range(len(data_info)))
    #x_train, x_test, y_train, y_test = train_test_split(noisy_patch,target_patch,test_size=0.08,shuffle=True)
    #print(x_train.shape, x_test.shape)
    torch.cuda.empty_cache()
    