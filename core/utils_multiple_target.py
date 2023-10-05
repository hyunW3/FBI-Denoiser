import sys,os
import random
import time
import datetime
import numpy as np
import scipy.io as sio

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvF
import h5py
import random
import torch
from torch.autograd import Variable
import math
#from skimage import measure # measure.compare_ssim is deprecated after 0.19
import sys
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.metrics import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

class TrdataLoader_multiple_target():

    def __init__(self, _tr_data_dir=None, _args = None):
        # print(sys.path)
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.tr_data_dir = _tr_data_dir
        self.args = _args
        self.data = h5py.File(self.tr_data_dir, "r")
        self.noisy_arr, self.clean_arr = None, None
        assert self.args.y_f_num == 'F#', 'y_f_num should be F#'
        if self.args.y_f_num_type == 'v1':
            self.args.y_f_num_candidate = ['F02','F04','F08','F16','F32','F64']
        else :
            self.args.y_f_num_candidate = ['F02','F04','F08'] + ['F02','F04','F08']
        random.shuffle(self.args.y_f_num_candidate)
            
        set_num_list = ['SET05','SET06','SET07','SET08','SET09','SET10']
        img_len = 3600
        # print('tr_data_laoder',img_len,set_num_list)
        
        for idx,set_num in enumerate(set_num_list):
            y_f_num_current = self.args.y_f_num_candidate[idx]
            set_num_idx = int(set_num[3:])
            if self.args.test_set is not None and set_num_idx in self.args.test_set:
                continue
            if set_num_idx >=5 :
                f_num_list = ['F01','F02','F04','F08','F16','F32','F64']
                self.data = h5py.File("./data/train_Samsung_SNU_patches_SET050607080910_divided_by_fnum_setnum.hdf5", "r")
            else:
                f_num_list = ['F08','F16','F32','F64']
                self.data = h5py.File("./data/train_Samsung_SNU_patches_SET01020304_divided_by_fnum_setnum.hdf5", "r")
            
            mask = [False] * len(self.data[set_num][self.args.x_f_num]) # 7200
            mask[:img_len] = [True] * img_len    
            
            if self.noisy_arr is None:
                self.noisy_arr = self.data[set_num][f'{self.args.x_f_num}'][mask]
                self.clean_arr = self.data[set_num][f'{y_f_num_current}'][mask]
            else:
                self.noisy_arr = np.concatenate((self.noisy_arr,self.data[set_num][f'{self.args.x_f_num}'][mask]),axis=0)
                self.clean_arr = np.concatenate((self.clean_arr,self.data[set_num][f'{y_f_num_current}'][mask]),axis=0)
                
            print(f'===Tr loader {set_num} {self.args.x_f_num} vs {y_f_num_current}===')
            print(f'{self.noisy_arr.shape[0]}/{self.clean_arr.shape[0]}  images are loaded')
            
                    
                    
        
        print("noisy_arr : ",self.noisy_arr.shape, f"pixel value range from {self.noisy_arr[-1].min()} ~ {self.noisy_arr[-1].max()}")
        print("clean_arr : ",self.clean_arr.shape, f"pixel value range from {self.clean_arr[-1].min()} ~ {self.clean_arr[-1].max()}")

        self.num_data = self.clean_arr.shape[0]
        print ('num of training atches : ', self.num_data)
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        
        # random crop
        

        if self.args.noise_type == 'Gaussian' or self.args.noise_type == 'Poisson-Gaussian':

            clean_img = self.clean_arr[index,:,:]
            noisy_img = self.noisy_arr[index,:,:]
            # to check valid array
            #print(self.clean_arr[index,:,:].shape)
            #print(index,np.unique(self.clean_arr[index]),np.unique(clean_img))
            if self.args.data_type == 'Grayscale':
                rand = random.randrange(1,10000)
                clean_patch,noisy_patch = None,None
                if clean_img.shape[0] <= self.args.crop_size and clean_img.shape[1] <= self.args.crop_size:
                    clean_patch = clean_img
                    noisy_patch = noisy_img
                else :
                    clean_patch = image.extract_patches_2d(image = clean_img ,patch_size = (self.args.crop_size, self.args.crop_size), 
                                             max_patches = 1, random_state = rand)
                    noisy_patch = image.extract_patches_2d(image = noisy_img ,patch_size = (self.args.crop_size, self.args.crop_size), 
                                             max_patches = 1, random_state = rand)
                
                    # Random horizontal flipping
                if random.random() > 0.5:
                    clean_patch = np.fliplr(clean_patch)
                    noisy_patch = np.fliplr(noisy_patch)

                # Random vertical flipping
                if random.random() > 0.5:
                    clean_patch = np.flipud(clean_patch)
                    noisy_patch = np.flipud(noisy_patch)
                # need to expand_dims since grayscale has no channel dimension
                clean_patch, noisy_patch = np.expand_dims(clean_patch,axis=0), np.expand_dims(noisy_patch,axis=0)
            else:

                rand_x = random.randrange(0, (clean_img.shape[0] - self.args.crop_size -1) // 2)
                rand_y = random.randrange(0, (clean_img.shape[1] - self.args.crop_size -1) // 2)

                clean_patch = clean_img[rand_x*2 : rand_x*2 + self.args.crop_size, rand_y*2 : rand_y*2 + self.args.crop_size].reshape(1, self.args.crop_size, self.args.crop_size)
                noisy_patch = noisy_img[rand_x*2 : rand_x*2 + self.args.crop_size, rand_y*2 : rand_y*2 + self.args.crop_size].reshape(1, self.args.crop_size, self.args.crop_size)

            

            if self.args.loss_function == 'MSE':
            
                source = torch.from_numpy(noisy_patch.copy())
                target = torch.from_numpy(clean_patch.copy())
            
            elif self.args.loss_function[:10] == 'MSE_Affine' or self.args.loss_function == 'N2V' or self.args.loss_function == 'Noise_est' or self.args.loss_function == 'EMSE_Affine':
                
                source = torch.from_numpy(noisy_patch.copy())
                target = torch.from_numpy(clean_patch.copy())
                
                target = torch.cat([source,target], dim = 0) # (512,256) -> (2,256,256)
                
            return source, target

        else: ## real data

            return source, target
