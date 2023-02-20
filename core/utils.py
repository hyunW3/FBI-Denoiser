import sys
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
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import image
import sys
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage import median_filter
from skimage.metrics import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

class TrdataLoader():

    def __init__(self, _tr_data_dir=None, _args = None):

        self.tr_data_dir = _tr_data_dir
        self.args = _args
        self.data = h5py.File(self.tr_data_dir, "r")
        self.noisy_arr, self.clean_arr = None, None
        
        if self.args.integrate_all_set is True:
            if self.args.individual_noisy_input is True:
                print(self.data.keys())
                for set_num in self.data.keys():
                    set_num_idx = int(set_num[3:])
                    if set_num_idx >=5 :
                        f_num_list = ['F01','F02','F04','F08','F16','F32','F64']
                    else:
                        f_num_list = ['F08','F16','F32','F64']
                    # print(f_num_list)
                    # print(self.args.x_f_num, self.args.y_f_num)
                    if self.args.x_f_num not in f_num_list or self.args.y_f_num not in f_num_list:
                        continue
                    
                    if self.noisy_arr is None:
                        self.noisy_arr = self.data[set_num][f'{self.args.x_f_num}']
                        self.clean_arr = self.data[set_num][f'{self.args.y_f_num}']
                    else:
                        self.noisy_arr = np.concatenate((self.noisy_arr,self.data[set_num][f'{self.args.x_f_num}']),axis=0)
                        self.clean_arr = np.concatenate((self.clean_arr,self.data[set_num][f'{self.args.y_f_num}']),axis=0)
                        
                    print(f'===Tr loader {set_num} {self.args.x_f_num} vs {self.args.y_f_num}===')
                    print(f'{self.noisy_arr.shape[0]}/{self.clean_arr.shape[0]}  images are loaded')
            else :
                # noisy_f_num_list = ['F01','F02','F04','F08','F16','F32']
                # adjust the number of images for each set
                mask = None
                set_num_list = ['SET05','SET06','SET07','SET08','SET09','SET10']
                data_dict = h5py.File("./data/train_Samsung_SNU_patches_SET01020304_divided_by_fnum_setnum.hdf5", "r")
                # print(data_dict)
                if self.args.wholedataset_version == 'v1' and self.args.y_f_num in ['F16','F32','F64']:
                    # take 2400 patches for fnum
                    set_num_list = ['SET01','SET02','SET03','SET04'] + set_num_list
                    mask_len_dict = {'F16': 540} # TODO : option for F32,F64 
                     #{'F16' : 540, 'F08' : 1200, 'F04' : 1800, 'F02' : 3600} 
                    mask_len = mask_len_dict[self.args.y_f_num]
                    mask_v1 = [False] * len(data_dict['SET01']['F08']) # 7200
                    image_step = 480
                    num_crop = 144 #mask_len//15
                    for idx in range(0,len(mask_v1),image_step):
                        # print(idx)
                        mask_v1[idx:idx+num_crop] = [True] * num_crop
                    mask_v2 = [False] * len(self.data['SET05']['F01']) # 7200
                    mask_v2[:mask_len] = [True] * mask_len
                elif self.args.wholedataset_version == 'v2':
                    # take 720 patches for fnum 
                    mask_len_dict = {'F16' : 900, 'F08' : 1200, 'F04' : 1800, 'F02' : 3600} 
                    mask_len = mask_len_dict[self.args.y_f_num]
                    mask_v2 = [False] * len(self.data['SET05']['F01'])# 3600
                    mask_v2[:mask_len] = [True] * mask_len
                else :
                    raise ValueError("wholedataset_version is not defined")
                    
                for set_num in set_num_list:
                    set_num_idx = int(set_num[3:])
                    if set_num_idx >=5 :
                        noisy_f_num_list = ['F01','F02','F04','F08','F16','F32']
                        data_dict = self.data
                        mask = mask_v2
                    else: # SET1~4
                        noisy_f_num_list = ['F08','F16','F32']
                        mask = mask_v1
                    for noisy_f_num in noisy_f_num_list:    
                        if noisy_f_num == self.args.y_f_num:
                            break
                        if self.noisy_arr is None:
                            print(data_dict[set_num].keys())
                            print(data_dict[set_num][noisy_f_num].shape)
                            self.noisy_arr = data_dict[set_num][noisy_f_num][mask]
                            self.clean_arr = data_dict[set_num][self.args.y_f_num][mask]
                        else:
                            self.noisy_arr = np.concatenate((self.noisy_arr,data_dict[set_num][noisy_f_num][mask]),axis=0)
                            self.clean_arr = np.concatenate((self.clean_arr,data_dict[set_num][self.args.y_f_num][mask]),axis=0)
                        print('===Tr loader ',set_num, noisy_f_num,'===')
                        print(f'{self.noisy_arr.shape[0]}/{self.clean_arr.shape[0]}  images are loaded')    
                    if noisy_f_num == self.args.y_f_num:
                        continue
                    
                    
        else:
            self.noisy_arr = self.data["noisy_images"]
            self.clean_arr = self.data["clean_images"]
        
        print("noisy_arr : ",self.noisy_arr.shape, f"pixel value range from {self.noisy_arr[-1].min()} ~ {self.noisy_arr[-1].max()}")
        print("clean_arr : ",self.clean_arr.shape, f"pixel value range from {self.clean_arr[-1].min()} ~ {self.clean_arr[-1].max()}")

        self.num_data = self.clean_arr.shape[0]
        print ('num of training patches : ', self.num_data)
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
                
                return source, target
            
            elif self.args.loss_function == 'MSE_Affine' or self.args.loss_function == 'N2V' or self.args.loss_function == 'Noise_est' or self.args.loss_function == 'EMSE_Affine':
                
                source = torch.from_numpy(noisy_patch.copy())
                target = torch.from_numpy(clean_patch.copy())
                
                target = torch.cat([source,target], dim = 0) # (512,256) -> (2,256,256)
                return source, target

        else: ## real data

            return source, target
    
class TedataLoader():

    def __init__(self,_te_data_dir=None, args = None):
        """
        te_data_dir : path to test data or single image
        """

        self.te_data_dir = _te_data_dir
        self.args = args
        if 'SIDD' in self.te_data_dir or 'DND' in self.te_data_dir or 'CF' in self.te_data_dir or 'TP' in self.te_data_dir:
            self.data = sio.loadmat(self.te_data_dir)
        elif self.te_data_dir.endswith('.hdf5'):
            self.data = h5py.File(self.te_data_dir, "r")
        elif self.te_data_dir.endswith('.npy'):
            self.data = np.load(self.te_data_dir, allow_pickle=True)
            self.data = self.data.item()
        elif self.te_data_dir.endswith('.png'):
            self.noisy_arr = np.array(Image.open(self.te_data_dir))
            clean_img_path = self.te_data_dir.split('/')[:-1]
            img_number = self.te_data_dir.split('_')[-1]
            clean_img_path = f"{clean_img_path}/F64_{img_number}"
            self.clean_arr = np.array(Image.open(clean_img_path))
            self.num_data = 1
            print("load single image")
            return
        else :
            raise ValueError('te_data_dir has to be .mat or .h5f5 or .npy')
        self.noisy_arr, self.clean_arr = None, None
        
        if self.args.integrate_all_set is True:
            
            noisy_f_num_list = ['F01','F02','F04','F08','F16','F32']
            if self.args.individual_noisy_input is True:
                #for set_num_idx in range(1,10+1):
                    #set_num = f"SET{set_num_idx:02d}"
                for set_num in self.data.keys():
                    set_num_idx = int(set_num[3:])
                    
                    if set_num_idx >=5 :
                        noisy_f_num_list = ['F01','F02','F04','F08','F16','F32']
                    else:
                        noisy_f_num_list = ['F08','F16','F32']
                    
                    if self.args.test_wholedataset is True:
                        img_len = 100
                        for noisy_f_num in noisy_f_num_list:   
                            if self.noisy_arr is None:
                                # self.noisy_arr = self.data[set_num][f'{self.args.x_f_num}']
                                self.noisy_arr = self.data[set_num][f'{noisy_f_num}'][:img_len]
                                self.clean_arr = self.data[set_num][f'F64'][:img_len]
                            else:
                                # self.noisy_arr = np.concatenate((self.noisy_arr,self.data[set_num][f'{self.args.x_f_num}']),axis=0)
                                self.noisy_arr = np.concatenate((self.noisy_arr,self.data[set_num][f'{noisy_f_num}'][:img_len]),axis=0)
                                self.clean_arr = np.concatenate((self.clean_arr,self.data[set_num][f'F64'][:img_len]),axis=0)
                            
                        print(f'===Te loader {set_num} {noisy_f_num_list}  train on {self.args.x_f_num} vs {self.args.y_f_num}===')
                        
                    print(f'{self.noisy_arr.shape[0]}/{self.clean_arr.shape[0]}  images are loaded')
            elif self.args.test_wholedataset is False: # self.args.individual_noisy_input is False:
                #for set_num_idx in range(1,10+1):
                    #set_num = f"SET{set_num_idx:02d}"
                for set_num in self.data.keys():
                    set_num_idx = int(set_num[3:])
                    if set_num_idx >=5 :
                        noisy_f_num_list = ['F01','F02','F04','F08','F16','F32']
                    else:
                        noisy_f_num_list = ['F08','F16','F32']
                    if self.args.test_wholedataset is True:
                        img_len = 100
                        for noisy_f_num in noisy_f_num_list:    
                            if self.noisy_arr is None:
                                self.noisy_arr = self.data[set_num][noisy_f_num][:img_len]
                                self.clean_arr = self.data[set_num]['F64'][:img_len]
                            else:
                                self.noisy_arr = np.concatenate((self.noisy_arr,self.data[set_num][noisy_f_num][:img_len]),axis=0)
                                self.clean_arr = np.concatenate((self.clean_arr,self.data[set_num]['F64'][:img_len]),axis=0)
                        print(f'===Te loader {set_num} {noisy_f_num_list}===')

                    print(f'{self.noisy_arr.shape[0]}/{self.clean_arr.shape[0]}  images are loaded')
            else : #if self.args.test_wholedataset is True:
                # add different dataset

                set_num_list = ['SET01','SET02','SET03','SET04','SET05','SET06','SET07','SET08','SET09','SET10']
                
                dataset_path = "./data/test_Samsung_SNU_patches_SET01020304_divided_by_fnum_setnum.hdf5" 
                data_dict = h5py.File(dataset_path, "r")
                
                img_len = 100
                print("test wholedataset ",set_num_list)
                for set_num in set_num_list:
                    set_num_idx = int(set_num[3:])
                    noisy_f_num_list = ['F08','F16','F32']   
                    if set_num_idx >= 5 :
                        dataset_path = "./data/test_Samsung_SNU_patches_SET050607080910_divided_by_fnum_setnum.hdf5"
                        data_dict = h5py.File(dataset_path, "r")
                        noisy_f_num_list = ['F01', 'F02', 'F04'] + noisy_f_num_list
                    for f_num in noisy_f_num_list:
                        if self.noisy_arr is None:
                            self.noisy_arr = data_dict[set_num][f_num][:img_len]
                            self.clean_arr = data_dict[set_num]['F64'][:img_len]
                        else:
                            self.noisy_arr = np.concatenate((self.noisy_arr, data_dict[set_num][f_num][:img_len]),axis=0)
                            self.clean_arr = np.concatenate((self.clean_arr, data_dict[set_num]['F64'][:img_len]),axis=0)
                    print(f'===Te loader {set_num} {noisy_f_num_list}===')
                    print(f'{self.noisy_arr.shape[0]}/{self.clean_arr.shape[0]}  images are loaded')
        else:
            self.clean_arr = self.data["clean_images"]
            self.noisy_arr = self.data["noisy_images"]
        self.num_data = self.noisy_arr.shape[0]
        
        print ('num of test images : ', self.num_data)
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""
        
        source = self.noisy_arr[index,:,:]
        target = self.clean_arr[index,:,:]
        
        if 'SIDD' in self.te_data_dir or 'CF' in self.te_data_dir or 'TP' in self.te_data_dir:
            source = source / 255.
            target = target / 255.

        source = torch.from_numpy(source.reshape(1,source.shape[0],source.shape[1])).float().cuda()
        target = torch.from_numpy(target.reshape(1,target.shape[0],target.shape[1])).float().cuda()
        
        if self.args.loss_function == 'MSE_Affine' or self.args.loss_function == 'N2V':
            target = torch.cat([source,target], dim = 0)

        return source, target
    
def get_PSNR(X, X_hat):

    mse = np.mean((X-X_hat)**2)
    test_PSNR = 10 * math.log10(1/mse)
    
    return test_PSNR

def get_SSIM(X, X_hat,data_type):
    
    ch_axis = 0
    #test_SSIM = measure.compare_ssim(np.transpose(X, (1,2,0)), np.transpose(X_hat, (1,2,0)), data_range=X.max() - X.min(), multichannel=multichannel)
    test_SSIM = compare_ssim(X, X_hat, data_range=1.0, channel_axis=ch_axis)
    return test_SSIM



def im2patch(im,pch_size,stride=1):
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')
    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.size()
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = torch.zeros((C, pch_H*pch_W, num_pch)).cuda()
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.view((C, pch_H, pch_W, num_pch))

def chen_estimate(im,pch_size=8):
    """
    Estimated GAT transformed noise to gaussian noise (supposed to be variance 1)
    """
    im=torch.squeeze(im)
    
    #grayscale
    im=im.unsqueeze(0)
    pch=im2patch(im,pch_size,3)
    num_pch=pch.size()[3]
    pch=pch.view((-1,num_pch))
    d=pch.size()[0]
    mu=torch.mean(pch,dim=1,keepdim=True)
    
    X=pch-mu
    sigma_X=torch.matmul(X,torch.t(X))/num_pch
    sig_value,_=torch.symeig(sigma_X,eigenvectors=True)
    sig_value=sig_value.sort().values
    
    
    start=time.time()
    # tensor operation for substituting iterative step.
    # These operation make  parallel computing possiblie which is more efficient

    triangle=torch.ones((d,d))
    triangle= torch.tril(triangle).cuda()
    sig_matrix= torch.matmul( triangle, torch.diag(sig_value)) 
    
    # calculate whole threshold value at a single time
    num_vec= torch.arange(d)+1
    num_vec=num_vec.to(dtype=torch.float32).cuda()
    sum_arr= torch.sum(sig_matrix,dim=1)
    tau_arr=sum_arr/num_vec
    
    tau_mat= torch.matmul(torch.diag(tau_arr),triangle)
    
    # find median value with masking scheme: 
    big_bool= torch.sum(sig_matrix>tau_mat,axis=1)
    small_bool= torch.sum(sig_matrix<tau_mat,axis=1)
    mask=(big_bool==small_bool).to(dtype=torch.float32).cuda()
    tau_chen=torch.max(mask*tau_arr)
      
# Previous implementation       
#    for ii in range(-1, -d-1, -1):
#        tau = torch.mean(sig_value[:ii])
#        if torch.sum(sig_value[:ii]>tau) == torch.sum(sig_value[:ii] < tau):
             #  return torch.sqrt(tau)
#    print('old: ', torch.sqrt(tau))

    return torch.sqrt(tau_chen)

def gat(z,sigma,alpha,g):
    _alpha=torch.ones_like(z)*alpha
    _sigma=torch.ones_like(z)*sigma
    z=z/_alpha
    _sigma=_sigma/_alpha
    f=(2.0)*torch.sqrt(torch.max(z+(3.0/8.0)+_sigma**2,torch.zeros_like(z)))
    return f
def vst(self,transformed,version='MSE'):    
        
        est=chen_estimate(transformed)
        if version=='MSE':
            return ((est-1)**2)
        elif version =='MAE':
            return abs(est-1)
        else :
            raise ValueError("version error in _vst function of train_pge.py")

def inverse_gat(z,sigma1,alpha,g,method='asym'):
   # with torch.no_grad():
    sigma=sigma1/alpha
    if method=='closed_form':
        exact_inverse = ( np.power(z/2.0, 2.0) +
              0.25* np.sqrt(1.5)*np.power(z, -1.0) -
              11.0/8.0 * np.power(z, -2.0) +
              5.0/8.0 * np.sqrt(1.5) * np.power(z, -3.0) -
              1.0/8.0 - sigma**2 )
        exact_inverse=np.maximum(0.0,exact_inverse)
    elif method=='asym':
        exact_inverse=(z/2.0)**2-1.0/8.0-sigma
    else:
        raise NotImplementedError('Only supports the closed-form')
    if alpha !=1:
        exact_inverse*=alpha
    if g!=0:
        exact_inverse+=g
    return exact_inverse

def normalize_after_gat_torch(transformed):
    min_transform=torch.min(transformed)
    max_transform=torch.max(transformed)

    transformed=(transformed-min_transform)/(max_transform-min_transform)
    transformed_sigma= 1/(max_transform-min_transform)
    transformed_sigma=torch.ones_like(transformed)*(transformed_sigma)
    return transformed, transformed_sigma, min_transform, max_transform


