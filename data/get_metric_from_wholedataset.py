import scipy.io as sio
import sys
import PIL,cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import h5py
import argparse
from skimage.metrics import *
from copy import deepcopy
from scipy.ndimage import median_filter

def apply_median_filter(img : np.array ,kernel_size  : tuple = (11,11), repeat: int =3 ,plot : bool =False):
    out = deepcopy(img)
    if plot is True:
        plt.title(f'before iter')
        plt.imshow(out[:200,:200])
        plt.pause(0.01)
        plt.figure(figsize=(20,4))
        plt.plot(out[50][:200])
        plt.pause(0.01)
    for i in range(repeat):
        out = median_filter(out,kernel_size)
        if plot is True:
            plt.title(f'{i+1} iter')
            plt.imshow(out[:200,:200])
            plt.pause(0.01)
            plt.figure(figsize=(20,4))
            plt.plot(out[50][:200])
            plt.pause(0.01)
    return out

f_num_list = ['F01','F02','F04','F08','F16','F32']

argparser = argparse.ArgumentParser()
argparser.add_argument('--median-filter', action='store_true')
argparser.add_argument('--dataset-type', type=str, default='val', choices=['train','val','test'])
args = argparser.parse_args()
denoised_on = args.median_filter
print(f"=== apply median f ilter : {denoised_on} ====")

avg_psnr = {}
avg_ssim = {}
for i in range(1,11):
    for x_f_num in f_num_list:
        avg_psnr[f'SET{i:02d}_{x_f_num}'] = []
        avg_ssim[f'SET{i:02d}_{x_f_num}'] = []
with h5py.File(f"./{args.dataset_type}_Samsung_SNU_patches_whole_set1to10_divided_by_fnum.hdf5",'r') as f :
    clean_images = np.array(f['F64'])
for x_f_num,y_f_num in zip(f_num_list,['F64']*len(f_num_list)): 
    with h5py.File(f"./{args.dataset_type}_Samsung_SNU_patches_whole_set1to10_divided_by_fnum.hdf5",'r') as f :
        noisy_images = np.array(f[x_f_num])
    set_num = 1
    print(x_f_num,y_f_num)
    if x_f_num in ['F01','F02','F04']:
        set_num = 5
        add_index = 4000
    else :
        add_index = 0
    print(f"total images : {noisy_images.shape[0]}")
    for index in range(noisy_images.shape[0]):
    
        if denoised_on is False:
            psnr = peak_signal_noise_ratio(clean_images[index+add_index], noisy_images[index],data_range=1.0)
            ssim = structural_similarity(clean_images[index+add_index], noisy_images[index],data_range=1.0)
        else :
            denoised_img = apply_median_filter(noisy_images[index])
            
            psnr = peak_signal_noise_ratio(clean_images[index+add_index], denoised_img,data_range=1.0)
            ssim = structural_similarity(clean_images[index+add_index], denoised_img,data_range=1.0)
        try : 
            avg_psnr[f'SET{set_num:02d}_{x_f_num}'].append(psnr)
            avg_ssim[f'SET{set_num:02d}_{x_f_num}'].append(ssim)
        except Exception as e :
            print(e,avg_psnr[f'SET{set_num:02d}_{x_f_num}'])
            sys.exit(-1)
            avg_psnr[f'SET{set_num:02d}_{x_f_num}'] = [psnr]
            avg_ssim[f'SET{set_num:02d}_{x_f_num}'] = [ssim]
            

        remainder = index % 1000
            
        if remainder == 999:
            print(f"Total SET{set_num:02d}_{x_f_num} image : {len(avg_psnr[f'SET{set_num:02d}_{x_f_num}'])}")
            print("===PSNR===")
            print(f"{np.min(avg_psnr[f'SET{set_num:02d}_{x_f_num}']):.6f} ~ {np.max(avg_psnr[f'SET{set_num:02d}_{x_f_num}']):.6f}")
            print(f"AVG PSNR  : {np.mean(avg_psnr[f'SET{set_num:02d}_{x_f_num}']):.6f}")
            print("===SSIM===")
            print(f"{np.min(avg_ssim[f'SET{set_num:02d}_{x_f_num}']):.6f} ~ {np.max(avg_ssim[f'SET{set_num:02d}_{x_f_num}']):.6f}")
            print(f"AVG SSIM  : {np.mean(avg_ssim[f'SET{set_num:02d}_{x_f_num}']):.6f}")
            print("")
            avg_psnr[f'SET{set_num:02d}_{x_f_num}'] = np.mean(avg_psnr[f'SET{set_num:02d}_{x_f_num}'])
            avg_ssim[f'SET{set_num:02d}_{x_f_num}'] = np.mean(avg_ssim[f'SET{set_num:02d}_{x_f_num}'])
            
            set_num +=1
            if set_num > 10:
                set_num = 1

import pandas as pd
metric = pd.DataFrame(data = avg_psnr,index = avg_psnr.keys(), columns = ['PSNR',"SSIM"])

metric['PSNR'] = avg_psnr.values()
metric['SSIM'] = avg_ssim.values()
save_metric_name = f'{args.dataset_type}_original_wholeset_metric.csv'
if denoised_on is True:
    save_metric_name = f'{args.dataset_type}_apply_median_filter_wholeset_metric.csv'
metric.to_csv(save_metric_name)
print(f"saving metric to {save_metric_name}")