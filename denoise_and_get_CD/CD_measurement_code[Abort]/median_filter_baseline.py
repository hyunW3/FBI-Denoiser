import numpy as np
import pandas as pd
import h5py
import cv2
import os, sys
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage import median_filter
from skimage.metrics import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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

samsung_images = {}
with h5py.File("../data/val_Samsung_SNU_patches_SET7_divided_by_fnum.hdf5", 'r') as f:
    for f_num in ['F1', 'F2', 'F4', 'F8', 'F16', 'F32', 'F64']:
        samsung_images[f_num] = np.array(f[f_num])

for key,item in samsung_images.items():
    print(key,item.shape)
denoised_metric = {}
for f_num in ['F1', 'F2', 'F4', 'F8', 'F16', 'F32']:
    avg_psnr = []
    avg_ssim = []
    denoised_metric[f_num] = {'PSNR' : 0, 'SSIM' : 0}
    for noisy, clean in zip(samsung_images[f_num], samsung_images['F64']):
        denoised_img = apply_median_filter(noisy)
        avg_psnr.append(psnr(clean,denoised_img))
        avg_ssim.append(ssim(clean,denoised_img))
    print(f"==== {f_num} denoised result ====")
    print(f"PSNR : {np.mean(avg_psnr):.6f} SSIM : {np.mean(avg_ssim):.6f}")
    denoised_metric[f_num]['PSNR'] = np.mean(avg_psnr)
    denoised_metric[f_num]['SSIM'] = np.mean(avg_ssim)
denoised_metric = pd.DataFrame.from_dict(denoised_metric)
denoised_metric.T.to_csv('median_filter_denoised_metric_SET7_val.csv')
