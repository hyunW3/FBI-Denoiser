import bm3d
import os, sys, glob
import cv2
import h5py
import numpy as np
import pandas as pd
import argparse
import PIL
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


parser = argparse.ArgumentParser(description='BM3D Denoising')
parser.add_argument('--sigma', default=40, type=int, help='For BM3D set sigma_psd')
parser.add_argument('--test', action='store_true')
args=parser.parse_args()
sigma_psd = args.sigma

orig_stdout = sys.stdout
orig_stderr = sys.stderr
output_folder = './output_bm3d'
os.makedirs(output_folder,exist_ok=True)
output_filename = f'{output_folder}/output_bm3d_patch_val_sigma{sigma_psd}.txt'

f = None
if args.test is False:
    f = open(output_filename, 'w')
print(f"BM3D denoise whole image log file save in {output_filename}")
if args.test is False:
    sys.stderr = f
    sys.stdout = f
print(f"BM3D denoise patch image")
print(f"sigma_psd : {sigma_psd}")


total_denoised_img = {}
metric_info = {}
for set_file in sorted(glob.glob("test_Samsung_SNU_patches_SET*.hdf5")):
    print(f"==== {set_file} ====")
    set_num = set_file.split('_')[-1].split('.')[0]
    #print(set_num)
    set_prev_psnr = []
    set_prev_ssim = []
    set_psnr = []
    set_ssim = []
    with h5py.File(set_file,'r') as f:
        
        for idx, img in enumerate(zip(f['noisy_images'],f['clean_images'])):
            noisy_img, clean_img = img 
            noisy_img_uint8, clean_img_uint8 = (noisy_img*255).astype('uint8'), (clean_img*255).astype('uint8') 
            # [0,255]:PSNR 15.7177, SSIM : 0.0883
            # [0,1] : PSNR 15.7177, SSIM : 0.1799
            #print(noisy.shape, clean.shape)
            
            ssim = structural_similarity(clean_img,noisy_img)
            psnr = peak_signal_noise_ratio(clean_img,noisy_img)
            set_prev_psnr.append(psnr); set_prev_ssim.append(ssim); 
            print(f"before  denoising PSNR {psnr:.4f}, SSIM : {ssim:.4f}")
            
            denoised_img = bm3d.bm3d(noisy_img_uint8,sigma_psd)
            
            total_denoised_img[f"{set_num}_idx{idx}"] = denoised_img
            denoised_img_uint8 = denoised_img.astype('uint8')
            # not uint8 PSNR 20.0293, SSIM : 0.2434
            # in uint8 PSNR 19.9339, SSIM : 0.2432  
            """
            ssim = structural_similarity(clean_img,denoised_img)
            psnr = peak_signal_noise_ratio(clean_img,denoised_img)
            """
            ssim = structural_similarity(clean_img_uint8,denoised_img_uint8)
            psnr = peak_signal_noise_ratio(clean_img_uint8,denoised_img_uint8)
            set_psnr.append(psnr); set_ssim.append(ssim); 
            print(f"after   denoising PSNR {psnr:.4f}, SSIM : {ssim:.4f}",flush=True)
            
            if args.test is True:
                break
    print(f"before  denoising {set_num} avg PSNR :{np.mean(set_prev_psnr):.4f} , avg SSIM : {np.mean(set_prev_ssim):.4f}")
    print(f"after   denoising {set_num} avg PSNR :{np.mean(set_psnr):.4f} , avg SSIM : {np.mean(set_ssim):.4f}")
    if args.test is True:
        break
    metric_info[f'{set_num}'] = {'before_avg_PSNR' : np.mean(set_prev_psnr), 'before_avg_SSIM' : np.mean(set_prev_ssim),
                        'after_avg_PSNR' : np.mean(set_psnr), 'after_avg_SSIM' : np.mean(set_ssim), }
    np.save(f'{output_folder}/bm_denoised_patch_val_{set_num}_sigma{sigma_psd}.npy',total_denoised_img)
    
    
sys.stdout = orig_stdout
sys.stderr = orig_stderr
f.close()
if args.test is True:
    sys.exit(0)
metric = pd.DataFrame(metric_info)#, columns=['before_avg_PSNR', 'before_avg_SSIM', 'after_avg_PSNR','after_avg_SSIM' ])
metric.to_csv(f'./{output_folder}/metric_patch_val_sigma{sigma_psd}.csv')
