import bm3d
import os, sys, glob
import cv2
import numpy as np
import argparse
import PIL
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from core.data_loader import load_whole_image_in_folder

parser = argparse.ArgumentParser(description='BM3D Denoising')
parser.add_argument('--sigma', default=40, type=int, help='For BM3D set sigma_psd')
parser.add_argument('--test', action='store_true')
args=parser.parse_args()
data_path = "Samsung_SNU_1474x3010_aligned"
whole_images = load_whole_image_in_folder(data_path)

sigma_psd = args.sigma

orig_stdout = sys.stdout
orig_stderr = sys.stderr
output_folder = './output_bm3d'
os.makedirs(output_folder,exist_ok=True)
output_filename = f'{output_folder}/output_bm3d_sigma{sigma_psd}.txt'
f = None
if args.test is False:
    f = open(output_filename, 'w')
print(f"BM3D denoise whole image log file save in {output_filename}")
if args.test is False:
    sys.stderr = f
    sys.stdout = f
print(f"BM3D denoise whole image")
print(f"data path : {data_path}, sigma_psd : {sigma_psd}")

print(whole_images.keys(), whole_images['SET1'].keys())
print("======================")


total_denoised_img = {}

for set_num in whole_images.keys():
    print(set_num)
    set_prev_psnr = []
    set_prev_ssim = []
    set_psnr = []
    set_ssim = []
    for img_name, noisy_img in whole_images[set_num].items():
        print(f"===== {img_name} ======")
        if 'F64' in img_name:
            continue
        clean_img_name = "F64_"+img_name.split('_')[-1]
        print(clean_img_name)
        clean_img = whole_images[set_num][clean_img_name]
        print(noisy_img.shape, clean_img.shape)
        ssim = structural_similarity(clean_img,noisy_img)
        psnr = peak_signal_noise_ratio(clean_img,noisy_img)
        set_prev_psnr.append(psnr); set_prev_ssim.append(ssim); 
        print(f"before  denoising PSNR {psnr:.4f}, SSIM : {ssim:.4f}")
        
        denoised_img = bm3d.bm3d(noisy_img,sigma_psd)
        total_denoised_img[f"{set_num}_{img_name}"] = denoised_img
        denoised_img_uint8 = denoised_img.astype('uint8')
        ssim = structural_similarity(clean_img,denoised_img_uint8)
        psnr = peak_signal_noise_ratio(clean_img,denoised_img_uint8)
        set_psnr.append(psnr); set_ssim.append(ssim); 
        print(f"after   denoising PSNR {psnr:.4f}, SSIM : {ssim:.4f}",flush=True)
        if args.test is True:
            break
    print(f"before  denoising {set_num} avg PSNR :{np.mean(set_prev_psnr)} , avg SSIM : {np.mean(set_prev_ssim)}")
    print(f"after   denoising {set_num} avg PSNR :{np.mean(set_psnr)} , avg SSIM : {np.mean(set_ssim)}")
    if args.test is True:
        break
np.save(f'{output_folder}/bm_denoised_img_{sigma_psd}.npy',total_denoised_img)


sys.stdout = orig_stdout
sys.stderr = orig_stderr
f.close()