from glob import glob
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import os
import cv2
import numpy as np
import sys
import pandas as pd
import math
search_path = "./denoised_img_submit"
for folder_name in os.listdir(search_path):
    # for img_name in sorted(os.listdir(os.path.join(search_path,folder_name))):
    if glob(search_path + f"/{folder_name}/*.png.png") != []:
        print("ext format is wrong", folder_name,glob(search_path + f"/{folder_name}/*.png.png")[0])
        for img_name in sorted(os.listdir(os.path.join(search_path,folder_name))):
            set_num, f_num, img_num = img_name.split("_")
            img_num = img_num[:-4]
            os.rename(search_path + f"/{folder_name}/{img_name}",search_path + f"/{folder_name}/{set_num}_{f_num}_{img_num}")
        
    if glob(search_path + f"/{folder_name}/SET5_*_01.png") != []:
        print("SET NUM is not well formatted", folder_name,glob(search_path + f"/{folder_name}/SET5_*_01.png")[0])
        for img_name in sorted(os.listdir(os.path.join(search_path,folder_name))):
            try:
                set_num, f_num, img_num = img_name.split("_")
            except Exception as e:
                ValueError("file name is not well formatted",img_name)
            set_number = int(set_num[3:])
            set_num = f"SET{set_number:02d}"
            os.rename(search_path + f"/{folder_name}/{img_name}",search_path + f"/{folder_name}/{set_num}_{f_num}_{img_num}")
    if glob(search_path + f"/{folder_name}/SET10_*_0.png") != []:
        print("it starts with 0", folder_name,glob(search_path + f"/{folder_name}/SET10_*_0.png")[0])
        for img_name in sorted(os.listdir(os.path.join(search_path,folder_name))):
            if img_name[0] == '.':
                continue
            print(img_name)
        
            set_num, f_num, img_num = img_name.split("_")
            img_num, ext = img_num.split(".")
            img_num = int(img_num) + 1
            os.rename(search_path + f"/{folder_name}/{img_name}",search_path + f"/{folder_name}/{set_num}_{f_num}_{img_num:02d}.{ext}")
            
            
metric = {'PSNR' : {},'SSIM' : {}}

def get_PSNR(X, X_hat):

    mse = np.mean((X-X_hat)**2)
    test_PSNR = 10 * math.log10(1/mse)
    
    return test_PSNR

def get_SSIM(X, X_hat):
    
    ch_axis = 0
    #test_SSIM = measure.compare_ssim(np.transpose(X, (1,2,0)), np.transpose(X_hat, (1,2,0)), data_range=X.max() - X.min(), multichannel=multichannel)
    test_SSIM = structural_similarity(X, X_hat, data_range=1.0, channel_axis=ch_axis)
    return test_SSIM

search_path = "./denoised_img_submit"
for folder_name in os.listdir(search_path):
    # if os.path.isdir(os.path.join(search_path,folder_name)) is False:
    #     continue
    if folder_name[:8] != 'denoised' and 'submit' not in folder_name:
        continue
    # if one 'F' in folder_name, then skip
    if folder_name.count('F') == 1: 
        continue
    split_str = folder_name.split("_")
    print(split_str)
    x_f_num, y_f_num,dataset_version = split_str[2], split_str[4],split_str[5]
    
    # tmp
    # if x_f_num != 'F08' or y_f_num != 'F16':
    #     continue
    # if x_f_num != 'F#':
    #     continue
    additional_info = ""
    if split_str[5] != split_str[-1]:
        additional_info = "-".join(split_str[6:])
    print(folder_name)
    print(x_f_num, y_f_num,dataset_version,additional_info)
    if additional_info == 'crushed':
        print("it is crushed, no reference image")
        continue
    key = f"{x_f_num}_{y_f_num}_{dataset_version}_{additional_info}"
    metric['PSNR'][key] = []
    metric['SSIM'][key] = []
    for img_name in sorted(os.listdir(os.path.join(search_path,folder_name))):
        if img_name[0] == '.':
            continue
        try:
            set_num, f_num, img_num = img_name.split("_")
        except Exception as e:
            print(e)
            print(img_name)
            sys.exit(0)
        
        if x_f_num == 'F#' and f_num != 'F01':
            continue
        elif x_f_num != 'F#' and x_f_num != f_num:
            continue
        img_num, ext = img_num.split(".")
        clean_img_path = f"./F64_img/{set_num}_F64_{img_num}.{ext}"
        print(img_name)
        if os.path.exists(clean_img_path) is False:
            raise ValueError("clean img not exist",clean_img_path) 
        clean_img = cv2.imread(clean_img_path,cv2.IMREAD_GRAYSCALE)
        if os.path.exists(os.path.join(search_path,folder_name,img_name)) is False:
            raise ValueError("noisy img not exist".img_name)
        noisy_img = cv2.imread(os.path.join(search_path,folder_name,img_name),cv2.IMREAD_GRAYSCALE)
        noisy_img, clean_img = np.expand_dims(noisy_img[512:768,512:768]/255.,axis=0) , np.expand_dims(clean_img[512:768,512:768]/255.,axis=0)
        psnr, ssim = get_PSNR(clean_img,noisy_img), get_SSIM(clean_img,noisy_img)
        metric['PSNR'][key].append(psnr)
        metric['SSIM'][key].append(ssim)
    # metric['PSNR'][key] = np.mean(metric['PSNR'][key])
    # metric['SSIM'][key] = np.mean(metric['SSIM'][key])

np.save("./metric.npy",metric)
# df = pd.DataFrame(metric)

# df.to_csv("metric.csv")   