
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # "0,1,2,3" # "0"


import numpy as np
import h5py
import cv2
import os, sys, gc
import argparse
from glob import glob
from core.get_args import get_args
from core.produce_denoised_img import produce_denoised_img_no_crop
from core.utils import TedataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from core.watershed import watershed,watershed_per_img,watershed_original
from core.median_filter import apply_median_filter_cpu, apply_median_filter_gpu, apply_median_filter_gpu_simple

args = get_args()
args.pge_weight_dir = None #pge_weight_dir
args.loss_function = "MSE_Affine"
args.noise_type = 'Poisson-Gaussian'
args.model_type = 'FBI_Net'
# args.set_num = '1'
args.data_name = 'Samsung'
args.data_type = 'Grayscale'
args.lr = 0.001
args.num_layers = 17
args.num_filters = 64
# args.crop_size = 200
args.debug = False

img_len = 50
if args.test is True:
    img_len = 10
def load_img():
    noisy_arr, clean_arr = None, None
    set_num_list = ['SET01','SET02','SET03','SET04','SET05','SET06','SET07','SET08','SET09','SET10']
                
    dataset_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/test_Samsung_SNU_patches_SET01020304_divided_by_fnum_setnum.hdf5" 
    data_dict = h5py.File(dataset_path, "r")
    
    print("test wholedataset ",set_num_list)
    for set_num in set_num_list:
        set_num_idx = int(set_num[3:])
        noisy_f_num_list = ['F08','F16','F32','F64']   
        if set_num_idx >= 5 :
            dataset_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/test_Samsung_SNU_patches_SET050607080910_divided_by_fnum_setnum.hdf5"
            data_dict = h5py.File(dataset_path, "r")
            noisy_f_num_list = ['F01', 'F02', 'F04'] + noisy_f_num_list
        for f_num in noisy_f_num_list:
            if noisy_arr is None:
                noisy_arr = data_dict[set_num][f_num][:img_len]
                clean_arr = data_dict[set_num]['F64'][:img_len]
            else:
                noisy_arr = np.concatenate((noisy_arr, data_dict[set_num][f_num][:img_len]),axis=0)
                clean_arr = np.concatenate((clean_arr, data_dict[set_num]['F64'][:img_len]),axis=0)
        print(f'===Te loader {set_num} {noisy_f_num_list}===')
        print(f'{noisy_arr.shape[0]}/{clean_arr.shape[0]}  images are loaded')
    return noisy_arr, clean_arr

noisy_arr, clean_arr = load_img()
print("========= noisy,clean shape ==========")
print(noisy_arr.shape, clean_arr.shape)


name_sequence = []
for i in range(1,11):
    set_num = f"SET{i:02d}_"
    if i < 5:
        f_num_list = map(lambda x : f"{set_num}{x}",['F08', 'F16', 'F32','F64'])
    else :
        f_num_list = map(lambda x : f"{set_num}{x}",['F01', 'F02', 'F04', 'F08', 'F16', 'F32','F64'])
    name_sequence += f_num_list
print("========= name_sequence ==========")
print(name_sequence)
def map_image_name_to_index(set_num : str, f_num : str):
    name_to_index = {name : [i*img_len,(i+1)*img_len] for i, name in enumerate(name_sequence)}
    return name_to_index[f"{set_num}_{f_num}"]




import itertools
def get_fbi_weight(f_num_list = ['F#','F01','F02','F04','F08','F16','F32','F64'],filter_keyword = ['median']):
    comb = list(itertools.combinations(f_num_list,2))
    total_fbi_weight_list = []
    for x_f_num,y_f_num in comb:
        fbi_weight_list = glob(f"../weights/230*_FBI_Net*x_as_{x_f_num}*y_as_{y_f_num}*MSE_Affine*")
        for keyword in filter_keyword:
            fbi_weight_list = list(filter(lambda x : keyword not in x,fbi_weight_list))
        total_fbi_weight_list += sorted(list(fbi_weight_list))
        print(x_f_num,y_f_num)
        for fbi_weight in fbi_weight_list:
            print(fbi_weight)
    return total_fbi_weight_list


target_fbi_weight = sorted(glob("../weights/230508*"))
print(target_fbi_weight)


import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
def get_PSNR(X, X_hat):

    mse = np.mean((X-X_hat)**2)
    test_PSNR = 10 * math.log10(1/mse)
    
    return test_PSNR

def get_SSIM(X, X_hat):
    
    ch_axis = 0
    #test_SSIM = measure.compare_ssim(np.transpose(X, (1,2,0)), np.transpose(X_hat, (1,2,0)), data_range=X.max() - X.min(), multichannel=multichannel)
    test_SSIM = structural_similarity(X, X_hat, data_range=1.0, channel_axis=ch_axis)
    return test_SSIM

def parsing_weight_name(fbi_weight_dir):
    info = fbi_weight_dir.split("N2N_")[1]
    info = info.split("_Samsung")[0]
    target_x, target_y = info.split("-")    
        
    return target_x,target_y


# if args.test is True:
#     for fbi_weight_dir in target_fbi_weight:
#         target_x,target_y = parsing_weight_name(fbi_weight_dir)
#         print(target_x,target_y)
#     sys.exit(0)
debug = True
metric = {}
for fbi_weight_dir in target_fbi_weight:
    target_x,target_y = parsing_weight_name(fbi_weight_dir)
        
    if debug is True:
        print(fbi_weight_dir)
            # print(median_filter_input,"median_filter_input")
        
    key = f"230414_{target_x}-{target_y}"
    folder_name = key
    os.makedirs(folder_name,exist_ok=True)
    print(f"========= {key} =========")
    model = produce_denoised_img_no_crop(_pge_weight_dir=None,_fbi_weight_dir = fbi_weight_dir,_args = args)
    
    # DENOISE IMAGE
    denoised_img_dict = {}
    if target_x == 'F#':
        f_num = 'F01'
    else :
        f_num = target_x
    if f_num in ['F08','F16','F32','F64']:
        set_num_list = ['SET01','SET02','SET03','SET04','SET05','SET06','SET07','SET08','SET09','SET10']
    else :
        set_num_list = ['SET05','SET06','SET07','SET08','SET09','SET10']
    for set_num in set_num_list:
        # print(set_num,f_num)
        metric[key] = {'PSNR':[],'SSIM':[],'before_PSNR':[],'before_SSIM':[]}
        index = map_image_name_to_index(set_num,f_num)
        noisy_images = noisy_arr[index[0]:index[1]]
        clean_index = map_image_name_to_index(set_num,'F64')
        clean_images = noisy_arr[clean_index[0]:clean_index[1]]
        print(index, clean_index,clean_images.shape,noisy_images.shape)
        for idx in range(img_len):
            noisy_img,clean_img = noisy_images[idx],clean_images[idx]
            # print(idx,noisy_img.shape, clean_img.shape)
            
            img = np.expand_dims(noisy_img,0)
            img = np.expand_dims(img,0)
            
            denoised_img = model.eval(img)[0][0]
            before_psnr, before_ssim = get_PSNR(clean_img,noisy_img),get_SSIM(clean_img,noisy_img)
            psnr,ssim = get_PSNR(clean_img,denoised_img),get_SSIM(clean_img,denoised_img)
            metric[key]['PSNR'].append(psnr)
            metric[key]['SSIM'].append(ssim)
            metric[key]['before_PSNR'].append(before_psnr)
            metric[key]['before_SSIM'].append(before_ssim)
            filename = f"{set_num}_{f_num}_{idx}_{psnr:.4f}_{ssim:.4f}.png"
            print("1 imwrite : ",f"{folder_name}/{filename}")
            cv2.imwrite(f"{folder_name}/{filename}",denoised_img*255)
            print("2 imwrite : ",f"{folder_name}/{filename}")
            filename = f"{set_num}_{f_num}_{idx}_{before_psnr:.4f}_{before_ssim:.4f}_before.png"
            cv2.imwrite(f"{folder_name}/{filename}",noisy_img*255)
            
        print(f"Denoising SET : {set_num}, f_num : {f_num} End")
            
        gc.collect()
    # np.save(f"./intermediate_result/{folder_name}_denoised_img_dict.npy",denoised_img_dict)
if args.log_off is True:
    sys.exit(0)
print("saving metric")
np.save("./PSNR_and_SSIM_metric_0414trained.npy",metric)
print("saved metric done")