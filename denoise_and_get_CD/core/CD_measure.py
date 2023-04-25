import numpy as np
import h5py
import cv2
from PIL import Image
import os, sys, gc
import argparse
import json
from glob import glob
from .get_args import get_args
from .produce_denoised_img import produce_denoised_img_no_crop
from .utils import TedataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import parmap
from functools import partial
from .watershed import *
from .watershed import watershed,watershed_per_img,watershed_original,segmentation_with_masking
from .median_filter import apply_median_filter_cpu, apply_median_filter_gpu, apply_median_filter_gpu_simple


def return_img(img_dict): 
    img_info = {}
    for set_num in img_dict.keys():
        for f_num in img_dict[set_num].keys():
            for idx,img in enumerate(img_dict[set_num][f_num]):
                img_info = {'set_num' : set_num, 'f_num' : f_num, 'idx' : idx}
                yield img_info, img

def return_img_fnum(img_dict,specific_f_num_list): 
    img_info = {}
    for set_num in img_dict.keys():
        for f_num in img_dict[set_num].keys():
            if f_num not in specific_f_num_list:
                continue
            for idx,img in enumerate(img_dict[set_num][f_num]):
                img_info = {'set_num' : set_num, 'f_num' : f_num, 'idx' : idx}
                yield img_info, img
def save_hole_img(img,img_info, img_hole_info, pad):
    img = np.squeeze(img.copy())
    CD_info = {'min_CD' : [],'max_CD' : [], 'avg_min_CD': None , 'avg_max_CD' : None }
    frames = []
    for i,j in img_hole_info:
        # crop image patch
        pos = {'r_start' : i-pad, 'r_end' : i+pad, 'c_start' : j-pad, 'c_end' : j+pad}
        need_arrange = []
        for key,item in pos.items():
            if item < 0:
                pos[key] = 0
                need_arrange.append(key)
        for key in need_arrange:
            correspond_key = f"{key[0]}_end"
            pos[correspond_key] = pad*2
        img_patch = img[pos['r_start']:pos['r_end'],pos['c_start']:pos['c_end']].copy()
        
        img_patch_uint8 = (img_patch*255).astype('uint8')
        frames.append(img_patch_uint8)
    # frames.append(np.ones((pad*2,pad*2),dtype=np.uint8))
    key = f'{img_info["set_num"]}_{img_info["f_num"]}_{img_info["idx"]:02d}'
    imageio.mimsave(f"./hole_img/{key}.gif",
                        frames, fps=2.5)
## get_CD subfunction
def get_crop_region(hole_idx, pad):
    try :   
        i,j = hole_idx
    except :
        return None
    pos = {'r_start' : i-pad, 'r_end' : i+pad, 'c_start' : j-pad, 'c_end' : j+pad}
    need_arrange = []
    for key,item in pos.items():
        if item < 0:
            pos[key] = 0
            need_arrange.append(key)
    for key in need_arrange:
        correspond_key = f"{key[0]}_end"
        pos[correspond_key] = pad*2
    return pos
    # print(img_patch.shape)
def crop_image(img,  hole_idx, pad):
    pos = get_crop_region(hole_idx, pad)
    if pos is None:
        return None
    img_patch = img[pos['r_start']:pos['r_end'],pos['c_start']:pos['c_end']].copy()
    if img_patch.shape != (pad*2,pad*2):
        return None
    else :
        return img_patch
def get_CD_per_hole(hole_idx, img, img_info,  pad, frames = [], print_img=False,save_img=False):
    i,j = hole_idx
    img_patch = crop_image(img, {i,j}, pad)
    if img_patch is None:
        return None,None, None

    # leave only circle imagein patch 
    circle_val = img[i,j]
    img_patch[img_patch != circle_val] = 0

    (cX,cY), circle_with_marker = find_center(img_patch)
    if (cX,cY) == (None,None):
        return None,None, None

    if print_img is True:
        print("center : ",cX,cY)
        plt.title(f"pos : {i},{j}")
        plt.imshow(img_patch)
        plt.scatter(cX,cY,color='r',marker='.')
        plt.pause(0.01)
    frame = None
    if save_img is True:
        img_patch_uint8 = (img_patch.copy() / img_patch.max())
        img_patch_uint8 = (img_patch_uint8*255).astype('uint8')
        frame = img_patch_uint8
    search_angle = list(range(0,180))# (map(lambda x : int(x),range(0,180)))

    # find min/max CD
    # partial_func = partial(measure_CD,img=img_patch,cX=cX,cY=cY,debug=False)
    # measured_CD = parmap.map(partial_func,range(0,180), pm_processes=12)
    # find min/max CD for jupyter
    measured_CD = []
    for angle in range(180):
        measured_CD.append(measure_CD(angle,img_patch,cX,cY,debug=False))

    if None in measured_CD:
        print(f"{img_info['set_num']}, {img_info['f_num']} {img_info['idx']} {i},{j} hole cannot get CD measurement due to None")
        cyan = (128, 128, 0)
        patch_img = cv2.rectangle(img, (j-pad//2, j+pad//2), (i+pad//2, i-pad//2), cyan, -1)
        patch_img = crop_image(patch_img, {i,j}, pad)
        return None,None, None # patch_img
    
    min_CD, max_CD = min(measured_CD), max(measured_CD)
    # print(measured_CD)
    return min_CD, max_CD, frame
def get_CD(img,img_info, img_hole_info, pad, print_img=False,save_img=False):
    img = np.squeeze(img.copy())
    CD_info = {'min_CD' : [],'max_CD' : [], 'avg_min_CD': None , 'avg_max_CD' : None }
    frames=[]
    partial_fun = partial(get_CD_per_hole,img=img,img_info=img_info,pad=pad,print_img=print_img,save_img=save_img)

    # min_CD_list, max_CD_list, frames = parmap.map(partial_fun,img_hole_info, pm_processes=32)

    CD_return = parmap.map(partial_fun,img_hole_info, pm_processes=32, pm_pbar=True )
    CD_return = np.array(CD_return)
    min_CD_list, max_CD_list, frames = CD_return[:,0], CD_return[:,1], CD_return[:,2]
    min_CD_list = list(filter(lambda x : x is not None, min_CD_list))
    max_CD_list = list(filter(lambda x : x is not None, max_CD_list))
    frames = list(filter(lambda x : x is not None, frames))

    CD_info['min_CD'] = min_CD_list
    CD_info['max_CD'] = max_CD_list
    
    # for i,j in img_hole_info:
    #     # crop image patch
    #     (min_CD, max_CD), frames = get_CD_per_hole(img, img_info, {i,j}, pad, print_img,save_img)

    #     CD_info['min_CD'].append(min_CD)
    #     CD_info['max_CD'].append(max_CD)
    if save_img is True :
        key = f'{img_info["set_num"]}_{img_info["f_num"]}_{img_info["idx"]:02d}_hole'
        imageio.mimsave(f"{img_info['folder_path']}/{key}_{len(frames)}.gif",
                        frames, fps=2.5)
    CD_info['avg_min_CD'] = np.mean(CD_info['min_CD'])
    CD_info['avg_max_CD'] = np.mean(CD_info['max_CD'])
    return CD_info

def get_metric(CD_dict, local_CD_info,key, collect_metric_list, method):
    for metric in collect_metric_list:
        CD_dict[key][f"{method}_{metric}"] = local_CD_info[metric]
    return CD_dict


def return_img(img_dict): 
    img_info = {}
    for set_num in img_dict.keys():
        for f_num in img_dict[set_num].keys():
            for idx,img in enumerate(img_dict[set_num][f_num]):
                img_info = {'set_num' : set_num, 'f_num' : f_num, 'idx' : idx}
                yield img_info, img
def denoise_and_segment_img_uint8(model, img,img_info, img_hole_info,print_img=False):
    if img.shape != (1,1,1474,3010):
        while len(img.shape) < 4:
            img = np.expand_dims(img,axis=0)
        if len(img.shape) > 4:
            img = np.squeeze(img)
            img = np.expand_dims(img,axis=0)
            img = np.expand_dims(img,axis=0) # (1,1,1474,3010)
    # print(img.shape)
    if type(model) is type(apply_median_filter_gpu_simple) :
        denoised_img = model(img)
    else :
        denoised_img = model.eval(img)
    denoised_img_uint8 = (denoised_img[0][0]*255).astype('uint8')
    key = f'{img_info["set_num"]}_{img_info["f_num"]}_{img_info["idx"]:02d}_denoised'
    cv2.imwrite(f"{img_info['folder_path']}/{key}.png",denoised_img_uint8)

    segmentataion_img_uint8,masked_img = segmentation_with_masking(denoised_img_uint8,img_info, img_hole_info)
    key = f'{img_info["set_num"]}_{img_info["f_num"]}_{img_info["idx"]:02d}_segmented'
    cv2.imwrite(f"{img_info['folder_path']}/{key}.png",segmentataion_img_uint8)
    key = f'{img_info["set_num"]}_{img_info["f_num"]}_{img_info["idx"]:02d}_masked'
    cv2.imwrite(f"{img_info['folder_path']}/{key}.png",masked_img)
    
    segmentataion_img_uint8 = np.expand_dims(segmentataion_img_uint8, axis=0)
    # print(segmentataion_img_uint8.shape)
    return segmentataion_img_uint8 # (1,1474,3010)



def CD_process(model,img,img_info, img_hole_info,pad = 150,print_img=False,save_img=False) -> dict:
    # print(img_info) img_info['method']
    if save_img is True:
        folder_path = f"./process_img/{img_info['method']}_{img_info['fbi_weight_dir']}_img"
        os.makedirs(folder_path,exist_ok=True)
        img_info['folder_path'] = folder_path
    segmentataion_img_uint8 = denoise_and_segment_img_uint8(model,img,img_info,img_hole_info,print_img=False)
    CD_info = get_CD(segmentataion_img_uint8,img_info, img_hole_info, pad,print_img,save_img)
    print(f"{img_info['set_num']}_{img_info['f_num']}_{img_info['idx']:02d}th image - Min/Max (len : {len(CD_info['min_CD'])}) CD : \
    {CD_info['avg_min_CD']:.4f} ~ {CD_info['avg_max_CD']:.4f}")
    
    
    return CD_info


