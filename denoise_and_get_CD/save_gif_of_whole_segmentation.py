import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # "0,1,2,3" # "0"
import numpy as np
import h5py
import cv2
import os, sys
import json
import argparse
from glob import glob
from core.get_args import get_args
from core.produce_denoised_img import produce_denoised_img_no_crop
from core.utils import TedataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from core.watershed import *
from core.watershed import watershed,watershed_per_img,watershed_original
from core.median_filter import apply_median_filter_cpu, apply_median_filter_gpu, apply_median_filter_gpu_simple
import imageio

img_dict = np.load("./intermediate_result/segmentation_img_F08_v2.npy",allow_pickle=True).item()

# Save whole segementation gif file
path = "./segmentation_img/tmp_whole/"
os.makedirs(path,exist_ok=True)
for set_num in img_dict.keys():
    for f_num in img_dict[set_num].keys():
        if 'F64' in f_num:
            continue
        for idx,img in enumerate(img_dict[set_num][f_num]):
            plt.title(f"{set_num}, {f_num}")
            plt.imshow(img)
            plt.savefig(f"{path}{set_num}_{f_num}_{idx:02d}.png")
    print(f"=== {set_num} {img_dict[set_num].keys()} finish ===")
path += "*"
frames = []
for img_path in sorted(glob(path)):
    if 'F64' in img_path:
        continue
    image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    # print(img_path)
    # print(image.shape)
    frames.append(image)
imageio.mimsave('./RN2N_F8_whole_segmentation.gif', frames, fps =2.5)
