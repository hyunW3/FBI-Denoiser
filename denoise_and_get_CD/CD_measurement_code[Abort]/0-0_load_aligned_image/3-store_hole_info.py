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
from core.data_loader import sem_generator_path
from core.produce_denoised_img import produce_denoised_img_no_crop
from core.utils import TedataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from core.watershed import *
from core.watershed import watershed,watershed_per_img,watershed_original
from core.median_filter import apply_median_filter_cpu, apply_median_filter_gpu, apply_median_filter_gpu_simple
import wandb
import pandas as pd
wandb.init(project="CD_process",name='hole_info_median')

# remember hole info

data_path = "../Samsung_SNU_dataset_1536x3072"
target = "median_filter"
hole_info_dict = {}
# for target in [['F16','v2']]:
#     target = f"{target[0]}_{target[1]}"
#     target_list.append(target)
set_list = list(filter(lambda x: x[0] == 'S', os.listdir(data_path)))
for set_num in set_list:
    for img_path in glob(f"{data_path}/{set_num}/*.png"):
        file_name = img_path.split("/")[-1][:-4]#.spilt('.')[0]
        print(file_name)
        if file_name[0] == '.':
            continue
        f_num, image_num = file_name.split("_")
        set_number = int(set_num[3:])
        f_number = int(f_num[1:])
        # file_name = f"F{f_number:02d}_{int(image_num):02d}.png"
        
        
        image_arr = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        key = f"SET{set_number:02d}_F{f_number:02d}_{int(image_num):02d}"
        hole_info_dict[key] = []
        print(set_num,key)

        hole_info = find_possible_section(image_arr)
        
        
        hole_info_dict[key].append(hole_info)
                    
with open(f'./hole_info_{target}.json', 'w') as f:
    f.write(json.dumps(hole_info_dict,indent="\t"))
wandb.log({"hole_info":wandb.Table(dataframe=pd.DataFrame(hole_info_dict))})
del hole_info_dict
print("save complete",target)

