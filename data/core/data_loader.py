import torch
import cv2 
from matplotlib import pyplot as plt
import os
import numpy as np
import PIL
from PIL import Image

def load_set_images(set_path : str,num_per_Ffolder : int) -> dict:
    set_number = set_path.split("/")[-1]
    print(f"=== extracting {set_number} ===")

    images = dict()
    for F_number in os.listdir(set_path):
        F_path = os.path.join(set_path,F_number)
        print(F_path)
        for i in range(1,num_per_Ffolder+1):
            file_name = f"{i}_{F_number}.png"
            absolute_path = os.path.join(F_path,file_name)
            if not os.path.exists(absolute_path) :
                print(f"{absolute_path}" is not exist)
            else :
                image = cv2.imread(absolute_path,cv2.IMREAD_GRAYSCALE)
                if F_number in images :
                    images[F_number][file_name] = image
                else :
                    images[F_number] = {file_name : image}
    return images
    
def load_whole_image(path : str,num_per_Ffolder : int) -> dict:
    whole_images : dict = dict()
    for set_number in sorted(os.listdir(path)):
        set_path = os.path.join(path,set_number)
        whole_images[set_number] = load_set_images(set_path,num_per_Ffolder)
    return whole_images
# whole_images = load_whole_image(data_path)
def sem_generator(data_path : str):
    for set_num in sorted(os.listdir(data_path)):
        set_path = os.path.join(data_path,set_num)
        cnt = 0
        set_num = set_num[1:-1]
        set_num = set_num[:3] + set_num[-1]
        for f_num in sorted(os.listdir(set_path)):
            f_path = os.path.join(set_path,f_num)
            avg_info = {"lowest_size" : -1, "mean_size" : 0, "cnt" : 0, "highest_size" : -1}
            for file_name in sorted(os.listdir(f_path)):
                file_path = os.path.join(f_path,file_name)
                yield [set_num,f_num,file_name.split("_")[0],cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)]