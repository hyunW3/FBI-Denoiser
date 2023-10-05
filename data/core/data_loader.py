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
        #print(F_path)
        for i in range(1,num_per_Ffolder+1):
            file_name = f"{i}_{F_number}.png"
            absolute_path = os.path.join(F_path,file_name)
            print(absolute_path)
            if not os.path.exists(absolute_path) :
                print(f"{absolute_path} is not exist")
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
def load_whole_image_in_folder(path : str) -> dict:
    whole_images : dict = dict()
    for set_number in sorted(os.listdir(path)):
        set_path = os.path.join(path,set_number)
        images = {}
        for img_name in sorted(os.listdir(set_path)):
            if "checkpoints" in img_name:
                continue
            img_path = os.path.join(set_path,img_name)
            images[f"{img_name}"] = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        whole_images[set_number] = images
    return whole_images
def crop_image(image : np.array,size : int):
    #return image[:size,:size]
    return image[size*4:size*5,size*4:size*5]
num_per_Ffolder = 16
def load_single_image(set_num : int, f_num : int, num : int, do_crop=False):
    flag1 = set_num <0 or set_num >4
    flag2 = f_num not in [8,16,32,64]
    flag3 = num not in range(1,num_per_Ffolder+1)
    if flag1 or flag2 or flag3:
        print(f"It has Undesirable arugment\n1st : {flag1}, 2nd : {flag2}, 3rd : {flag3}")
        raise    
    image = cv2.imread(f"data/Samsung_SNU/[SET {set_num}]/F{f_num}/{num}_F{f_num}.png",cv2.IMREAD_GRAYSCALE)        
    if do_crop is True:
        image = crop_image(image,256)
    return image
# whole_images = load_whole_image(data_path)
def sem_generator(data_path : str):
    """
        get path & return data
        return : set_num, f_num, image_num, image_array
    """
    for set_num in sorted(os.listdir(data_path)):
        set_path = os.path.join(data_path,set_num)
        cnt = 0
        set_num = set_num[1:-1]
        set_num = set_num[:3] + set_num.split(" ")[-1]
        for f_num in sorted(os.listdir(set_path)):
            f_path = os.path.join(set_path,f_num)
            f_number = int(f_num[1:])
            for file_name in sorted(os.listdir(f_path)):
                file_path = os.path.join(f_path,file_name)
                yield [set_num, f"F{f_number:02d}", file_name.split("_")[0],cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)]
# whole_images = load_whole_image(data_path)
def sem_generator_path(data_path : str):
    for set_num in sorted(os.listdir(data_path)):
        set_path = os.path.join(data_path,set_num)
        cnt = 0
        set_num = set_num[1:-1]
        set_num = set_num[:3] + set_num.split(" ")[-1]
        for f_num in sorted(os.listdir(set_path)):
            f_path = os.path.join(set_path,f_num)
            f_number = int(f_num[1:])
            for file_name in sorted(os.listdir(f_path)):
                file_path = os.path.join(f_path,file_name)
                yield [set_num,f"F{f_number:02d}",file_name.split("_")[0],file_path]

                
def patch_generator(data_path : str):
    for set_num in sorted(os.listdir(data_path)):
        set_path = os.path.join(data_path,set_num)
        cnt = 0
        #set_num = set_num[1:-1]
        #set_num = set_num[:3] + set_num[-1]
        for file_name in sorted(os.listdir(set_path)):
            f_num, file_num  = file_name.split("_")
            file_path = os.path.join(set_path,file_name)
            yield [set_num,f_num,file_num,cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)]