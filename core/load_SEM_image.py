import os
import cv2
import numpy as np
data_path = "data/Samsung_SNU"
num_per_Ffolder = 16
def load_set_images(set_path : str) -> dict:
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
    
def load_whole_image(path : str) -> dict:
    whole_images : dict = dict()
    for set_number in sorted(os.listdir(path)):
        set_path = os.path.join(path,set_number)
        whole_images[set_number] = load_set_images(set_path)
    return whole_images
def crop_image(image : np.array,size : int):
    return image[:size,:size]
def load_single_image(set_num : int, f_num : int, num : int):
    flag1 = set_num <0 or set_num >4
    flag2 = f_num not in [8,16,32,64]
    flag3 = num not in range(1,num_per_Ffolder+1)
    if flag1 or flag2 or flag3:
        print(f"True means Undesirable arugment\n1st : {flag1}, 2nd : {flag2}, 3rd : {flag3}")
        raise    
    image = cv2.imread(f"data/Samsung_SNU/[SET {set_num}]/F{f_num}/{num}_F{f_num}.png",cv2.IMREAD_GRAYSCALE)                                                               
    image = crop_image(image,256)
    return image