import torch
import torch.backends.cudnn as cudnn
import torchvision
import cv2 
from matplotlib import pyplot as plt
import os,sys, glob
from tqdm import tqdm
import numpy as np
import PIL
import random
from PIL import Image
import gc
import h5py
from sklearn.model_selection import train_test_split
from typing import Generator
from core.utils import seed_everything,poolcontext
from functools import partial


    
# make align b/w two img
def get_shift_info(img1, img2, v_width=30,h_width=30, crop_size=256,log=False):
    """
    img1's shift value compared to img2
    img1 : noisy image
    img2 : clean image (reference)
    v_width, h_width : width for search space
    crop_size : default 256
    log : bool, show log if True
    """
    assert(img1.shape == img2.shape)
    top = img1.shape[0]//2
    left = img1.shape[1]//2
    from skimage.metrics import structural_similarity,peak_signal_noise_ratio
    peak_ssim = None
    peak_psnr = None
    if log is True:
        ssim, psnr = structural_similarity(img2,img1), peak_signal_noise_ratio(img2,img1)
        print(f"original ssim (whole) : {ssim}, psnr : {psnr}")
        ssim  = structural_similarity(img2[top:top+crop_size,left:left+crop_size],img1[top:top+crop_size,left:left+crop_size])
        psnr  = peak_signal_noise_ratio(img2[top:top+crop_size,left:left+crop_size],img1[top:top+crop_size,left:left+crop_size])
        print(f"original ssim (patch): {ssim}, psnr : {psnr}")
    for v_shift in range(-v_width,v_width):
        #v_shift *= -1
        for h_shift in range(-h_width,h_width):
            #print(f"===== {v_shift}, {h_shift} ======")
            patch1 = img1[top+v_shift:top+crop_size+v_shift,left+h_shift:left+crop_size+h_shift]
            patch2 = img2[top:top+crop_size,left:left+crop_size]
            ssim, psnr = structural_similarity(patch2,patch1), peak_signal_noise_ratio(patch2,patch1)
            #print(ssim, psnr)
            if peak_ssim is None:
                peak_ssim = {'ssim' : ssim, 'psnr' : psnr, 'v_shift' : v_shift, 'h_shift' : h_shift}
            elif peak_ssim['ssim'] < ssim:
                peak_ssim = {'ssim' : ssim, 'psnr' : psnr, 'v_shift' : v_shift, 'h_shift' : h_shift}

            if peak_psnr is None:
                peak_psnr = {'ssim' : ssim, 'psnr' : psnr, 'v_shift' : v_shift, 'h_shift' : h_shift}
            elif peak_psnr['psnr'] < psnr:
                peak_psnr = {'ssim' : ssim, 'psnr' : psnr, 'v_shift' : v_shift, 'h_shift' : h_shift}
    if log is True:
        print(peak_ssim)
        print(peak_psnr)
    if peak_ssim['v_shift'] == peak_psnr['v_shift'] and peak_ssim['h_shift'] == peak_psnr['h_shift'] :
        return [peak_psnr['v_shift'],peak_psnr['h_shift']]
    elif (peak_ssim['ssim'] - peak_psnr['ssim']) < (peak_psnr['psnr'] - peak_ssim['psnr']):
        return [peak_psnr['v_shift'],peak_psnr['h_shift']]
    else :
        return [peak_ssim['v_shift'],peak_ssim['h_shift']]
def show_patch(crop_im):
    crop_im = crop_im.permute(1,2,0).cpu().numpy()
    crop_im = crop_im.astype('uint8')
    PIL.Image.fromarray(crop_im)
def read_image_and_crop(iter_num, image : np.array, target : np.array, 
                        device_id : int, crop_size = 256, is_test=False): #, return_list, lock):

    top = random.randrange(0,image.shape[0]-crop_size)
    left = random.randrange(0,image.shape[1]-crop_size)

    # torchvision transform not work with multiprocessing
    im_t,target_im_t = image/255. , target/255.
    crop_im = im_t[top:top+crop_size,left:left+crop_size]
    crop_target_im = target_im_t[top:top+crop_size,left:left+crop_size]
    if is_test is True:
        print(f"{os.getpid()} is cropped",flush=True)
    return crop_im,crop_target_im

def make_patch_of_image(image_path,data_path,set_num,device_id,num_crop=500,num_cores=16,is_test=False):
    #noisy_patch = torch.Tensor(num_crop,256,256).cuda(device_id) 
    #target_patch = torch.Tensor(num_crop,256,256).cuda(device_id) 
    noisy_patch = np.ones((num_crop,256,256))
    target_patch = np.ones((num_crop,256,256))
    
    image_num = image_path.split("_")[-1]
    clean_path = os.path.join(data_path,set_num,f"F64_{image_num}")
    #print(image_num, clean_path)
    im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    target_im = cv2.imread(clean_path,cv2.IMREAD_GRAYSCALE)
    
    
    num_iter = range(num_crop)
    with poolcontext(num_cores) as p:
        r = p.map(partial(read_image_and_crop, image=im,target=target_im, device_id=device_id,is_test=is_test),num_iter)
        if is_test is True:
            print(os.getpid(),len(r))
        for index,pair in enumerate(r):
            im_cropped, target_im_cropped = pair
            #noisy_patch[index],target_patch[index] = im_cropped.cuda(device_id),target_im_cropped.cuda(device_id)
            noisy_patch[index],target_patch[index] = im_cropped, target_im_cropped
    return noisy_patch,target_patch

def return_val_test_index():
    random_seed = random.randint(0,2**32-1)
    seed_everything(random_seed)
    val_image_index = random.randint(0,15)
    test_image_index = random.randint(0,15)
    while val_image_index == test_image_index:
        test_image_index = random.randint(0,15)
    return (val_image_index, test_image_index)

def append_tensor(original,new):
    if original is None:
        original = new
    else :
        original = np.concatenate((original, new),axis=0)
        #original = torch.cat((original, new),axis=0)
    return original
def save_to_hdf5(noisy_patch, target_patch, set_num :int ,type_name : str):
    if type_name not in ['train','val','test']:
        print("error in type_name")
        sys.exit(-1)
    #noisy_patch, target_patch = noisy_patch.cpu().numpy(), target_patch.cpu().numpy()
    with h5py.File(f"{type_name}_Samsung_SNU_patches_{set_num}.hdf5","w") as f:
        f.create_dataset("noisy_images", noisy_patch.shape, dtype='f', data=noisy_patch)
        f.create_dataset("clean_images", target_patch.shape, dtype='f', data=target_patch)
    gc.collect()

def make_dataset_per_set(data_path, set_num ,device_id,print_lock,num_crop=500,is_test=False):
    num_test_image = 3
    num_cores = 16
    set_path = os.path.join(data_path,set_num)
    print(f"====== {set_path} ======")
    image_list = sorted(os.listdir(set_path)) # 16
    image_list = list(filter(lambda x : "F64" not in x,image_list))
    image_list = list(filter(lambda x : "checkpoints" not in x,image_list))
    len_train = len(image_list)-num_test_image*2
    
    train_noisy_patch, train_target_patch = None, None
    val_noisy_patch, val_target_patch = None, None
    test_noisy_patch, test_target_patch = None, None
    gc.collect()
    noisy_f_num = ['F8','F16','F32']
    for i,f_num in enumerate(noisy_f_num): # f_num 마다 구분
        val_index, test_index = return_val_test_index()
        #print("======",i, f_num, set_path)
        #print("======",val_index,test_index)
        #print(set_path)
        for image_idx, image_path in tqdm(enumerate(glob.glob(f"{set_path}/{f_num}*.png"))):
            noisy, target = make_patch_of_image(image_path,data_path,set_num,device_id,num_crop,num_cores,is_test)
            if is_test is True:
                print(image_idx,image_path)
                print(noisy.shape,target.shape)
            if image_idx == val_index:
                #print(f"This is val image")
                val_noisy_patch = append_tensor(val_noisy_patch,noisy)
                val_target_patch = append_tensor(val_target_patch,target)
            elif image_idx == test_index:
                #print(f"This is test image")
                test_noisy_patch = append_tensor(test_noisy_patch,noisy)
                test_target_patch = append_tensor(test_target_patch,target)
            else :
                train_noisy_patch = append_tensor(train_noisy_patch,noisy)
                train_target_patch = append_tensor(train_target_patch,target)
            noisy, target = None, None
            gc.collect()
            if is_test is True:
                SET_NUM = f"SET{device_id+1}"
    print_lock.acquire()
    print(f"create dataset of {set_num}")
    print(f"[train] noisy & target(clean) patch shape")
    print(f"{train_noisy_patch.shape} and {train_target_patch.shape}")
    print(f"max : {train_target_patch[0].max()}, min : {train_target_patch[0].min()}")
    print(f"[val] noisy & target(clean) patch shape")
    print(f"{val_noisy_patch.shape} and {val_target_patch.shape}")
    print(f"max : {val_noisy_patch[0].max()}, min : {val_target_patch[0].min()}")
    print(f"[test] noisy & target(clean) patch shape")
    print(f"{test_noisy_patch.shape} and {test_target_patch.shape}")
    print(f"max : {test_noisy_patch[0].max()}, min : {test_target_patch[0].min()}")
    print_lock.release()
    save_to_hdf5(train_noisy_patch, train_target_patch,set_num,'train')
    save_to_hdf5(val_noisy_patch, val_target_patch,set_num,'val')
    save_to_hdf5(test_noisy_patch, test_target_patch,set_num,'test')
    #x_train, x_test, y_train, y_test = train_test_split(noisy_patch,target_patch,test_size=0.08,shuffle=True)
    #print(x_train.shape, x_test.shape
    del train_noisy_patch, train_target_patch, val_noisy_patch, val_target_patch,test_noisy_patch, test_target_patch
    torch.cuda.empty_cache()
    print(f"complete {set_path}")