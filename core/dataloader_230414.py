import sys
import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
def data_parser(data_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/train_Samsung_SNU_patches_230414.hdf5",
                dataset_type = 'train', return_img_info = False):
    """
    it returns set of 16 frames of F01 images
    """
    data = h5py.File(data_path,'r')
    if return_img_info is True :
        img_info = {}
    for i in range(1,42+1):
        keys = list(filter(lambda x : f"F01_{i:02d}" in x, data.keys()))
        if keys == []:
            continue
        target_key = f"F32_{i:02d}"
        # print(keys)
        data_list = np.array([data[key] for key in keys])
        target_data = np.array(data[target_key])
         
        for idx in range(data_list.shape[1]):
            # print(keys[0],idx)
            if return_img_info is True:
                img_info = {'img_name' : keys[0], 'img_idx' : idx}
                yield img_info, data_list[:,idx], target_data[idx]
            else :
                yield data_list[:,idx], target_data[idx]

def to_tensor(img1,img2,img3):
    img1 = torch.from_numpy(img1.copy()).float()
    img2 = torch.from_numpy(img2.copy()).float()
    img3 = torch.from_numpy(img3.copy()).float()
    return img1,img2,img3
def augment_img(img1,img2,img3):
    if random.random() > 0.5:
        img1 = np.fliplr(img1)
        img2 = np.fliplr(img2)
        img3 = np.fliplr(img3)
    # Random vertical flipping
    if random.random() > 0.5:
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
        img3 = np.flipud(img3)
    
    return img1,img2,img3
def average_img(imgs : np.array):
    """
        get average of imgs [num_imgs,num_crop, height, width] -> [num_crop, height, width]
    """    
    return np.mean(imgs, axis = 0)
    

def loader_for_RN2N(data_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/train_Samsung_SNU_patches_230414.hdf5",
                 x_avg_num = 1,y_avg_num = 1, return_img_info = False):
    """
        transform (16, 256, 256) to 15 pairs of two F01 images
        
    """
    if x_avg_num + y_avg_num >16:
        raise ValueError("x_avg_num + y_avg_num should be less than 16")
    # {2 : 33, 4 : 140, 8 : 260, 16 : 520}
    num_crop_to_train = {2 : 66, 3 : 105, 4 : 132, 6 : 263, 8 : 263, 16 : 526}
    data = h5py.File(data_path,'r')
    if return_img_info is True :
        img_info = {}
    key = x_avg_num + y_avg_num
    # key = 2**np.ceil(np.log2(x_avg_num+y_avg_num)) # 2,4,8,16으로 올림해야한다.
    # print("sum of fnumber :",key,num_crop_to_train[key])
    num_images = 42
    for i in range(1,num_images+1):
        keys = list(filter(lambda x : f"F01_{i:02d}" in x, data.keys()))
        if keys == []:
            continue

        target_key = f"F32_{i:02d}"
        data_list = np.array([data[key] for key in keys])
        target_data = np.array(data[target_key]) 
        
        for idx in range(num_crop_to_train[key]): # 40 in (16,40,256,256)
        # for idx in range(25): # 40 in (16,40,256,256)
            # print(target_data[idx].shape, type(target_data[idx]))
            target_data_sample = np.expand_dims(target_data[idx],axis=0)
            step = x_avg_num + y_avg_num
            for pair_idx in range(0,data_list.shape[0],step): # 16 in (16,40,256,256)
                if pair_idx + step > data_list.shape[0]:
                    break
                noisy1 = data_list[pair_idx:pair_idx+x_avg_num,idx]
                noisy2 = data_list[pair_idx+x_avg_num:pair_idx+x_avg_num + y_avg_num,idx]
                # print(f"{pair_idx} ~ {pair_idx+avg_num}, {pair_idx+avg_num} ~ {pair_idx+2*avg_num}")
                # print(noisy1.shape,noisy2.shape)
                noisy1 = average_img(noisy1)
                noisy2 = average_img(noisy2)
                # print(idx,pair_idx,noisy1.shape,noisy2.shape)
                # (256,256) -> (1,256,256)
                noisy1 = np.expand_dims(noisy1,axis=0)
                noisy2 = np.expand_dims(noisy2,axis=0)
                
                # print(noisy1.shape, "\n",min(noisy1),max(noisy1))
                if return_img_info is True:
                    img_info = {'img_name' : keys[0][:-3], 'img_idx' : idx, 
                                'pair_idx' : pair_idx, 'avg_num' : (x_avg_num,y_avg_num)}
                    yield img_info, noisy1,noisy2, target_data_sample
                else :
                    yield noisy1,noisy2,target_data_sample

class RN2N_Dataset(Dataset):
    def __init__(self,data_path="/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/train_Samsung_SNU_patches_230414.hdf5",
                 x_avg_num=1,y_avg_num = 1, transform=True, return_img_info = False):
        self.dataset = list(loader_for_RN2N(data_path=data_path,
                                            x_avg_num=x_avg_num,y_avg_num = y_avg_num, 
                                        return_img_info=return_img_info)
                            )
        self.transform = transform
        self.x_avg_num = x_avg_num
        self.y_avg_num = y_avg_num
        self.return_img_info = return_img_info
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        if self.return_img_info is True:
            img_info, noisy1,noisy2, target_data = self.dataset[index]
        else :
            noisy1,noisy2, target_data = self.dataset[index]
        
        if self.transform is True:
            noisy1, noisy2,target_data  = augment_img(noisy1,noisy2,target_data)
        
        noisy1, noisy2,target_data = to_tensor(noisy1, noisy2,target_data)
        
        if self.return_img_info is True:
            return img_info, noisy1,noisy2, target_data
        else :
            return noisy1,noisy2, target_data
if __name__ == '__main__':
    # gen = list(data_parser(return_img_info = True))
    # print("length : ", len(gen))
    # random.shuffle(gen)
    # gen = iter(gen)
    # for i in range(10):
    #     img_info, noisy, clean = next(gen)
    #     print(img_info)
    #     print(noisy.shape, clean.shape)
    print("=====================================")
    print("N2N")
    gen = RN2N_Dataset(x_avg_num=4,y_avg_num=4, return_img_info=True)
    gen = torch.utils.data.DataLoader(gen, batch_size=2, shuffle=True, num_workers=20)
    print("dataloader load end")
    # print(len(gen)) # avg_num : 4 -> 1538
    
    idx = 0
    info = None
    for epoch in range(2):
        # for idx, (noisy1, noisy2) in enumerate(gen):
        for idx, (img_info,noisy1,noisy2,clean) in enumerate(gen):
            print(img_info, noisy1.shape, noisy2.shape,clean.shape)

            # print(".",end="")
            if idx % 50 == 0:
                print("\n",idx, noisy1.shape, noisy2.shape,clean.shape)
                break
            sys.exit(0)