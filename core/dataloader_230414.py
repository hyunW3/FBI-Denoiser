import sys
import h5py
import numpy as np
import random
import torch

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
def augment_and_scale_img(img1,img2,img3):
    if random.random() > 0.5:
        img1 = np.fliplr(img1)
        img2 = np.fliplr(img2)
        img3 = np.fliplr(img3)
    # Random vertical flipping
    if random.random() > 0.5:
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
        img3 = np.flipud(img3)
    img1 = torch.from_numpy(img1.copy()).float()
    img2 = torch.from_numpy(img2.copy()).float()
    img3 = torch.from_numpy(img3.copy()).float()
    return img1,img2,img3
def loader_for_N2N(data_path = "/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/train_Samsung_SNU_patches_230414.hdf5",
                 return_img_info = False):
    """
        transform (16, 256, 256) to 15 pairs of two F01 images
    """
    data = h5py.File(data_path,'r')
    if return_img_info is True :
        img_info = {}
    for i in range(1,42+1):
        keys = list(filter(lambda x : f"F01_{i:02d}" in x, data.keys()))
        if keys == []:
            continue

        target_key = f"F32_{i:02d}"
        data_list = np.array([data[key] for key in keys])
        target_data = np.array(data[target_key])
        for idx in range(35): # 40 in (16,40,256,256)
            # print(target_data[idx].shape, type(target_data[idx]))
            target_data_sample = np.expand_dims(target_data[idx],axis=0)
            
            for pair_idx in range(data_list.shape[0]-1): # 16 in (16,40,256,256)
                # (256,256) -> (1,256,256)
                noisy1 = np.expand_dims(data_list[pair_idx,idx],axis=0)
                noisy2 = np.expand_dims(data_list[pair_idx+1,idx],axis=0)
                noisy1, noisy2,target_data_sample_tmp  = augment_and_scale_img(noisy1,noisy2,target_data_sample)
                # print(noisy1.shape, "\n",min(noisy1),max(noisy1))
                if return_img_info is True:
                    img_info = {'img_name' : keys[0][:-3], 'img_idx' : idx, 'pair_idx' : pair_idx}
                    yield img_info, noisy1,noisy2, target_data_sample_tmp
                else :
                    yield noisy1,noisy2,target_data_sample_tmp
from torch.utils.data import Dataset
class N2N_F01_Dataset(Dataset):
    def __init__(self,data_path="/mnt/ssd/hyun/fbi-net/FBI-Denoiser/data/train_Samsung_SNU_patches_230414.hdf5",
                 return_img_info = False):
        self.dataset = list(loader_for_N2N(data_path=data_path,
                                        return_img_info=return_img_info)
                            )
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        
        return self.dataset[index]

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
    gen = N2N_F01_Dataset(return_img_info=True)
    gen = torch.utils.data.DataLoader(gen, batch_size=2, shuffle=True, num_workers=4)
        
    idx = 0
    info = None
    print(len(gen))
    for epoch in range(2):
        # for idx, (noisy1, noisy2) in enumerate(gen):
        for idx, (img_info,noisy1,noisy2,clean) in enumerate(gen):
            print(img_info)

            # print(".",end="")
            if idx % 50 == 0:
                print("\n",idx, noisy1.shape, noisy2.shape,clean.shape)
                break
            sys.exit(0)