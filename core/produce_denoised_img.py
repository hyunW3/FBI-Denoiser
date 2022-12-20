import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import scipy.io as sio

from datetime import date
from .utils import TedataLoader, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
from .loss_functions import mse_bias, mse_affine, emse_affine
from .models import New_model
from .fcaide import FC_AIDE
from .dbsn import DBSN_Model
from .unet import est_UNet
import h5py
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark=True

class produce_denoised_img(object):
    """
    produce denoised image with pretrained weights
    """
    def __init__(self,_pge_weight_dir=None,_fbi_weight_dir=None, _save_file_name = None, _args = None):
        
        self.args = _args
        assert self.args.pge_weight_dir != None, "Need pretrained PGE weight directory information"
        
        self.pge_weight_dir = '../weights/' + self.args.pge_weight_dir
        

        self.result_psnr_arr = []
        self.result_ssim_arr = []
        self.result_time_arr = []
        self.result_denoised_img_arr = []
        
        self.save_file_name = _save_file_name
        self.date = date.today().isoformat()
        num_output_channel = 1
        if self.args.loss_function== 'MSE': # upper bound 1-1(optional)
            self.loss = torch.nn.MSELoss()
            num_output_channel = 1
            self.args.output_type = "linear"
        elif self.args.loss_function == 'N2V': #lower bound
            self.loss = mse_bias
            num_output_channel = 1
            self.args.output_type = "linear"
        elif self.args.loss_function == 'MSE_Affine': # 1st(our case upper bound)
            self.loss = mse_affine
            num_output_channel = 2
            self.args.output_type = "linear"
        elif self.args.loss_function == 'EMSE_Affine':
            
            self.loss = emse_affine
            num_output_channel = 2
            
            ## load PGE model
            self.pge_model=est_UNet(num_output_channel,depth=3)
            self.pge_model.load_state_dict(torch.load(self.pge_weight_dir))
            self.pge_model=self.pge_model.cuda()
            
            for param in self.pge_model.parameters():
                param.requires_grad = False
            
            self.args.output_type = "sigmoid"
        
        if self.args.model_type == 'FC-AIDE':
            self.model = FC_AIDE(channel = 1, output_channel = num_output_channel, filters = 64, num_of_layers=10, output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value)
        elif self.args.model_type == 'DBSN':
            self.model = DBSN_Model(in_ch = 1,
                            out_ch = num_output_channel,
                            mid_ch = 96,
                            blindspot_conv_type = 'Mask',
                            blindspot_conv_bias = True,
                            br1_block_num = 8,
                            br1_blindspot_conv_ks =3,
                            br2_block_num = 8,
                            br2_blindspot_conv_ks = 5,
                            activate_fun = 'Relu')
        else:
            self.model = New_model(channel = 1, output_channel =  num_output_channel, filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = self.args.model_type, output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value)
        self.model.load_state_dict(torch.load(_fbi_weight_dir))
        self.model = self.model.cuda()
        
    def get_X_hat(self, Z, output):

        X_hat = output[:,:1] * Z + output[:,1:]
            
        return X_hat
        
    def eval(self,image):
        """
        Evaluates denoiser on validation set.
        image : (1474, 3010) samsung sem image
        TODO : implement other denoise method(EMSE_Affine .. etc)
        
        """

        psnr_arr = []
        ssim_arr = []
        image = torch.Tensor(image/255.).cuda()
        x_len = image.shape[0]
        y_len = image.shape[1]
        crop_size = self.args.crop_size
        length_of_x = len(range(0,x_len,crop_size))
        length_of_y = len(range(0,y_len,crop_size))
        imgs = torch.zeros([length_of_x*length_of_y,1,crop_size,crop_size])
        
        #print(imgs.shape, length_of_x, length_of_y)
        index = 0
        with torch.no_grad():

            for x in range(0,x_len,crop_size):
                for y in range(0,y_len,crop_size):
                    img = image[x:x+crop_size,y:y+crop_size]
                    
                    if img.shape[0] != crop_size and img.shape[1] != crop_size:
                        img = image[-crop_size:,-crop_size:]
                    elif img.shape[0] != crop_size:
                        img = image[-crop_size:,y:y+crop_size]
                            
                    elif img.shape[1] != crop_size:
                        img = image[x:x+crop_size,-crop_size:]
                    
                    img = img.unsqueeze(axis=0).unsqueeze(axis=0)
                    # print(x,y,img.shape)
                    # print(x,y,imgs.shape,img.shape)
                    imgs[index] = img 
                    # print(index,imgs[index][0][2][2])
                    index +=1
                    if self.args.debug is True:
                        plt.title(f"{x},{y}")
                        plt.imshow(img[0][0].cpu().numpy())
                        plt.pause(0.1)
            # print(imgs.shape[0]*256**2,torch.count_nonzero(imgs))
            imgs = imgs.cuda()
            output = self.model(imgs)
            if self.args.debug is True:
                print("output shape ",output.shape)
            ## affine mapping에 맞게 이미지를 바꾸어 준다.
            X_hat = np.clip(self.get_X_hat(imgs,output).cpu(), 0, 1)
            if self.args.debug is True:
                plt.subplot(121)
                plt.imshow(X_hat[0][0])
                plt.subplot(122)
                plt.imshow(X_hat[1][0])
                       
            if self.args.debug is True:
                print("X_hat shape ",X_hat.shape)
            denoised_img = torch.zeros(image.shape)
            if self.args.debug is True:
                for i in range(72):
                    plt.imshow(X_hat[i][0])
                    plt.pause(0.1)
            # 나누어서 디노이징한 이미지를 위치에 맞게 넣어준다. 
            # if문은 사이즈가 맞지 않는 마지막부분을 사이즈에 맞게 넣어주는 것. 
            index = 0
            for x in range(0,x_len,crop_size):
                for y in range(0,y_len,crop_size):
                    img = denoised_img[x:x+crop_size,y:y+crop_size]
                    if self.args.debug is True:
                        print(index,x,y,img.shape)
                    
                    if img.shape[0] != crop_size and img.shape[1] != crop_size:
                        #print(denoised_img[x:x+crop_size,y:y+crop_size].shape,imgs[index][0][-img.shape[0]:,-img.shape[1]:].shape)
                        denoised_img[x:x+crop_size,y:y+crop_size] = X_hat[index][0][-img.shape[0]:,-img.shape[1]:]
                    elif img.shape[0] != crop_size:
                        #print(denoised_img[x:x+crop_size,y:y+crop_size].shape, imgs[index][0][-img.shape[0]:,:].shape)
                        denoised_img[x:x+crop_size,y:y+crop_size] = X_hat[index][0][-img.shape[0]:,:]
                    elif img.shape[1] != crop_size:
                        #print(imgs[index][0][:,-img.shape[1]:].shape)
                        denoised_img[x:x+crop_size,y:y+crop_size] = X_hat[index][0][:,-img.shape[1]:]
                    else : 
                        denoised_img[x:x+crop_size,y:y+crop_size] = X_hat[index][0]
                    index +=1
            # denoising is completed
            if self.args.debug is True:
                print(denoised_img.shape)
                plt.imshow(denoised_img.cpu().numpy())
                plt.pause(0.1)
            return denoised_img.numpy()
            
            """
                    if self.args.debug is True:
                        plt.imshow(img.cpu().numpy())
                        plt.pause(0.1)
                    img = img.unsqueeze(axis=0).unsqueeze(axis=0) # 1,1,256,256
                    if self.args.debug is True:
                        print(x,y,img.shape)
                    output = self.model(img)
                    
                    if self.args.debug is True:
                        print(output.shape,output.min(),output.max())
                        plt.imshow(output[0][0].cpu().numpy(),cmap='Spectral')
                        plt.pause(0.1)
                        plt.imshow(output[0][1].cpu().numpy(),cmap='Spectral')
                        plt.pause(0.1)
                    
                    # X_hat = np.clip(self.get_X_hat(img,output).cpu().numpy(), 0, 1)
                    # X_hat = np.clip(self.get_X_hat(img[0][0],output).cpu().numpy(), 0, 1)
                    
                    X_hat = np.clip(self.get_X_hat(img[0][0],output).cpu().numpy(), 0, 1)
                    
                    if self.args.debug is True:
                        print(X_hat.shape)
                        plt.axhline(50,color='r')
                        plt.imshow(X_hat[0][0],cmap='Spectral')
                        plt.pause(0.1)
                        plt.plot(X_hat[0][0][50])
                        plt.pause(0.1)
                        plt.plot(img[0][0][50].cpu().numpy())
                        plt.pause(0.1)
                        sys.exit()
                """
"""
                start = time.time()

                source = source.cuda()
                target = target.cuda()
                
                # Denoise
                if self.args.loss_function =='EMSE_Affine':
                                        
                    est_param=self.pge_model(source)
                    original_alpha=torch.mean(est_param[:,0])
                    original_sigma=torch.mean(est_param[:,1])
                    
                    transformed=gat(source,original_sigma,original_alpha,0)
                    transformed, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed)
                    
                    transformed_target = torch.cat([transformed, transformed_sigma], dim = 1)
#                     target = torch.cat([target,transformed_sigma], dim = 1)

                    output = self.model(transformed)
    
                    # loss = self.loss(output, transformed_target)
        
                else:
                    # Denoise image
                    if self.args.model_type == 'DBSN':
                        output, _ = self.model(source)
                        # loss = self.loss(output, target)
                    else:
                        output = self.model(source)
                        # loss = self.loss(output, target)
                

                if self.args.loss_function == 'MSE':
                    output = output.cpu().numpy()
                    X_hat = np.clip(output, 0, 1)
                    X = target.cpu().numpy()
                    
                elif self.args.loss_function == 'MSE_Affine':
                    
                    Z = target[:,:1]
                    X = target[:,1:].cpu().numpy()
                    X_hat = np.clip(self.get_X_hat(Z,output).cpu().numpy(), 0, 1)
                    
                elif  self.args.loss_function == 'N2V':
                    X = target[:,1:].cpu().numpy()
                    X_hat = np.clip(output.cpu().numpy(), 0, 1)
                    
                
                elif self.args.loss_function == 'EMSE_Affine':
                    
                    transformed_Z = transformed_target[:,:1]
                    X = target.cpu().numpy()
                    X_hat = self.get_X_hat(transformed_Z,output).cpu().numpy()
                    
                    transformed=transformed.cpu().numpy()
                    original_sigma=original_sigma.cpu().numpy()
                    original_alpha=original_alpha.cpu().numpy()
                    min_t=min_t.cpu().numpy()
                    max_t=max_t.cpu().numpy()
                    X_hat =X_hat*(max_t-min_t)+min_t
                    X_hat=np.clip(inverse_gat(X_hat,original_sigma,original_alpha,0,method='closed_form'), 0, 1)
                
                inference_time = time.time()-start
                source = source.cpu().numpy()
                psnr_val = get_PSNR(X[0], source[0])
                ssim_val = get_SSIM(X[0], source[0],self.args.data_type)
                print(X[0].min(), X[0].max(), " and ", X_hat[0].min(),X_hat[0].max())
                print(f"image : {batch_idx:02d} ->\t psnr : {round(float(psnr_val),4):.4f}, ssim : {round(float(ssim_val),6):.6f} before denoise {self.args.loss_function}")
       
                psnr_val = get_PSNR(X[0], X_hat[0])
                
                ssim_val = get_SSIM(X[0], X_hat[0],self.args.data_type)
                psnr_arr.append(psnr_val)
                ssim_arr.append(ssim_val)
                time_arr.append(inference_time)
                noisy_img_arr.append(source)
                denoised_img_arr.append(X_hat[0].reshape(X_hat.shape[2],X_hat.shape[3]))
                print(f"image : {batch_idx:02d} ->\t psnr : {round(float(psnr_val),4):.4f}, ssim : {round(float(ssim_val),6):.6f} after  denoise {self.args.loss_function}")
                if self.args.test is True:
                    break
        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        mean_time = np.mean(time_arr)

            
        print ('PSNR : ', round(mean_psnr,4), '\tSSIM : ', round(mean_ssim,4))
        denoised_img_arr = np.array(denoised_img_arr)
        noisy_img_arr = np.array(noisy_img_arr)
        data_path = self.target_data_dir.split('/')[-1].split('.')[0]
        np.save(f'./result_data/only_denoised_{data_path}.npy',denoised_img_arr)
"""        
  



