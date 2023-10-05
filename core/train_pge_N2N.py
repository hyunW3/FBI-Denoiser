import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms.functional as tvF
from torchvision import transforms
import scipy.interpolate as sip
import wandb
from core.dataloader_230414 import RN2N_Dataset
from .utils import TedataLoader, TrdataLoader, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch, chen_estimate
from .loss_functions import mse_bias, mse_affine, emse_affine, mse_affine_with_tv
from .FMD.data_loader_RN2N import load_denoising_rn2n, fluore_to_tensor
from .FMD.data_loader import load_denoising_test_mix, load_denoising_n2n_train
# from .loss_functions import *
from .logger import Logger
import torchvision as vision
import sys
from .unet import est_UNet  
import time
import sys, os

def select_data_type(_tr_data_dir, _te_data_dir, x_avg_num, y_avg_num,args):
    if args.data_name == 'Samsung':
        tr_data_loader = RN2N_Dataset(data_path = _tr_data_dir, 
                                        x_avg_num = x_avg_num, y_avg_num = y_avg_num,
                                        return_img_info = False)
        tr_data_loader = DataLoader(tr_data_loader, batch_size=args.batch_size, shuffle=True, 
                                    num_workers=0, drop_last=True)
        if "230414" in _te_data_dir:
            te_data_loader =  RN2N_Dataset(data_path = _te_data_dir, 
                                            x_avg_num = x_avg_num, y_avg_num = y_avg_num,
                                            return_img_info = False)
        else :
            te_data_loader = TedataLoader(_te_data_dir, args)
        te_data_loader = DataLoader(te_data_loader, batch_size=1, shuffle=False,
                                    num_workers=0, drop_last=False)
    elif args.data_type == 'FMD' :
        data_types = None if args.data_name == 'ALL_FMD' else  args.data_name
        if args.x_f_num != args.y_f_num:
            tr_data_loader = load_denoising_rn2n(data_root=_tr_data_dir, train=True, types=data_types,
                                                    noise_levels=[1,2,4,8,16],
                                                    batch_size=args.batch_size,args=args)
        else :
            transform = None
            tr_data_loader = load_denoising_n2n_train(_tr_data_dir,
                batch_size=args.batch_size, noise_levels=[1,2,4,8,16], 
                types=None, transform=transform, target_transform=transform, 
                patch_size=args.crop_size)
        transform = transforms.Compose([
            transforms.FiveCrop(args.crop_size),
            transforms.Lambda(lambda crops: torch.stack([
                fluore_to_tensor(crop) for crop in crops[:4]])),
            transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
            ])
        te_data_loader = load_denoising_test_mix(_te_data_dir,
                                noise_levels=[int(args.x_f_num[1:])], 
                                transform=transform, patch_size=args.crop_size,
                                batch_size=2)
        
    else :
        raise NotImplementedError("Not implemented data type", args.data_type, args.data_name)
    return tr_data_loader, te_data_loader

class TrainN2N_PGE(object):
    def __init__(self,_tr_data_dir=None, _te_data_dir=None, _save_file_name = None, _args = None):
        
        self.tr_data_dir = _tr_data_dir
        self.te_data_dir = _te_data_dir
        self.args = _args
        self.x_avg_num = int(self.args.x_f_num[1:])
        self.y_avg_num = int(self.args.y_f_num[1:])
        self.save_file_name = _save_file_name
        
        # self.tr_data_loader = TrdataLoader(_tr_data_dir, self.args)
        # self.tr_data_loader = DataLoader(self.tr_data_loader, batch_size=self.args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        # self.te_data_loader = TedataLoader(_te_data_dir, self.args)
        # self.te_data_loader = DataLoader(self.te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.tr_data_loader, self.te_data_loader = select_data_type(_tr_data_dir, _te_data_dir, 
                                                                    self.x_avg_num, self.y_avg_num,
                                                                    self.args)
        
        self.result_tr_loss_arr = []

        self.logger = Logger(self.args.nepochs, len(self.tr_data_loader))
        self._compile()

    def _compile(self):
        
        self.model=est_UNet(2,depth=self.args.unet_layer)
        
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.drop_epoch, gamma=self.args.drop_rate)
        
        self.model = self.model.cuda()

       
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), './weights/'+self.save_file_name  + '.w')
        return
    
        
    def eval(self):
        """Evaluates denoiser on validation set."""        
        
        name_sequence = []
        if self.args.data_name == "Samsung":
            for i in range(1,11):
                set_num = f"SET{i:02d}_"
                if i < 5:
                    f_num_list = map(lambda x : f"{set_num}{x}",['F08', 'F16', 'F32'])
                else :
                    f_num_list = map(lambda x : f"{set_num}{x}",['F01', 'F02', 'F04', 'F08', 'F16', 'F32'])
                name_sequence += f_num_list
        loss_arr = []
        a_arr=[]
        b_arr=[]
        estimated_gaussian_noise_level_arr = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.te_data_loader):
                source = source.cuda()
                # target = target.cuda()
                # Denoise
                
                noise_hat = self.model(source)
  
                # target = target.cpu().numpy()
                # noise_hat = noise_hat.cpu().numpy()

                target = target.cpu().numpy()
                noise_hat = noise_hat.cpu()

                predict_alpha=torch.mean(noise_hat[:,0])
                predict_sigma=torch.mean(noise_hat[:,1])
                transformed_with_pge = gat(source,predict_sigma,predict_alpha,0)
                loss=self._vst(transformed_with_pge,self.args.vst_version)
                estimated_gaussian_noise_level = chen_estimate(transformed_with_pge)
                
                loss_arr.append(loss.item())
                a_arr.append(predict_alpha.item())
                b_arr.append(predict_sigma.item())
                estimated_gaussian_noise_level_arr.append(estimated_gaussian_noise_level.item())

                if batch_idx % 50 == 0 and self.args.log_off is False:
                    name_str = f""
                    if self.args.data_name == "Samsung":
                        name_str += f"{name_sequence[batch_idx // 50]}"

                    # self.args.logger.log({'test/source' : wandb.Image(source, 
                    #     caption=f'Batch : {batch_idx} Alpha: {a_arr[-1]:.4f}, Sigma: {b_arr[-1]:.8f}, \
                    #         Estimated Noise Level: {estimated_gaussian_noise_level_arr[-1]:.4f}')})
                if batch_idx % 50 == 49 and self.args.log_off is False:
                    self.args.logger.log({f"eval_{name_str}/loss" : np.mean(loss_arr[-50:])})
                    self.args.logger.log({f"eval_{name_str}/alpha" : np.mean(a_arr[-50:])})
                    self.args.logger.log({f"eval_{name_str}/sigma" : np.mean(b_arr[-50:])}) 
                    self.args.logger.log({f"eval_{name_str}/estimated_gaussian_noise_level" : np.mean(estimated_gaussian_noise_level_arr[-50:])})
        mean_te_loss = np.mean(loss_arr)
        return mean_te_loss, a_arr,b_arr ,estimated_gaussian_noise_level_arr
    
    def _on_epoch_end(self, epoch, mean_tr_loss):
        """Tracks and saves starts after each epoch."""
        self.save_model(epoch)
        mean_te_loss, a_arr, b_arr,estimated_gaussian_noise_level_arr = self.eval()
        self.result_tr_loss_arr.append(mean_tr_loss)
        # sio.savemat('./result_data/'+self.save_file_name +'_result',{'tr_loss_arr':self.result_tr_loss_arr,'a_arr':a_arr, 'b_arr':b_arr,
        #                                             'estimated_gaussian_noise_level_arr' : estimated_gaussian_noise_level_arr})
      
        self.args.logger.log({
                'train/mean_loss'   : mean_tr_loss,
                'test/mean_loss'    : mean_te_loss,
                'alpha'         : a_arr,
                'sigma'         : b_arr,
                'estimated_gaussian_noise_level' : estimated_gaussian_noise_level_arr
        })
    def _vst(self,transformed,version='MSE'):    
        
        est=chen_estimate(transformed)
        if version=='MSE':
            return ((est-1)**2)
        elif version =='MAE':
            return abs(est-1)
        else :
            raise ValueError("version error in _vst function of train_pge.py")
     
    def train(self):
        """Trains denoiser on training set."""
        num_batches = len(self.tr_data_loader)
        
        for epoch in range(self.args.nepochs):
            self.scheduler.step()
            tr_loss = []

            for batch_idx, (noisy1,noisy2, target) in enumerate(self.tr_data_loader):
                self.optim.zero_grad()
                source = noisy1.cuda()
                # target = target.cuda()
                
                noise_hat=self.model(source)
                
                predict_alpha=torch.mean(noise_hat[:,0])
                predict_sigma=torch.mean(noise_hat[:,1])
                #print(torch.unique(source),"\n",torch.unique(target))
                predict_gat=gat(source,predict_sigma,predict_alpha,0)     
#                 predict_gat=gat(source,torch.tensor(0.02).to(torch.float32),torch.tensor(0.01).to(torch.float32),0)     
                
                loss=self._vst(predict_gat,self.args.vst_version)
                #print(predict_gat,"loss : ",loss)
                loss.backward()
                self.optim.step()  

                self.logger.log(losses = {'loss': loss, 'pred_alpha': predict_alpha, 'pred_sigma': predict_sigma}, lr = self.optim.param_groups[0]['lr'])
                tr_loss.append(loss.detach().cpu().numpy())
                if self.args.test is True:
                    break
                
            mean_tr_loss = np.mean(tr_loss)
            self._on_epoch_end(epoch+1, mean_tr_loss)    
            



