import torch
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio

from .utils import TedataLoader, TrdataLoader, get_PSNR, get_SSIM
from .loss_functions import mse_bias, mse_affine, estimated_bias, estimated_affine
from .logger import Logger
from .models import New_model, New_model_ablation

import time

torch.backends.cudnn.benchmark=True

class Train(object):
    def __init__(self,_tr_data_dir=None, _te_data_dir=None, _save_file_name = None, _args = None):
        
        self.args = _args
        
        self.tr_data_loader = TrdataLoader(_tr_data_dir, self.args)
        self.tr_data_loader = DataLoader(self.tr_data_loader, batch_size=self.args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        self.te_data_loader = TedataLoader(_te_data_dir, self.args)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.result_psnr_arr = []
        self.result_ssim_arr = []
        self.result_time_arr = []
        self.result_denoised_img_arr = []
        self.result_te_loss_arr = []
        self.result_tr_loss_arr = []
        self.best_psnr = 0
        self.save_file_name = _save_file_name

        self.logger = Logger(self.args.nepochs, len(self.tr_data_loader))
        
        if self.args.loss_function== 'MSE':
            self.loss = torch.nn.MSELoss()
            num_output_channel = 1
        elif self.args.loss_function == 'N2V':
            self.loss = mse_bias
            num_output_channel = 1
        elif self.args.loss_function == 'affine':
            self.loss = mse_affine
            num_output_channel = 2
        
        if self.args.model_type == 'case1':
            self.model = New_model_ablation(channel = 1, output_channel = num_output_channel, filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = 'case1')
        elif self.args.model_type == 'case2':
            self.model = New_model_ablation(channel = 1, output_channel = num_output_channel, filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = 'case2')
        else:
            self.model = New_model(channel = 1, output_channel =  num_output_channel, filters = self.args.num_filters, num_of_layers=self.args.num_layers)
            
        self.model = self.model.cuda()
            
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.drop_epoch, gamma=self.args.drop_rate)
        
    def save_model(self):
        
        torch.save(self.model.state_dict(), './weights/'+self.save_file_name + '.w')
        
        return
    
    def get_X_hat(self, Z, output):

        X_hat = output[:,:self.channel] * Z + output[:,self.channel:]
            
        return X_hat
        
    def eval(self):
        """Evaluates denoiser on validation set."""

        psnr_arr = []
        ssim_arr = []
        loss_arr = []
        time_arr = []
        denoised_img_arr = []

        with torch.no_grad():

            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                start = time.time()

                source = source.cuda()
                target = target.cuda()

                # Denoise
                output = self.model(source)

                # Update loss
                if self.args.loss_function == 'MSE':
                    loss = self.loss(output, target)
                    
                    loss = loss.cpu().numpy()
                    output = output.cpu().numpy()

                    X_hat = np.clip(output, 0, 1)
                    X = target.cpu().numpy()

                inference_time = time.time()-start

                loss_arr.append(loss)
                psnr_arr.append(get_PSNR(X[0], X_hat[0]))
                ssim_arr.append(get_SSIM(X[0], X_hat[0]))
                time_arr.append(inference_time)
                denoised_img_arr.append(X_hat[0].reshape(X_hat.shape[2],X_hat.shape[3]))

        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        mean_time = np.mean(time_arr)

        if self.best_psnr <= mean_psnr:
            self.best_psnr = mean_psnr
            self.result_denoised_img_arr = denoised_img_arr.copy()
            
            
        return mean_loss, mean_psnr, mean_ssim, mean_time
    
    def _on_epoch_end(self, epoch, mean_tr_loss):
        """Tracks and saves starts after each epoch."""
        
        mean_te_loss, mean_psnr, mean_ssim, mean_time = self.eval()

        self.result_psnr_arr.append(mean_psnr)
        self.result_ssim_arr.append(mean_ssim)
        self.result_time_arr.append(mean_time)
        self.result_te_loss_arr.append(mean_te_loss)
        self.result_tr_loss_arr.append(mean_tr_loss)

        sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr,'time_arr':self.result_time_arr, 'denoised_img':self.result_denoised_img_arr})

#         sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr})

        print ('Epoch : ', epoch, ' Tr loss : ', round(mean_tr_loss,4), ' Te loss : ', round(mean_te_loss,4), ' PSNR : ', round(mean_psnr,2), ' SSIM : ', round(mean_ssim,4),' Best PSNR : ', round(self.best_psnr,4)) 
            
  
    def train(self):
        """Trains denoiser on training set."""
        
        for epoch in range(self.args.nepochs):

            
            tr_loss = []

            for batch_idx, (source, target) in enumerate(self.tr_data_loader):

                self.optim.zero_grad()

                source = source.cuda()
                target = target.cuda()

                # Denoise image
                source_denoised = self.model(source)
                
                if self.args.loss_function == 'MSE':
                    loss = self.loss(source_denoised, target)
                    
                loss.backward()
                self.optim.step()
                
                self.logger.log(losses = {'loss': loss}, lr = self.optim.param_groups[0]['lr'])

                tr_loss.append(loss.detach().cpu().numpy())

            mean_tr_loss = np.mean(tr_loss)
            self._on_epoch_end(epoch+1, mean_tr_loss)    
            if self.args.nepochs == epoch +1:
                self.save_model()
                
            self.scheduler.step()


