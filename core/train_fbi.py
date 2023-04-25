import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import scipy.io as sio
from datetime import date
from .utils import TedataLoader, TrdataLoader, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
from .loss_functions import mse_bias, mse_affine, emse_affine, mse_affine_with_tv
from .logger import Logger
from .models import New_model
from .fcaide import FC_AIDE
from .dbsn import DBSN_Model
from .unet import est_UNet

import time

torch.backends.cudnn.benchmark=True

class Train_FBI(object):
    def __init__(self,_tr_data_dir=None, _te_data_dir=None, _save_file_name = None, _args = None):
        
        self.args = _args
        
        if self.args.pge_weight_dir != None:
            self.pge_weight_dir = './weights/' + self.args.pge_weight_dir
        
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
        self.date = date.today().isoformat()
        self.logger = Logger(self.args.nepochs, len(self.tr_data_loader))
        if "Samsung" in self.args.data_name:
            log_folder = "./samsung_log"
        else :
            log_folder = "./log"
        os.makedirs(log_folder,exist_ok=True)
        self.writer = SummaryWriter(log_folder)

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
            if self.args.with_originalPGparam is False:
                ## load PGE model
                self.pge_model=est_UNet(num_output_channel,depth=3)
                self.pge_model.load_state_dict(torch.load(self.pge_weight_dir))
                self.pge_model=self.pge_model.cuda()
            
                for param in self.pge_model.parameters():
                    param.requires_grad = False
            
            self.args.output_type = "sigmoid"
        elif self.args.loss_function == 'MSE_Affine_with_tv':
            self.loss = mse_affine_with_tv
            num_output_channel = 2
            self.args.output_type = "linear"

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
            self.model = New_model(channel = 1, output_channel =  num_output_channel, 
                                filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = self.args.model_type, 
                                BSN_type = {"type" : self.args.BSN_type, "param" : self.args.BSN_param},
                                output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value)
            
        self.model = self.model.cuda()
        if self.args.log_off is False :
            self.args.logger.watch(self.model)
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.drop_epoch, gamma=self.args.drop_rate)
        
    def save_model(self):
        
        torch.save(self.model.state_dict(), './weights/'+self.save_file_name + '.w')
        
        return
    
    def get_X_hat(self, Z, output):

        X_hat = output[:,:1] * Z + output[:,1:]
            
        #print(X_hat[0])
        return X_hat
    def update_log(self,epoch,output):
        # args = self.args
        mean_tr_loss, mean_te_loss, mean_psnr, mean_ssim = output
        self.args.logger.log({
                'train/mean_loss'   : mean_tr_loss,
                'test/mean_loss'    : mean_te_loss,
                'mean_psnr'         : mean_psnr,
                'mean_ssim'         : mean_ssim,
        })
        

    def eval(self,epoch):
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
        psnr_arr = []
        ssim_arr = []
        loss_arr = []
        time_arr = []
        denoised_img_arr = []
        # # next prob-BSN
        if self.args.BSN_type == 'prob-BSN' and self.args.prob_BSN_test_mode is True:
            self.model.eval()
        with torch.no_grad():

            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                start = time.time()

                source = source.cuda()
                target = target.cuda()
                
                # Denoise
                if self.args.loss_function =='EMSE_Affine':
                    if self.args.with_originalPGparam is True:
                        original_alpha=torch.tensor(self.args.alpha).unsqueeze(0).cuda()
                        original_sigma=torch.tensor(self.args.beta).unsqueeze(0).cuda()
                    else :
                        est_param=self.pge_model(source)
                        original_alpha=torch.mean(est_param[:,0])
                        original_sigma=torch.mean(est_param[:,1])
                    if self.args.test is True:
                        print('original_alpha : ',original_alpha)
                        print('original_sigma : ',original_sigma)
                    transformed=gat(source,original_sigma,original_alpha,0)
                    transformed, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed)
                    
                    transformed_target = torch.cat([transformed, transformed_sigma], dim = 1)
#                     target = torch.cat([target,transformed_sigma], dim = 1)

                    output = self.model(transformed)
    
                    loss = self.loss(output, transformed_target)
        
                else:
                    # Denoise image
                    if self.args.model_type == 'DBSN':
                        output, _ = self.model(source)
                        loss = self.loss(output, target)
                    else:
                        output = self.model(source)
                        if self.args.loss_function == 'MSE_Affine_with_tv':
                            loss = self.loss(output, target, self.args.lambda_val)
                        else :
                            loss = self.loss(output, target)
                
                loss = loss.cpu().numpy()
                
                # Update loss
                if self.args.loss_function == 'MSE':
                    output = output.cpu().numpy()
                    X_hat = np.clip(output, 0, 1)
                    X = target.cpu().numpy()
                    
                elif self.args.loss_function[:10] == 'MSE_Affine':
                    
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
                #    print (X_hat.shape)
                #    print (X.shape)

                inference_time = time.time()-start
                
                loss_arr.append(loss)
                psnr_arr.append(get_PSNR(X[0], X_hat[0]))
                ssim_arr.append(get_SSIM(X[0], X_hat[0],self.args.data_type))
                denoised_img_arr.append(X_hat[0].reshape(X_hat.shape[2],X_hat.shape[3]))
                if batch_idx % 50 == 0 and self.args.log_off is False:
                    name_str = f""
                    if self.args.data_name == "Samsung":
                        name_str += f"{name_sequence[batch_idx // 50]}"
                    image = wandb.Image(denoised_img_arr[-1], caption = f'EPOCH : {epoch} Batch : {batch_idx}\nPSNR: {psnr_arr[-1]:.4f}, SSIM: {ssim_arr[-1]:.4f}')
                    self.args.logger.log({f"eval/denoised_img_{name_str}" : image})
                if batch_idx % 50 == 49 and self.args.log_off is False:
                    self.args.logger.log({f"eval_{name_str}/loss" : np.mean(loss_arr[-50:])})
                    self.args.logger.log({f"eval_{name_str}/psnr" : np.mean(psnr_arr[-50:])})
                    self.args.logger.log({f"eval_{name_str}/ssim" : np.mean(ssim_arr[-50:])}) 
                    # self.args.logger.log({f"eval/inference_time" : np.mean(time_arr[-100:])})
                time_arr.append(inference_time)

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
        
        mean_te_loss, mean_psnr, mean_ssim, mean_time = self.eval(epoch)

        self.result_psnr_arr.append(mean_psnr)
        self.result_ssim_arr.append(mean_ssim)
        self.result_time_arr.append(mean_time)
        self.result_te_loss_arr.append(mean_te_loss)
        self.result_tr_loss_arr.append(mean_tr_loss)
        # if self.args.log_off is False:
        sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr,'time_arr':self.result_time_arr, 'denoised_img':self.result_denoised_img_arr})

#         sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr})

        print ('Epoch : ', epoch, ' Tr loss : ', round(mean_tr_loss,4), ' Te loss : ', round(mean_te_loss,4),
             ' PSNR : ', round(mean_psnr,4), ' SSIM : ', round(mean_ssim,4),' Best PSNR : ', round(self.best_psnr,4)) 
        #if self.args.test is False:
        if self.args.log_off is False:
            self.update_log(epoch,[mean_tr_loss, mean_te_loss, mean_psnr, mean_ssim])
  
    def train(self):
        """Trains denoiser on training set."""
        if self.args.BSN_type == 'prob-BSN':
            self.model.train()
        for epoch in range(self.args.nepochs):

            tr_loss = []

            for batch_idx, (source, target) in enumerate(self.tr_data_loader):

                self.optim.zero_grad()

                source = source.cuda()
                target = target.cuda()
                #print(source.shape, target.shape) # torch.Size([1, 256, 256]) torch.Size([1,2 ,256, 256])
                # Denoise
                if self.args.loss_function =='EMSE_Affine':
                    if self.args.with_originalPGparam is True:
                        original_alpha = self.args.alpha
                        original_sigma = self.args.beta
                    else :
                        est_param=self.pge_model(source)
                        original_alpha=torch.mean(est_param[:,0])
                        original_sigma=torch.mean(est_param[:,1])
                    if self.args.test is True:
                        print('original_alpha : ',original_alpha)
                        print('original_sigma : ',original_sigma)
                        
                    
                    transformed=gat(source,original_sigma,original_alpha,0)
                    transformed, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed)
                    
                    target = torch.cat([transformed, transformed_sigma], dim = 1)
#                     target = torch.cat([target,transformed_sigma], dim = 1)

                    output = self.model(transformed)
                else:
                    # Denoise image
                    if self.args.model_type == 'DBSN':
                        output, _ = self.model(source)
                    else:
                        output = self.model(source)
                    
                if self.args.loss_function == 'MSE_Affine_with_tv':
                    loss = self.loss(output, target, self.args.lambda_val)
                else :
                    loss = self.loss(output, target)
                    
                loss.backward()
                self.optim.step()
                
                self.logger.log(losses = {'loss': loss}, lr = self.optim.param_groups[0]['lr'])

                tr_loss.append(loss.detach().cpu().numpy())
                if self.args.test is True:
                    break
            mean_tr_loss = np.mean(tr_loss)
            
            self._on_epoch_end(epoch+1, mean_tr_loss)    
            if self.args.nepochs == epoch +1:
                self.save_model()
                
            self.scheduler.step()
        self.writer.close()




