import os,cv2
from random import sample
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import wandb
import numpy as np
import scipy.io as sio
from datetime import date

from core.dataloader_230414 import RN2N_Dataset
# from no_bsn.utils import get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
# from no_bsn.loss_functions import mse_bias, mse_affine, emse_affine, mse_affine_with_tv
# from no_bsn.logger import Logger
# from no_bsn.models import New_model
# from no_bsn.fcaide import FC_AIDE
# from no_bsn.dbsn import DBSN_Model
# from no_bsn.unet import est_UNet
# from no_bsn.FMD.data_loader_RN2N import *
# from no_bsn.FMD.data_loader import *


from .utils import TedataLoader, TrdataLoader, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
from .loss_functions import mse_bias, mse_affine, emse_affine, mse_affine_with_tv
from .FMD.data_loader_RN2N import load_denoising_rn2n, fluore_to_tensor
from .FMD.data_loader import load_denoising_test_mix, load_denoising_n2n_train
from .logger import Logger
from .models import New_model
from .fcaide import FC_AIDE
from .dbsn import DBSN_Model
from .unet import est_UNet
from .model_UNet import UNet
from .model_NAFNet import NAFNet
torch.multiprocessing.set_start_method('spawn')


import time

torch.backends.cudnn.benchmark=True
torch.multiprocessing.set_sharing_strategy('file_system')
def select_model_type(args,pge_weight_dir):
    num_output_channel = 1
    pge_model = None
    if args.loss_function== 'MSE': # upper bound 1-1(optional)
        loss = torch.nn.MSELoss()
        num_output_channel = 1
        args.output_type = "linear"
    elif args.loss_function == 'N2V': #lower bound
        loss = mse_bias
        num_output_channel = 1
        args.output_type = "linear"
    elif args.loss_function == 'MSE_Affine': # 1st(our case upper bound)
        loss = mse_affine
        num_output_channel = 2
        args.output_type = "linear"
    elif args.loss_function == 'EMSE_Affine':
        
        loss = emse_affine
        num_output_channel = 2
        if args.with_originalPGparam is False:
            ## load PGE model
            pge_model=est_UNet(num_output_channel,depth=3)
            pge_model.load_state_dict(torch.load(pge_weight_dir))
            pge_model = pge_model.cuda()
            for param in pge_model.parameters():
                param.requires_grad = False
        
        args.output_type = "sigmoid"
    elif args.loss_function == 'MSE_Affine_with_tv':
        loss = mse_affine_with_tv
        num_output_channel = 2
        args.output_type = "linear"

    if args.model_type == 'FC-AIDE':
        model = FC_AIDE(channel = 1, output_channel = num_output_channel, filters = 64, num_of_layers=10, output_type = args.output_type, sigmoid_value = args.sigmoid_value)
    elif args.model_type == 'DBSN':
        model = DBSN_Model(in_ch = 1,
                        out_ch = num_output_channel,
                        mid_ch = 96,
                        blindspot_conv_type = 'Mask',
                        blindspot_conv_bias = True,
                        br1_block_num = 8,
                        br1_blindspot_conv_ks =3,
                        br2_block_num = 8,
                        br2_blindspot_conv_ks = 5,
                        activate_fun = 'Relu')
    elif args.model_type == 'UNet':
            model = UNet(dim = 1, output_channel = num_output_channel, ngf_factor = 12, depth = 4)
    elif args.model_type == 'NAFNet':
        model =  NAFNet(img_channel=1,output_channel= num_output_channel, width=32, middle_blk_num=12,
                    enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2])
    elif args.model_type == 'NAFNet_light':
        model =  NAFNet(img_channel=1,output_channel= num_output_channel,width=8, middle_blk_num=4,
                    enc_blk_nums=[2,2,4,4], dec_blk_nums=[2,2,2,2]) # 809257
    else:
        model = New_model(channel = 1, output_channel =  num_output_channel, 
                            filters = args.num_filters, num_of_layers=args.num_layers, case = args.model_type, 
                            # BSN_type = {"type" : args.BSN_type, "param" : args.BSN_param},
                            output_type = args.output_type, sigmoid_value = args.sigmoid_value)
    model = model.cuda()
    return model, pge_model, loss, args
def select_data_type(_tr_data_dir, _te_data_dir, x_avg_num, y_avg_num,args):
    if args.data_name == 'Samsung':
        tr_data_loader = RN2N_Dataset(data_path = _tr_data_dir, 
                                        x_avg_num = x_avg_num, y_avg_num = y_avg_num,
                                        return_img_info = False, average_mode=args.average_mode)
        tr_data_loader = DataLoader(tr_data_loader, batch_size=args.batch_size, shuffle=True, 
                                    num_workers=0, drop_last=True)
        if "230414" in _te_data_dir:
            te_data_loader =  RN2N_Dataset(data_path = _te_data_dir, 
                                            x_avg_num = x_avg_num, y_avg_num = y_avg_num,
                                            return_img_info = True)
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
class TrainN2N_FBI(object):
    def __init__(self,_tr_data_dir=None, _te_data_dir=None, _save_file_name = None, _args = None):
        
        self.args = _args
        self.x_avg_num = int(self.args.x_f_num[1:])
        self.y_avg_num = int(self.args.y_f_num[1:])
        print("x_avg_num, y_avg_num",self.x_avg_num, self.y_avg_num)
        self.pge_weight_dir = None
        if self.args.pge_weight_dir != None:
            self.pge_weight_dir = './weights/' + self.args.pge_weight_dir
        self.tr_data_loader, self.te_data_loader = select_data_type(_tr_data_dir, _te_data_dir, 
                                                                    self.x_avg_num, self.y_avg_num,
                                                                    self.args)
        self.original_test_set = False if "230414" in _te_data_dir else True
        print(len(self.tr_data_loader), len(self.te_data_loader))
        # if self.args.test is True:
        #     sys.exit(0)

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
        self.model, self.pge_model, self.loss, self.args = select_model_type(self.args, self.pge_weight_dir)
            
        if self.args.log_off is False :
            self.args.logger.watch(self.model)
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        # if self.args.test is True:
        #     sys.exit(0)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.drop_epoch, gamma=self.args.drop_rate)
        
    def save_model(self):
        
        torch.save(self.model.state_dict(), './weights/'+self.save_file_name + '.w')
        
        return
    
    def get_X_hat(self, Z, output):

        X_hat = output[:,:1] * Z + output[:,1:]
            
        return X_hat
    def update_log(self,epoch,output):
        # args = self.args
        mean_tr_loss, mean_te_loss, mean_psnr, mean_ssim = output
        self.args.logger.log({
                'train/mean_loss'   : mean_tr_loss,
                'test/mean_loss'    : mean_te_loss,
                'mean_psnr'         : mean_psnr,
                'mean_ssim'         : mean_ssim,
        },step=epoch)
        

    def eval(self,epoch):
        """Evaluates denoiser on validation set."""
        print(f"\n===={epoch}th epoch start evaluation ====")
        psnr_arr = []
        ssim_arr = []
        loss_arr = []
        time_arr = []
        denoised_img_arr = []
        clean_img_arr = []
        noisy_img_arr = []
        
        if self.args.BSN_type == 'prob-BSN' and self.args.prob_BSN_test_mode is True:
            self.model.eval()
        sample_pic_len = min(50,len(self.te_data_loader)//5)
        with torch.no_grad():
            for batch_idx, (data) in enumerate(self.te_data_loader):
                if self.args.data_name == 'Samsung':
                    if self.original_test_set is False:
                        img_info, noisy1,noisy2, clean = data
                    else :
                        noisy1, clean = data
                        img_info = ["SET05","SET06","SET07","SET08","SET09","SET10"]
                else :
                    # noisy1,noisy2, clean = data
                    noisy1, clean = data
                    # since transform is four crop
                    noisy1 = noisy1.view(-1, *noisy1.shape[2:]) + 0.5 # -0.5 ~ 0.5 to 0 ~ 1
                    clean = clean.view(-1, *clean.shape[2:]) + 0.5 
                    if self.args.test is True:
                        print("noisy1 range",torch.min(noisy1),torch.max(noisy1))
                        print("clean range",torch.min(clean),torch.max(clean))
                    
                start = time.time()

                source = noisy1.cuda()
                target = clean.cuda()
                if self.original_test_set is False:
                    if self.args.loss_function[:10] == 'MSE_Affine' or self.args.loss_function == 'N2V':
                        target = torch.cat([source,target], dim = 1)
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
                    # print(target.shape, X.shape)
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

                # inference_time = time.time()-start
                # print(X.shape,X_hat.shape)
                loss_arr.append(loss)
                psnr_arr.append(get_PSNR(X[0], X_hat[0]))
                ssim_arr.append(get_SSIM(X[0], X_hat[0]))
                denoised_img_arr.append(X_hat[0].reshape(X_hat.shape[2],X_hat.shape[3]))
                clean_img_arr.append(clean[0].cpu().numpy())
                noisy_img_arr.append(noisy1[0].detach().cpu().numpy())
                # if self.args.test is True and batch_idx % sample_pic_len == sample_pic_len-1 :
                #     denoised_img = torch.from_numpy(np.array(denoised_img_arr[-sample_pic_len:]).reshape(sample_pic_len,1,X_hat.shape[2],X_hat.shape[3]))
                    
                #     clean_test = torch.from_numpy(np.array(clean_img_arr[-sample_pic_len:]))
                #     print(denoised_img.shape,clean_test.shape)
                    
                    
                #     denoised_grid = torchvision.utils.make_grid(denoised_img)
                #     print(denoised_grid.shape)
                #     clean_grid = torchvision.utils.make_grid(torch.from_numpy(np.array(clean_img_arr[-sample_pic_len:])))
                #     noisy_grid = torchvision.utils.make_grid(torch.from_numpy(np.array(noisy_img_arr[-sample_pic_len:])))
                    
                #     denoised_img = wandb.Image(denoised_grid, caption = f'EPOCH : {epoch} Batch : {batch_idx}\nPSNR : {np.mean(psnr_arr[-sample_pic_len:]):.4f}, SSIM : {np.mean(ssim_arr[-sample_pic_len:]):.4f}')
                #     break
                if self.args.log_off is False and batch_idx == sample_pic_len-1:
                    img_len = min(16,sample_pic_len )
                    denoised_img = torch.from_numpy(np.array(denoised_img_arr[-sample_pic_len:]).reshape(sample_pic_len,1,X_hat.shape[2],X_hat.shape[3]))
                    denoised_grid = torchvision.utils.make_grid(denoised_img[:img_len])
                    clean_grid = torchvision.utils.make_grid(torch.from_numpy(np.array(clean_img_arr)[:img_len]))
                    noisy_grid = torchvision.utils.make_grid(torch.from_numpy(np.array(noisy_img_arr)[:img_len]))
                    
                    name_str = f"{batch_idx}th batch"
                    if self.args.data_name == "Samsung":
                        if self.original_test_set is False:
                            name_str += f"{img_info['img_name'][0]}_{img_info['img_idx'].item()}th_{img_info['pair_idx'].item()}th_pair"
                        else :
                            name_str = f"{img_info[batch_idx//sample_pic_len]}"
                
                    denoised_img = wandb.Image(denoised_grid, caption = f'EPOCH : {epoch} {name_str}\n\
                        PSNR : {np.mean(psnr_arr[-sample_pic_len:]):.4f}, SSIM : {np.mean(ssim_arr[-sample_pic_len:]):.4f}')
                    clean_img = wandb.Image(clean_grid, caption = f'EPOCH : {epoch} {name_str}')
                    noisy_img = wandb.Image(noisy_grid, caption = f'EPOCH : {epoch} {name_str}')
                    self.args.logger.log({f"eval/denoised_img" : denoised_img,
                                            f"eval/clean_img" : clean_img,
                                            f"eval/noisy_img" : noisy_img},step=epoch)
                        
                    # self.args.logger.log({f"eval/inference_time" : np.mean(time_arr[-100:])})
                # time_arr.append(inference_time)

        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        # mean_time = np.mean(time_arr)
        if self.args.log_off is False:
            # min_psnr,   max_psnr = np.min(psnr_arr), np.max(psnr_arr)
            # min_ssim, max_ssim = np.min(ssim_arr), np.max(ssim_arr)
            
            self.args.logger.log({f"eval/loss" : mean_loss},step=epoch)
            self.args.logger.log({f"eval/psnr" : mean_psnr},step=epoch)
            self.args.logger.log({f"eval/ssim" : mean_ssim},step=epoch)
            # self.args.logger.log({f"eval/min_psnr" : min_psnr, f"eval/max_psnr" : max_psnr},step=epoch)
            # self.args.logger.log({f"eval/min_ssim" : min_ssim, f"eval/max_ssim" : max_ssim},step=epoch)
            
            # self.args.logger.log({f"eval/inference_time" : mean_time})
        if self.best_psnr <= mean_psnr:
            self.best_psnr = mean_psnr
            self.result_denoised_img_arr = denoised_img_arr.copy()
        
            
        return mean_loss, mean_psnr, mean_ssim
    
    def _on_epoch_end(self, epoch, mean_tr_loss):
        """Tracks and saves starts after each epoch."""
        
        mean_te_loss, mean_psnr, mean_ssim = self.eval(epoch)

        self.result_psnr_arr.append(mean_psnr)
        self.result_ssim_arr.append(mean_ssim)
        # self.result_time_arr.append(mean_time)
        self.result_te_loss_arr.append(mean_te_loss)
        self.result_tr_loss_arr.append(mean_tr_loss)
        if self.args.test is False:
            os.makedirs("./result_data",exist_ok=True)
            sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr, 'denoised_img':self.result_denoised_img_arr}) # 'time_arr':self.result_time_arr,

#         sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr})

        print ('Epoch : ', epoch, ' Tr loss : ', round(mean_tr_loss,4), ' Te loss : ', round(mean_te_loss,4),
             ' PSNR : ', round(mean_psnr,4), ' SSIM : ', round(mean_ssim,4),' Best PSNR : ', round(self.best_psnr,4)) 
        
        
        if self.args.test is True:
            sys.exit(0)
        if self.args.log_off is False:
            self.update_log(epoch,[mean_tr_loss, mean_te_loss, mean_psnr, mean_ssim])
  
    def train(self):
        """Trains denoiser on training set."""
        if self.args.BSN_type == 'prob-BSN':
            self.model.train()
        for epoch in range(self.args.nepochs):

            tr_loss = []
                
            for batch_idx, (noisy1,noisy2, clean) in enumerate(self.tr_data_loader):

                self.optim.zero_grad()
                
                source = noisy1.cuda()
                target = noisy2.cuda()
                if self.args.loss_function[:10] == 'MSE_Affine' or self.args.loss_function == 'N2V' or self.args.loss_function == 'Noise_est' or self.args.loss_function == 'EMSE_Affine':
                    target = torch.cat([source,target], dim = 1) # (512,256) -> (2,256,256)
                    
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
                    # print(output.shape,target.shape)
                    loss = self.loss(output, target)
                    
                loss.backward()
                self.optim.step()
                
                self.logger.log(losses = {'loss': loss}, lr = self.optim.param_groups[0]['lr'])

                tr_loss.append(loss.detach().cpu().numpy())
                if batch_idx > 10 and self.args.test is True:
                    print("break since it is test mode")
                    break
            mean_tr_loss = np.mean(tr_loss)
            
            self._on_epoch_end(epoch+1, mean_tr_loss)  
            if self.args.test is True:
                print("test end")
                sys.exit(0)  
            if self.args.nepochs == epoch +1:
                self.save_model()
                
            self.scheduler.step()




