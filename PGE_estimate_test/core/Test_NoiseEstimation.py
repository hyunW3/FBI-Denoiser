import sys,os
sys.path.append(os.path.dirname(os.path.abspath("../core")))
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import scipy.io as sio

from .utils import TedataLoader, chen_estimate, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
from .unet import est_UNet

import time

torch.backends.cudnn.benchmark=True

class Test_NoiseEstimation(object):
    def __init__(self,_te_data_dir=None,_pge_weight_dir=None,_save_file_name = None, _args = None):
        
        self.args = _args
        self.te_data_loader = TedataLoader(_te_data_dir, self.args)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.result_alpha_arr = []
        self.result_beta_arr = []
        self.result_time_arr = []
        self.save_file_name = _save_file_name

        ## load PGE model
        
        num_output_channel = 2
        if self.args.PGE_ON is True:
            self.pge_model=est_UNet(num_output_channel,depth=3)
            self.pge_model.load_state_dict(torch.load(_pge_weight_dir))
            self.pge_model.cuda()
        
    def eval(self):
        """Evaluates denoiser on validation set."""
        print(f"data name : {self.args.data_name}")
        alpha_arr = []
        beta_arr = []
        time_arr = []
        estimated_gaussian_noise_level_arr = []
        est_param = pd.read_csv(self.args.param_dir).to_numpy()
        # print(est_param[0][1], type(est_param[0]))
        print(f"With PGE? : {self.args.PGE_ON}")
        with torch.no_grad():

            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                start = time.time()

                source = source.cuda()
                target = target.cuda()
                
                # Denoise
                
                if self.args.PGE_ON is True:   
                    est_param=self.pge_model(source)
                    original_alpha=torch.mean(est_param[:,0])
                    original_beta=torch.mean(est_param[:,1])
                else :
                    original_alpha = est_param[batch_idx][0]
                    original_beta = est_param[batch_idx][1]

                
                inference_time = time.time()-start
                is_alpha_zero = "not zero" if original_alpha != 0 else "zero"
                print(f"image : {batch_idx:04d} ->\t alpha :  {original_alpha:.6f} {is_alpha_zero},  sigma : {original_beta:.6f} ")
                transformed_with_pge = gat(source,original_beta,original_alpha,0)
                estimated_gaussian_noise_level = chen_estimate(transformed_with_pge)
                print(f"estimated gaussian_noise  : {estimated_gaussian_noise_level:.6f}")
                transformed_with_pge, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed_with_pge)
                
                original_beta=original_beta#.cpu().numpy()
                original_alpha=original_alpha#.cpu().numpy()
                if self.args.PGE_ON is True :
                    original_beta=original_beta.cpu().numpy()
                    original_alpha=original_alpha.cpu().numpy()

                estimated_gaussian_noise_level_arr.append(estimated_gaussian_noise_level.cpu().numpy())
                alpha_arr.append(original_alpha)
                beta_arr.append(original_beta)
                time_arr.append(inference_time)
        mean_alpha = np.mean(alpha_arr)
        mean_beta = np.mean(beta_arr)
        mean_time = np.mean(time_arr)
        alpha_sigma = np.stack((alpha_arr,beta_arr),axis=1)
        os.chdir("./PGE_estimate_test")
        np.save(f'./estimated_gaussian_noise_level_arr_{self.save_file_name}.npy',estimated_gaussian_noise_level_arr)
        print(os.getcwd())
        np.savetxt(f'./output_log/${self.save_file_name}_alpha_sigma.txt',alpha_sigma,delimiter='\t')
        print ('Mean(alpha) : ', round(mean_alpha,4), 'Mean(beta) : ', round(mean_beta,6))
        f = open("./result.txt",'a')
        summary_output = f"Foi {self.args.PGE_ON} {self.args.data_name} {self.args.dataset_type} Mean estimated_gaussian_noise_level : {np.mean(estimated_gaussian_noise_level_arr)}"
        print(summary_output,file=f)
        f.close()
        print(summary_output)
        return 
  


