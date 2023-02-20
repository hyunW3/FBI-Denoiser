from core.test_fbi import Test_FBI
from arguments import get_args
import torch
import numpy as np
import os, sys
import random
from neptune_setting import init_neptune

args = get_args()

# control the randomness
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':
    """Trains Noise2Noise."""
    save_file_name = ""
    te_data_dir = ""
    fbi_weight_dir = ""
    pge_weight_dir = ""
    if args.noise_type == 'Poisson-Gaussian':
        
        if args.data_type == 'RawRGB' and args.data_name == 'fivek' and args.alpha == 0 and args.beta == 0:
            
            te_data_dir = './data/test_fivek_rawRGB_random_noise.hdf5'
            fbi_weight_dir = './weights/211127_FBI_Net_RawRGB_random_noise_layers_x17_filters_x64_cropsize_220.w'
            pge_weight_dir = './weights/211127_PGE_Net_RawRGB_random_noise_cropsize_200.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_' + 'random_noise'
        
        elif args.data_type == 'RawRGB' and args.data_name == 'fivek' and args.alpha != 0 and args.beta != 0:
            
            # te_data_dir = './data/test_fivek_rawRGB_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            te_data_dir = f'./PGE_estimate_test_fivek/test_fivek_rawRGB_alpha_{args.alpha}_beta_{args.beta}_20sampling20images.hdf5'
            fbi_weight_dir = './weights/211127_FBI_Net_RawRGB_fivek_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'_layers_x17_filters_x64_cropsize_220.w'
            pge_weight_dir = './weights/211127_PGE_Net_RawRGB_fivek_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'_cropsize_200.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta)
        
        elif args.data_type == 'RawRGB' and args.data_name == 'SIDD':
            
            te_data_dir = './data/test_SIDD.mat'
            fbi_weight_dir = './weights/FBI_Denoiser_SIDD.w'
            pge_weight_dir = './weights/PGE_Net_SIDD.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'RawRGB' and args.data_name == 'DND':
            
            te_data_dir = './data/test_DND.mat'
            fbi_weight_dir = './weights/FBI_Denoiser_DND.w'
            pge_weight_dir = './weights/PGE_Net_DND.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
        
        elif args.data_type == 'FMD' and args.data_name == 'CF_FISH':
            
            te_data_dir = './data/test_CF_FISH.mat'
            fbi_weight_dir = './weights/FBI_Denoiser_CF_FISH.w'
            pge_weight_dir = './weights/PGE_Net_CF_FISH.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'FMD' and args.data_name == 'CF_MICE':
            
            te_data_dir = './data/test_CF_MICE.mat'
            fbi_weight_dir = './weights/FBI_Denoiser_CF_MICE.w'
            pge_weight_dir = './weights/PGE_Net_CF_MICE.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'FMD' and args.data_name == 'TP_MICE':
            
            te_data_dir = './data/test_TP_MICE.mat'
            fbi_weight_dir = './weights/FBI_Denoiser_TP_MICE.w'
            pge_weight_dir = './weights/PGE_Net_TP_MICE.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
        else : # for samsung SEM image
            te_data_dir = f"./data/{args.dataset_type}_Samsung_SNU_patches_SET{args.set_num}.hdf5"
            if args.speed_test is True:
                te_data_dir = f"./denoise_and_get_CD/test_speed_of_denoising/one_img_SET01_F32_02_x32.hdf5" # for denoising speed test
            fbi_weight_dir = f'./weights/{args.date}_FBI_Net_Grayscale_Samsung_SET{args.set_num}_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
            pge_weight_dir = f'./weights/{args.date}_PGE_Net_Grayscale_Samsung_SET{args.set_num}_Noise_est_cropsize_256.w'
            
    
            save_file_name = f"{args.date}_{args.dataset_type}data_{args.model_type}_{args.data_type}_{args.data_name}_SET{args.set_num}"
            if args.use_other_target is True:
                te_data_dir = f'./data/Samsung_tmp_dataset/{args.dataset_type}_Samsung_SNU_patches_SET{args.set_num}_divided_by_fnum.hdf5'
                fbi_weight_dir = f'./weights/{args.date}_FBI_Net_Grayscale_Samsung_SET{args.set_num}_x_as_{args.x_f_num}_y_as_{args.y_f_num}_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
                pge_weight_dir = f'./weights/{args.date}_PGE_Net_Grayscale_Samsung_SET{args.set_num}_{args.x_f_num}_Noise_est_cropsize_256.w'
                if args.integrate_all_set is True:
                    save_file_name += f"_integratedSET"

                    te_data_dir = []
                    for set_num in range(1,5):
                        te_data_dir.append(f'./data/{args.dataset_type}_Samsung_SNU_patches_SET{set_num}_divided_by_fnum.hdf5')
        
                save_file_name += f'_x_as_{args.x_f_num}_y_as_{args.y_f_num}'
            else :
                fbi_weight_dir = f'./weights/{args.date}_FBI_Net_Grayscale_Samsung_SET{args.set_num}_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
                pge_weight_dir = f'./weights/{args.date}_PGE_Net_Grayscale_Samsung_SET{args.set_num}_Noise_est_cropsize_256.w'
    if args.with_originalPGparam is True:
        save_file_name += f"_with_originalPGparam"
    print("batch size : ",args.batch_size)
    print ('te data dir : ', te_data_dir)
    print ('fbi weight dir : ', fbi_weight_dir)
    print ('pge weight dir : ', pge_weight_dir)
    save_file_name += f"_{args.loss_function}"
    save_file_name += '_layers_x' + str(args.num_layers) + '_filters_x' + str(args.num_filters)
    print ('save_file_name : ', save_file_name)

    output_folder = './output_log'
    os.makedirs(output_folder,exist_ok=True)
    f = None
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    if args.test is False:
        f = open(f"./{output_folder}/{save_file_name}",'w')
        sys.stderr = f
        sys.stdout = f       

    # Initialize model and train
    test_fbi = Test_FBI(_te_data_dir=te_data_dir, _pge_weight_dir = pge_weight_dir, _fbi_weight_dir = fbi_weight_dir, _save_file_name = save_file_name,  _args = args)
    test_fbi.eval()
    
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    if args.test is False:
        f.close()   
