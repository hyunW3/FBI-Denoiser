from core.test_pge import Test_PGE
from arguments import get_args
import torch
import numpy as np
import random
import sys,os

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
    
    if args.noise_type == 'Poisson-Gaussian':
        
        if args.data_type == 'RawRGB' and args.data_name == 'fivek' and args.alpha == 0 and args.beta == 0:
            
            te_data_dir = './data/test_fivek_rawRGB_random_noise.hdf5'
            pge_weight_dir = './weights/211127_PGE_Net_RawRGB_random_noise_cropsize_200.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_' + 'random_noise'
        
        elif args.data_type == 'RawRGB' and args.data_name == 'fivek' and args.alpha != 0 and args.beta != 0:
            
            te_data_dir = './data/test_fivek_rawRGB_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            pge_weight_dir = './weights/211127_PGE_Net_RawRGB_fivek_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'_cropsize_200.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta)
        
        elif args.data_type == 'RawRGB' and args.data_name == 'SIDD':
            
            te_data_dir = './data/test_SIDD.mat'
            pge_weight_dir = './weights/PGE_Net_SIDD.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'RawRGB' and args.data_name == 'DND':
            
            te_data_dir = './data/test_DND.mat'
            pge_weight_dir = './weights/PGE_Net_DND.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
        
        elif args.data_type == 'FMD' and args.data_name == 'CF_FISH':
            
            te_data_dir = './data/test_CF_FISH.mat'
            pge_weight_dir = './weights/PGE_Net_CF_FISH.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'FMD' and args.data_name == 'CF_MICE':
            
            te_data_dir = './data/test_CF_MICE.mat'
            pge_weight_dir = './weights/PGE_Net_CF_MICE.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'FMD' and args.data_name == 'TP_MICE':
            
            te_data_dir = './data/test_TP_MICE.mat'
            pge_weight_dir = './weights/PGE_Net_TP_MICE.w'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
        else : # for samsung SEM image
            args.dataset_type =   'val' # 'test' #
            te_data_dir = f'./data/{args.dataset_type}_Samsung_SNU_patches_SET{args.set_num}.hdf5'
            
            pge_weight_dir = f'./weights/221005_PGE_Net_Grayscale_Samsung_SET{args.set_num}_Noise_est_cropsize_256.w'
            save_file_name = f"{args.date}_{args.dataset_type}data_{args.model_type}_{args.data_type}_{args.data_name}"
    
            if args.use_other_target is True:
                pge_weight_dir = f'./weights/221025_PGE_Net_Grayscale_Samsung_SET{args.set_num}_{args.x_f_num}_Noise_est_cropsize_256.w'
                tr_data_dir = f'./data/train_Samsung_SNU_patches_SET{args.set_num}_divided_by_fnum.hdf5'
                te_data_dir = f'./data/val_Samsung_SNU_patches_SET{args.set_num}_divided_by_fnum.hdf5'
                if args.integrate_all_set is True:
                    save_file_name += f"_integratedSET"
                    pge_weight_dir = f'./weights/221025_PGE_Net_Grayscale_Samsung_integratedSET_{args.x_f_num}_Noise_est_cropsize_256.w'
                
                    tr_data_dir, te_data_dir = [], []
                    for set_num in range(1,5):
                        tr_data_dir.append(f'./data/train_Samsung_SNU_patches_SET{set_num}_divided_by_fnum.hdf5')
                        te_data_dir.append(f'./data/val_Samsung_SNU_patches_SET{set_num}_divided_by_fnum.hdf5')
                else:
                    save_file_name += f"_SET{args.set_num}"
        
                save_file_name += f'_{args.x_f_num}'  
            
        print ('te data dir : ', te_data_dir)
        print ('pge weight dir : ', pge_weight_dir)
    
    print ('save_file_name : ', save_file_name)
    
    output_folder = './output_log'
    os.makedirs(output_folder,exist_ok=True)
    f = None
    if args.test is False:
        f = open(f"./{output_folder}/{save_file_name}",'w')
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    if args.test is False:
        sys.stderr = f
        sys.stdout = f


    # Initialize model and train
    test_pge = Test_PGE(_te_data_dir=te_data_dir, _pge_weight_dir = pge_weight_dir, _save_file_name = save_file_name,  _args = args)
    test_pge.eval()

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    if args.test is False:
        f.close()   