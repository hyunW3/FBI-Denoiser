from core.produce_denoised_img import produce_denoised_img
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
    """produce_denoised_img """
    for dataset_type in ['train','val']:

        args.dataset_type = dataset_type
        
        fbi_weight_dir = f'./weights/{args.date}_FBI_Net_Grayscale_Samsung_SET{args.set_num}_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
        pge_weight_dir = f'./weights/{args.date}_PGE_Net_Grayscale_Samsung_SET{args.set_num}_Noise_est_cropsize_256.w'

        target_data_dir = f'./data/{args.dataset_type}_Samsung_SNU_patches_SET{args.set_num}.hdf5'
        save_file_name = f"produce_denoised_img_{args.date}_{args.model_type}_{args.dataset_type}_{args.data_name}_SET{args.set_num}_{args.loss_function}"
        
        print ('te data dir : ', target_data_dir)
        print ('fbi weight dir : ', fbi_weight_dir)
        print ('pge weight dir : ', pge_weight_dir)
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
        test_fbi = produce_denoised_img(target_data_dir=target_data_dir, _pge_weight_dir = pge_weight_dir, _fbi_weight_dir = fbi_weight_dir, _save_file_name = save_file_name,  _args = args)
        test_fbi.eval()
        
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if args.test is False:
            f.close()   