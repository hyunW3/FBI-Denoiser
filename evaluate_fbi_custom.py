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
    
    args.dataset_type = 'val'
    te_data_dir = f'./data/{args.dataset_type}_Samsung_SNU_patches_SET{args.set_num}.hdf5'
    fbi_weight_dir = f'./weights/{args.date}_FBI_Net_Grayscale_Samsung_SET{args.set_num}_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
    pge_weight_dir = f'./weights/{args.date}_PGE_Net_Grayscale_Samsung_SET{args.set_num}_Noise_est_cropsize_256.w'

    save_file_name = f"{args.date}_{args.dataset_type}data_{args.model_type}_{args.data_type}_{args.data_name}_SET{args.set_num}"
    if args.use_other_target is True:
        te_data_dir = f'./data/{args.dataset_type}_Samsung_SNU_patches_SET{args.set_num}_divided_by_fnum.hdf5'
        #fbi_weight_dir = f'./weights/{args.date}_FBI_Net_Grayscale_Samsung_SET{args.set_num}_x_as_{args.x_f_num}_y_as_{args.y_f_num}_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
        fbi_weight_dir = f'./weights/221219_FBI_Net_Grayscale_Samsung_SET7_x_as_F8_y_as_F16_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
        pge_weight_dir = f'./weights/221219_PGE_Net_Grayscale_Samsung_SET7_{args.x_f_num}_Noise_est_cropsize_256.w'
#        fbi_weight_dir = f'./weights/221025_FBI_Net_Grayscale_Samsung_SET1_x_as_F8_y_as_F16_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
#        pge_weight_dir = f'./weights/221025_PGE_Net_Grayscale_Samsung_SET1_{args.x_f_num}_Noise_est_cropsize_256.w'
        

        save_file_name += f'_test_x_as_{args.x_f_num}_y_as_{args.y_f_num}'
    else :
        fbi_weight_dir = f'./weights/{args.date}_FBI_Net_Grayscale_Samsung_SET{args.set_num}_{args.loss_function}_layers_x17_filters_x64_cropsize_256.w'
        pge_weight_dir = f'./weights/{args.date}_PGE_Net_Grayscale_Samsung_SET{args.set_num}_Noise_est_cropsize_256.w'

    print ('te data dir : ', te_data_dir)
    print ('fbi weight dir : ', fbi_weight_dir)
    print ('pge weight dir : ', pge_weight_dir)
    save_file_name += f"_{args.loss_function}"
    save_file_name += "_trained_with_SET7_" +fbi_weight_dir.split(f'SET7')[1].split(f"_{args.loss_function}")[0]
#    save_file_name += "_trained_with_" +fbi_weight_dir.split(f'SET{args.set_num}')[1].split(f"_{args.loss_function}")[0]
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
