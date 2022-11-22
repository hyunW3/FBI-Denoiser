from core.train_fbi import Train_FBI
from core.train_pge import Train_PGE
from arguments import get_args
import os,sys
import torch
import numpy as np
import random

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
    save_file_name =""
    if args.noise_type == 'Poisson-Gaussian':
        if args.data_name == 'fivek': 
            if args.data_type == 'RawRGB' and args.alpha == 0 and args.beta == 0:
                tr_data_dir = './data/train_fivek_rawRGB_25000x256x256_cropped_random_noise.hdf5'
                te_data_dir = './data/test_fivek_rawRGB_random_noise.hdf5'
            
                save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_' + 'random_noise'
            
            else:
                tr_data_dir = './data/train_fivek_rawRGB_25000x256x256_cropped_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
                te_data_dir = './data/test_fivek_rawRGB_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            
                save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta)
        else : #Samsung SEM image
            tr_data_dir = f"./data/train_Samsung_SNU_patches_SET{args.set_num}.hdf5"
            te_data_dir = f'./data/val_Samsung_SNU_patches_SET{args.set_num}.hdf5'
            save_file_name = f"{args.date}_{args.model_type}_{args.data_type}_{args.data_name}_SET{args.set_num}"

            if args.test :
                tr_data_dir = f"./data/val_Samsung_SNU_patches_SET{args.set_num}.hdf5"
            
            if args.train_with_MSEAffine :
                tr_data_dir = f"./result_data/denoised_with_MSE_Affine_train_Samsung_SNU_patches_SET{args.set_num}.hdf5"
                if args.test :
                    tr_data_dir = f"./result_data/denoised_with_MSE_Affine_val_Samsung_SNU_patches_SET{args.set_num}.hdf5"
                te_data_dir = f'./result_data/denoised_with_MSE_Affine_val_Samsung_SNU_patches_SET{args.set_num}.hdf5'
                save_file_name += '_clean_as_MSE_Affine'
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name) 

            if args.use_other_target is True:
                tr_data_dir = f'./data/train_Samsung_SNU_patches_SET{args.set_num}_divided_by_fnum.hdf5'
                te_data_dir = f'./data/val_Samsung_SNU_patches_SET{args.set_num}_divided_by_fnum.hdf5'
                if args.integrate_all_set is True:
                    save_file_name += f"_integratedSET"

                    tr_data_dir, te_data_dir = [], []
                    for set_num in range(1,5):
                        tr_data_dir.append(f'./data/train_Samsung_SNU_patches_SET{set_num}_divided_by_fnum.hdf5')
                        te_data_dir.append(f'./data/val_Samsung_SNU_patches_SET{set_num}_divided_by_fnum.hdf5')
                else:
                    save_file_name += f"_SET{args.set_num}"
        
                if args.model_type != 'PGE_Net':
                    save_file_name += f'_x_as_{args.x_f_num}_y_as_{args.y_f_num}'
                else:
                    save_file_name += f'_{args.x_f_num}'        
        
        print ('tr data dir : ', tr_data_dir)
        print ('te data dir : ', te_data_dir)
        
    save_file_name += f"_{args.loss_function}"
    if args.model_type == 'FC-AIDE':
        save_file_name += '_layers_x' + str(10) + '_filters_x' + str(64)
    elif args.model_type == 'DBSN':
        save_file_name = ''
    elif args.model_type == 'PGE_Net':
        save_file_name += '_cropsize_' + str(args.crop_size)
    elif args.model_type == 'FBI_Net':
        save_file_name += '_layers_x' + str(args.num_layers) + '_filters_x' + str(args.num_filters)+ '_cropsize_' + str(args.crop_size)
    
    print ('save_file_name : ', save_file_name)
    # Initialize model and train
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
    if args.model_type != 'PGE_Net':
        train = Train_FBI(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    else:
        train = Train_PGE(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    train.train()
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    if args.test is False:
        f.close()   
    
