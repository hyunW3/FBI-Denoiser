from core.train_fbi import Train_FBI
from core.train_pge import Train_PGE
from arguments import get_args
import os,sys
import torch
import numpy as np
import random
import wandb
print("torch version : ",torch.__version__)
args = get_args()

# control the randomness
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.log_off is True:
    args.logger = {}
else :
    args.logger = wandb.init(anonymous="allow")
os.makedirs("./result_data", exist_ok=True)
if __name__ == '__main__':
    """Trains Noise2Noise."""
    save_file_name = f"{args.date}_{args.model_type}_{args.loss_function}_RN2N_{args.x_f_num}-{args.y_f_num}_{args.data_name}"
    #Samsung SEM image
    if args.data_name == 'Samsung':
        if args.integrate_all_set is True:
            tr_data_dir = f'./train_Samsung_SNU_patches.hdf5'
            te_data_dir = f'./test_Samsung_SNU_patches.hdf5'

            
                
        if args.loss_function == 'MSE_Affine_with_tv':
            save_file_name += f'_l1_on_img_gradient_{args.lambda_val}'
        print ('tr data dir : ', tr_data_dir)
        print ('te data dir : ', te_data_dir)
    else:
        raise NotImplementedError
       
    if args.model_type == 'FC-AIDE':
        save_file_name += '_layers_x' + str(10) + '_filters_x' + str(64)
    elif args.model_type == 'DBSN':
        save_file_name = ''
    elif args.model_type == 'PGE_Net':
        save_file_name += f"_cropsize_{args.crop_size}_vst_{args.vst_version}"
    elif args.model_type == 'FBI_Net':
        save_file_name += '_layers_x' + str(args.num_layers) + '_filters_x' + str(args.num_filters)+ '_cropsize_' + str(args.crop_size)
        
    if args.log_off is False:
        args.logger.config.update({'Network' : args.model_type})
        args.logger.config.update({'save_filename' : save_file_name})

        args.logger.config.update({'args' : args})
    print ('save_file_name : ', save_file_name)
    
    
    if args.model_type != 'PGE_Net':
        train = Train_FBI(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    else:
        train = Train_PGE(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    train.train()
    
    print ('Finsh training - save_file_name : ', save_file_name)
