from core.train_fbi import Train_FBI
from core.train_fbi_N2N import TrainN2N_FBI
from core.train_pge_N2N import TrainN2N_PGE
from arguments import get_args
import os,sys
import torch
import numpy as np
import random
from data.core import data_loader
from neptune_setting import init_neptune
from wandb_setting import init_wandb
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
    # run_id = f"FBI-Net_semi_BSN_test_BSN_type_{args.BSN_type}_BSN_param_{args.BSN_param}_{args.data_name}_{args.noise_type}_{args.data_type}_alpha_{args.alpha}_beta_{args.beta}_mul_{args.mul}_num_of_layers_{args.num_layers}_output_type_{args.output_type}_sigmoid_value_{args.sigmoid_value}_seed_{args.seed}_date_{args.date}"
    model_type = 'RN2N' if args.x_f_num != args.y_f_num else 'N2N'
    tag_list = [model_type,args.loss_function,args.data_name,f"batch_size_{args.batch_size}", f"img-{args.average_mode}"]
    run_id = f"{model_type}_({args.x_f_num}-{args.y_f_num})_{args.loss_function}"

    if args.loss_function == 'MSE_Affine_with_tv':
        run_id += f'TV_{args.lambda_val}'
        tag_list.append(f"TV_{args.lambda_val}")
    # run_id = f"FBI-Net_train_with_originalPGparam_{args.with_originalPGparam}_{args.data_name}_{args.noise_type}_{args.data_type}_alpha_{args.alpha}_beta_{args.beta}_mul_{args.mul}_num_of_layers_{args.num_layers}_output_type_{args.output_type}_sigmoid_value_{args.sigmoid_value}_seed_{args.seed}_date_{args.date}"
    # if args.data_name == 'Samsung':
    #     project_name = 'N2N_RN2N'
    # else :
        # project_name = f'N2N_RN2N_{args.data_type}_{args.data_name}'
    project_name = 'on N2N settings'
    tag_list.append(f"{args.x_f_num}-{args.y_f_num}")

    args.logger = init_wandb(project_name = project_name, run_id = run_id,tag=tag_list)
    

if __name__ == '__main__':
    """Trains Noise2Noise."""
    save_file_name =""
        
    # load dataloader
    if args.data_name == 'Samsung':
        tr_data_dir = f"./data/train_Samsung_SNU_patches_230414.hdf5"
        # te_data_dir = f'./data/test_Samsung_SNU_patches_230414.hdf5'
        te_data_dir = f'./data/test_Samsung_SNU_patches_SET050607080910_divided_by_fnum_setnum.hdf5'
        args.integrate_all_set = True
        args.test_wholedataset = False
    elif args.data_type == 'FMD' and args.data_name in ['ALL_FMD','CF_FISH', 'CF_MICE','TP_MICE']:
        tr_data_dir = f"/mnt/ssd/hyun/datasets/FMD_dataset"
        te_data_dir = f'/mnt/ssd/hyun/datasets/FMD_dataset'
    save_file_name = f"{args.date}_{args.model_type}_{args.loss_function}_{args.data_type}_N2N_{args.x_f_num}-{args.y_f_num}_{args.data_name}_img-{args.average_mode}"
    
    
    if args.loss_function == 'MSE_Affine_with_tv':
        save_file_name += f'_l1_on_img_gradient_{args.lambda_val}'
    print ('tr data dir : ', tr_data_dir)
    print ('te data dir : ', te_data_dir)
        
    save_file_name += f"_{args.loss_function}"
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
        train = TrainN2N_FBI(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    else:
        train = TrainN2N_PGE(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    train.train()
    
    print ('Finsh training - save_file_name : ', save_file_name)
