from core.train_fbi import Train_FBI
from core.train_pge import Train_PGE
from arguments import get_args
import os,sys
import torch
import numpy as np
import random
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
    tag_list = [args.loss_function,args.data_name,f"batch_size_{args.batch_size}"]
    run_id = f"{args.data_name}_({args.alpha},{args.beta})_{args.loss_function}"
    if args.apply_median_filter_target is True:
        run_id += "_apply_median_filter_target"
    if args.loss_function == 'MSE_Affine_with_tv':
        run_id += f'TVlambda_{args.lambda_val}'
        tag_list += f"lambda_{args.lambda_val}"
    # run_id = f"FBI-Net_train_with_originalPGparam_{args.with_originalPGparam}_{args.data_name}_{args.noise_type}_{args.data_type}_alpha_{args.alpha}_beta_{args.beta}_mul_{args.mul}_num_of_layers_{args.num_layers}_output_type_{args.output_type}_sigmoid_value_{args.sigmoid_value}_seed_{args.seed}_date_{args.date}"
    tag_list = []
    if args.data_name == 'Samsung':
        tag_list += f"{args.x_f_num}-{args.y_f_num}"
    args.logger = init_wandb(project_name = "RN2N_variation", run_id = run_id,tag=tag_list)
if __name__ == '__main__':
    """Trains Noise2Noise."""
    save_file_name =""
    if args.noise_type == 'Poisson-Gaussian':
        if args.data_name == 'fivek': 
            if args.data_type == 'RawRGB' and args.alpha == 0 and args.beta == 0:
                tr_data_dir = './data/Fivek_dataset/train_fivek_rawRGB_25000x256x256_cropped_random_noise.hdf5'
                te_data_dir = './data/Fivek_dataset/test_fivek_rawRGB_random_noise.hdf5'
            
                save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_' + 'random_noise'
            
            else:
                tr_data_dir = './data/Fivek_dataset/train_fivek_rawRGB_25000x256x256_cropped_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
                te_data_dir = './data/Fivek_dataset/test_fivek_rawRGB_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            
                save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta)
            if args.with_originalPGparam is True:
                save_file_name += "_with_originalPGparam"
        else : #Samsung SEM image
            raise ValueError("Invalid data_name ",args.data_name)
        if args.apply_median_filter_target:
            save_file_name += "_median_filter_target"
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
    # Initialize model and train
    # output_folder = './output_log'
    # os.makedirs(output_folder,exist_ok=True)
    # f  = None
    # if args.test is False:
    #     f = open(f"./{output_folder}/{save_file_name}",'w')
    # orig_stdout = sys.stdout
    # orig_stderr = sys.stderr
    # if args.test is False:
    #     sys.stderr = f
    #     sys.stdout = f
    
    if args.model_type != 'PGE_Net':
        train = Train_FBI(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    else:
        train = Train_PGE(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    train.train()
    # sys.stdout = orig_stdout
    # sys.stderr = orig_stderr
    # if args.test is False:
    #     f.close()   
    print ('Finsh training - save_file_name : ', save_file_name)
