#! /bin/bash
cd ../
# GPU_NUM=`expr $1 - 1`

# data_type = ['RawRGB', 'SIDD', 'DND', 'FMD']
# data_name = ['fivek', 'CF_MISE', 'CF_FISH', 'TP_MICE']
# `date '+%Y%m%d' | cut -d " " -f 4`
DATA_NAME='Samsung'
DATA_TYPE='Grayscale'
SET_NUM=$1
DATE=221005
pge_net_weight_file=${DATE}"_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_Noise_est_cropsize_256.w" 

# CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 
# CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 

# CUDA_VISIBLE_DEVICES=0 python evaluate_fbi.py --use-other-target --x-f-num F8 --y-f-num F16 --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 
CUDA_VISIBLE_DEVICES=1 python evaluate_fbi.py --use-other-target --x-f-num F16 --y-f-num F64 --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 
CUDA_VISIBLE_DEVICES=2 python evaluate_fbi.py --use-other-target --x-f-num F32 --y-f-num F64 --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 


# CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file  

# CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'N2V' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 
