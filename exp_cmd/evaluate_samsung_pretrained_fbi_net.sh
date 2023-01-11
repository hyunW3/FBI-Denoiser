#! /bin/bash
cd ../
# GPU_NUM=`expr $1 - 1`

# data_type = ['RawRGB', 'SIDD', 'DND', 'FMD']
# data_name = ['fivek', 'CF_MISE', 'CF_FISH', 'TP_MICE']
# `date '+%Y%m%d' | cut -d " " -f 4`
DATA_NAME='Samsung'
DATA_TYPE='Grayscale'
SET_NUM=$1
DATE=221221 # 221005 # for SET1~5
pge_net_weight_file=${DATE}"_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_Noise_est_cropsize_256.w" 
# CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 
# CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 
# FBI-Net weight는 F8-F32 로 하고, F32 denoise : 23dB->25dB
# FBI-Net weight는 F8-F16 로 하고, F32 denoise : 23dB->26dB
X_F_NUM=$2 # 'F32'
GPU=$3
Y_F_NUM='F64'
# CUDA_VISIBLE_DEVICES=0 python evaluate_fbi.py --use-other-target --x-f-num F8 --y-f-num F16 --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file 
CUDA_VISIBLE_DEVICES=$GPU python evaluate_fbi.py --test \
--use-other-target --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
--date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
--loss-function 'MSE_Affine' --model-type 'FBI_Net' \
--data-type $DATA_TYPE --data-name $DATA_NAME --dataset-type 'val' \
--set-num $SET_NUM --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 256 \
--pge-weight-dir $pge_net_weight_file 
