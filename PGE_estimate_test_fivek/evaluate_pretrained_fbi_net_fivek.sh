#! /bin/bash
cd ../
GPU_NUM=3

# data_type = ['RawRGB', 'SIDD', 'DND', 'FMD']
# data_name = ['fivek', 'CF_MISE', 'CF_FISH', 'TP_MICE']

# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'

DATE=230206

ALPHA=$1
BETA=$2 # == sigma of Poisson Gaussian noise


SEED=$3
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date $DATE \
    --seed $SEED --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' \
    --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 200 \
    --with-originalPGparam 

CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date $DATE \
    --seed $SEED --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' \
    --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size 1 --num-layers 17 --num-filters 64 --crop-size 200 
