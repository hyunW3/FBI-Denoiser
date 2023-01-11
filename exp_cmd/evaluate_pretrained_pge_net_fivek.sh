#! /bin/bash
cd ../
GPU_NUM=3

# data_type = ['RawRGB', 'SIDD', 'DND', 'FMD']
# data_name = ['fivek', 'CF_MISE', 'CF_FISH', 'TP_MICE']

# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'
date=230110
ALPHA=$1 #0.01 
BETA=$2 #0.0002 # == sigma of Poisson Gaussian noise
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --alpha $ALPHA --beta $BETA --batch-size 1 --num-layers 17 --num-filters 64
