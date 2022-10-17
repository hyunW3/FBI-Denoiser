#! /bin/bash
cd ../
GPU_NUM=0

# data_type = ['RawRGB', 'SIDD', 'DND', 'FMD']
# data_name = ['fivek', 'CF_MISE', 'CF_FISH', 'TP_MICE']

# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'

ALPHA=0.01
BETA=0.02 # == sigma of Poisson Gaussian noise
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date 211127 --seed 0 --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --alpha $ALPHA --beta $BETA --batch-size 1 --num-layers 17 --num-filters 64
