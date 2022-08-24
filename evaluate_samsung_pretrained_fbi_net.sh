#! /bin/bash

GPU_NUM=0,1

# data_type = ['RawRGB', 'SIDD', 'DND', 'FMD']
# data_name = ['fivek', 'CF_MISE', 'CF_FISH', 'TP_MICE']
# `date '+%Y%m%d' | cut -d " " -f 4`
DATA_NAME='Samsung'
DATA_TYPE='Grayscale'
F_NUM=8
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_fbi.py --date 220824 --seed 0 --noise-type 'Poisson-Gaussian' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --f-num $F_NUM --batch-size 1 --num-layers 17 --num-filters 64



