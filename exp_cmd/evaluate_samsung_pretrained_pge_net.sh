#! /bin/bash
cd ../

GPU_NUM=0 #`expr $1 - 1`
DATE='221102'
# data_type = ['RawRGB', 'SIDD', 'DND', 'FMD']
# data_name = ['fivek', 'CF_MISE', 'CF_FISH', 'TP_MICE']
# `date '+%Y%m%d' | cut -d " " -f 4`
DATA_NAME='Samsung'
DATA_TYPE='Grayscale'
SET_NUM=1 #$1
BATCH_SIZE=1

CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
   --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size $BATCH_SIZE --num-layers 17 --num-filters 64


# CUDA_VISIBLE_DEVICES=0 python evaluate_pge.py --use-other-target --x-f-num F8 --y-f-num F64 --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
#     --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size $BATCH_SIZE --num-layers 17 --num-filters 64
# CUDA_VISIBLE_DEVICES=1 python evaluate_pge.py --use-other-target --x-f-num F16 --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
#     --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size $BATCH_SIZE --num-layers 17 --num-filters 64
# CUDA_VISIBLE_DEVICES=0 python evaluate_pge.py --use-other-target --x-f-num F32 --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
#     --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --batch-size $BATCH_SIZE --num-layers 17 --num-filters 64



