#!/bin/bash

cd ../
DATE="230914"
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'

BATCH_SIZE=1
GPU_NUM=`expr $1 % 4`
X_F_NUM=$2
Y_F_NUM=$3
shift 3
OPTION=$@
echo "GPU_NUM:"$GPU_NUM
echo "X_F_NUM:"$X_F_NUM " Y_F_NUM:"$Y_F_NUM
ALPHA=0.0
BETA=0.0

echo $pge_net_weight_path "not exist, Train PGE"
CUDA_VISIBLE_DEVICES=$GPU_NUM alert_knock python main_N2N.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
    --integrate-all-set --individual-noisy-input --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
    --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 256 

