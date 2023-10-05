#!/bin/bash

cd ../
DATE="230924"
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'

BATCH_SIZE=1
GPU_NUM=`expr $1 % 4`
X_F_NUM=$2
Y_F_NUM=$3
LOSS_FUNCTION=$4
ARCH=$5
shift 5
OPTION=$@
echo "GPU_NUM:"$GPU_NUM
echo "X_F_NUM:"$X_F_NUM " Y_F_NUM:"$Y_F_NUM
CUDA_VISIBLE_DEVICES=$GPU_NUM alert_knock python main_N2N.py --nepochs 10 \
     --x-f-num $X_F_NUM --y-f-num $Y_F_NUM $OPTION \
    --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function $LOSS_FUNCTION \
    --model-type $ARCH --data-type $DATA_TYPE --data-name $DATA_NAME \
    --batch-size $BATCH_SIZE --lr 0.001 --num-layers 17 \
    --num-filters 64 --crop-size 256 

# --loss-function 'MSE_Affine' \
#     --model-type 'FBI_Net' \
