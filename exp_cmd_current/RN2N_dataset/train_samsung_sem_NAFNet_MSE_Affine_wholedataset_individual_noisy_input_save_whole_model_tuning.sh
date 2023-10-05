#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="230728"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
GPU_NUM=`expr $1 % 4`
X_F_NUM=$2
Y_F_NUM=$3
DATASET_VERSION=$4

echo "Integrated SET GPU_NUM :"${GPU_NUM}
# echo "X_F_NUM : F"$X_F_NUM ", Y_F_NUM : F"${Y_F_NUM}

# noise estimation
# lr : 0.001 (default)
echo $DATE
echo "=== Train FBI with MSE_AFFINE === integrated SET individual noisy input " $X_F_NUM"<->"$Y_F_NUM"==="

CUDA_VISIBLE_DEVICES=$GPU_NUM python optuna_NAFNet.py \
    --integrate-all-set --individual-noisy-input --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
    --wholedataset-version $DATASET_VERSION --save-whole-model \
    --test-wholedataset \
    --date $DATE --seed 0 --noise-type 'Poisson-Gaussian'  \
    --model-type 'NAFNet' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --crop-size 256 
# 'MSE_Affine'