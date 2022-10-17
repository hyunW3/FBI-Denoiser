#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="220908"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
GPU_NUM=2
SET_NUM=1
# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# noise estimation
#CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 256
echo "=== MSE_AFFINE ==="
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir "220908_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_cropsize_256.w"
echo "=== N2V ==="
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'N2V' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir ${DATE}"_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_cropsize_256.w" 
echo "=== MSE ==="
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir ${DATE}"_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_cropsize_256.w" 
