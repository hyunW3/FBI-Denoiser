#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="220927"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
GPU_NUM=1
SET_NUM=1
BATCH_SIZE=1 #defualt 1
## batchsize 128 not work 
# return F.conv2d(input, weight, bias, self.stride,
# RuntimeError: Given groups=1, weight of size [64, 1, 3, 3], expected input[1, 128, 256, 256] to have 1 channels, but got 128 channels instead

# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# noise estimation
#CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 256
echo "=== N2V FC-AIDE ==="
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'N2V' --model-type 'FC-AIDE' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir "220908_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_cropsize_256.w"
