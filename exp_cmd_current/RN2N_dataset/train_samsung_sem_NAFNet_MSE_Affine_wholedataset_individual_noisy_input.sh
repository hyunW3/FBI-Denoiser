#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="230804"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
GPU_NUM=`expr $1 % 4`
X_F_NUM='F01'
Y_F_NUM='F02'
lr=$2
wd=$3
BATCH_SIZE=$4

echo "Integrated SET GPU_NUM :"${GPU_NUM}
# echo "X_F_NUM : F"$X_F_NUM ", Y_F_NUM : F"${Y_F_NUM}
echo "lr : "${lr}", wd : "${wd}
echo "BATCH Size : "${BATCH_SIZE}
## batchsize 128 not work 
# return F.conv2d(input, weight, bias, self.stride,
# RuntimeError: Given groups=1, weight of size [64, 1, 3, 3], expected input[1, 128, 256, 256] to have 1 channels, but got 128 channels instead

# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# --test-wholedataset
# noise estimation
# lr : 0.001 (default)
echo $DATE
echo "=== Train FBI with MSE_AFFINE === integrated SET individual noisy input " $X_F_NUM"<->"$Y_F_NUM"==="
echo "== Apply Median Filter Input : "$APPLY_MEDIAN_FILTER_TARGET" (blank if not apply)=="
CUDA_VISIBLE_DEVICES=$GPU_NUM python main_RN2N.py --nepochs 10 \
    --integrate-all-set --individual-noisy-input --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
    --wholedataset-version 'v2' \
    --test-wholedataset \
    --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' \
    --model-type 'NAFNet' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr $lr --weight-decay $wd --num-layers 17 \
    --num-filters 64 --crop-size 256 \
    
