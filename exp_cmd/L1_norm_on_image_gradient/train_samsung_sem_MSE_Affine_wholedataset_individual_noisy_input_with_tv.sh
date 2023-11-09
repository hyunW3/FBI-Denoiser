#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="230530_find_lambda"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
GPU_NUM=`expr $1 % 4`
X_F_NUM=$2
Y_F_NUM=$3
DATASET_VERSION=$4
# v1 : SET1~SET10, v2 : SET5~SET10
LAMBDA=$5
# receive argument after 6th
shift 5
OPTION=$@

echo "Integrated SET GPU_NUM :"${GPU_NUM}
# echo "X_F_NUM : F"$X_F_NUM ", Y_F_NUM : F"${Y_F_NUM}
BATCH_SIZE=1
echo "BATCH Size : "${BATCH_SIZE}
## batchsize 128 not work 
# return F.conv2d(input, weight, bias, self.stride,
# RuntimeError: Given groups=1, weight of size [64, 1, 3, 3], expected input[1, 128, 256, 256] to have 1 channels, but got 128 channels instead

# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# --test-wholedataset
# noise estimation
echo $DATE
echo "=== Train FBI with MSE_AFFINE === integrated SET individual noisy input " $X_F_NUM"<->"$Y_F_NUM"==="
echo "== LAMBDA :  "$LAMBDA " (blank if not apply)=="
CUDA_VISIBLE_DEVICES=$GPU_NUM python main_RN2N.py --nepochs 10 $OPTION \
    --integrate-all-set --individual-noisy-input --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
    --wholedataset-version $DATASET_VERSION --lambda-val $LAMBDA \
    --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine_with_tv' \
    --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.001 --num-layers 17 \
    --num-filters 64 --crop-size 256 

# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F16 F32 v2