#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="221219"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
GPU_NUM=$4 #`expr $1 - 1`

SET_NUM=$1``
X_F_NUM=$2
Y_F_NUM=$3 #'F64' # $3
echo "SET NUM : "$SET_NUM ", GPU_NUM :"${GPU_NUM}
echo "X_F_NUM : "$X_F_NUM ", Y_F_NUM : "${Y_F_NUM}
BATCH_SIZE=1 #defualt 1
## batchsize 128 not work 
# return F.conv2d(input, weight, bias, self.stride,
# RuntimeError: Given groups=1, weight of size [64, 1, 3, 3], expected input[1, 128, 256, 256] to have 1 channels, but got 128 channels instead

# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# noise estimation

pge_net_weight_file="221217_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_"${X_F_NUM}"_Noise_est_cropsize_256.w" 
pge_net_weight_path="./weights/"$pge_net_weight_file
if [ ! -f $pge_net_weight_path  ]; then
    echo $pge_net_weight_path "not exist, Train PGE"
    CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --nepochs 30 --use-other-target --x-f-num $X_F_NUM --y-f-num $Y_F_NUM --date $DATE --seed 0 \
        --noise-type 'Poisson-Gaussian' --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
        --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 256
fi
echo "=== Train FBI with MSE_AFFINE === SET"${SET_NUM}
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --test --nepochs 20 --use-other-target --x-f-num $X_F_NUM --y-f-num $Y_F_NUM --date $DATE --seed 0 \
    --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file
