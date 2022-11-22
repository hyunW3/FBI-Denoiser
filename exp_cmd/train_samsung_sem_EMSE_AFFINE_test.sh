#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="221017"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
GPU_NUM=1
SET_NUM=$1
# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# noise estimation
BATCH_SIZE=250
pge_net_weight_file="221005_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_Noise_est_cropsize_256.w" 
pge_net_weight_path="./weights/"$pge_net_weight_file
if [ ! -f $pge_net_weight_path  ]; then
    echo $pge_net_weight_path "not exist, Train PGE"
    CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
        --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.0001 --crop-size 256
fi
echo "=== EMSE_AFFINE === SET"${SET_NUM}
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
    --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir $pge_net_weight_file
