#! /bin/bash
cd ../

GPU_NUM=2
DATE=`date "+%y%m%d"`

# Synthetic noise datasets
DATA_TYPE='Grayscalße'
DATA_NAME='Samsung'
SET_NUM=1
# ALPHA == 0, BETA == 0 : Mixture Noise
#echo  "${DATE}_PGE_Net_${DATA_TYPE}_cropsize_256.w" 
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
#CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 256
# Epoch 050/050 [22080/22080] lr : [0.000003] -- loss: 0.0102 | pred_alpha: 0.0153 | pred_sigma: 0.0082 -- ETA: 0:00:00
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --test --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 256 --pge-weight-dir "220902_PGE_Net_Grayscale_Samsung_SET"${SET_NUM}"_cropsize_256.w" 
#"${DATE}_PGE_Net_${DATA_TYPE}_cropsize_256.w"
