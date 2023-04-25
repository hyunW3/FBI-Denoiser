#! /bin/bash
cd ../

GPU_NUM=$1
#DATE=`date "+%y%m%d"`
DATE="230417_find_lambda" #"220907"
# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'

# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=$2
BETA=$3 # == sigma of Poisson Gaussian noise
lambda_val=$4
#CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date "220907" --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 220
#echo $ALPHA $BETA
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE \
    --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine_with_tv' --lambda-val 1 \
    --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' \
    --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 \
    --num-layers 17 --num-filters 64 --crop-size 220 
    # --pge-weight-dir '211127_PGE_Net_RawRGB_fivek_alpha_'$ALPHA'_beta_'$BETA'_cropsize_220.w' 
