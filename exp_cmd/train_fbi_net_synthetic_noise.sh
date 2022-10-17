#! /bin/bash
cd ../

GPU_NUM=0
#DATE=`date "+%y%m%d"`
DATE="220907"
# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'

# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=$1
BETA=$2 # == sigma of Poisson Gaussian noise
#CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date "220907" --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 220
#echo $ALPHA $BETA
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date "220907" --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 --pge-weight-dir '220907_PGE_Net_RawRGB_fivek_alpha_'$ALPHA'_beta_'$BETA'_cropsize_220.w' 
