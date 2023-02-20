#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="230205" #"220907"
# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'

# ALPHA == 0, BETA == 0 : Mixture Noise
# SEED=$1
# ALPHA=$2
# BETA=$3 # == sigma of Poisson Gaussian noise

ALPHA=$1
BETA=$2 # == sigma of Poisson Gaussian noise
#CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date "220907" --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 220
#echo $ALPHA $BETA
# PGE_NET_WEIGHTS='211127_PGE_Net_RawRGB_fivek_alpha_'$ALPHA'_beta_'$BETA'_cropsize_200.w' 

PGE_NET_WEIGHTS='230127_PGE_Net_RawRGB_fivek_alpha_'$ALPHA'_beta_'$BETA'_Noise_est_cropsize_200.w' 


GPU_NUM=1
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE \
    --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' \
    --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' \
    --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 \
    --num-layers 17 --num-filters 64 --crop-size 200 \
    --pge-weight-dir $PGE_NET_WEIGHTS &

GPU_NUM=3
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --date $DATE \
    --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' \
    --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' \
    --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 \
    --num-layers 17 --num-filters 64 --crop-size 200 \
    --pge-weight-dir $PGE_NET_WEIGHTS \
    --with-originalPGparam