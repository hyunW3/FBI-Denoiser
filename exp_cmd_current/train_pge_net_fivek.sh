#! /bin/bash
cd ../

GPU_NUM=3
DATE="230127" #`date "+%y%m%d"`

# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'


ALPHA=0.05
BETA=0.02 # == sigma of Poisson Gaussian noise
CUDA_VISIBLE_DEVICES=3 python main.py --date $DATE \
    --seed 0 --noise-type 'Poisson-Gaussian' \
    --loss-function 'Noise_est' --model-type 'PGE_Net' --vst-version 'MSE'\
    --data-type 'RawRGB' --data-name 'fivek' --alpha $ALPHA --beta $BETA \
    --batch-size 1 --lr 0.0001 --crop-size 200 
    

# CUDA_VISIBLE_DEVICES=2 python main.py --date $DATE \
#     --seed 0 --noise-type 'Poisson-Gaussian' \
#     --loss-function 'Noise_est' --model-type 'PGE_Net' --vst-version 'MAE'\
#     --data-type 'RawRGB' --data-name 'fivek' --alpha $ALPHA --beta $BETA \
#     --batch-size 1 --lr 0.0001 --crop-size 200
    