#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="221222"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
SET_NUM=$1
X_F_NUM=$2
Y_F_NUM='F64'
NUM=${X_F_NUM:1}
GPU_NUM=`echo "l($NUM) / l(2)" | bc -l `
GPU_NUM=${GPU_NUM::1}
GPU_NUM=`expr $GPU_NUM % 4`
# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# noise estimation
pge_net_weight_file="221217_PGE_Net_Grayscale_Samsung_SET7_"${X_F_NUM}"_Noise_est_cropsize_256.w"
echo "=== EMSE_AFFINE Train FBI==="
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py --test --use-other-target --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
	--nepochs 30 --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' \
	--model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME --set-num $SET_NUM \
	--alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 256  \
	 --pge-weight-dir $pge_net_weight_file

