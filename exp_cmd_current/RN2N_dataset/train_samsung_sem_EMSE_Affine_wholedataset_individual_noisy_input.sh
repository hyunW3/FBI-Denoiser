#! /bin/bash
cd ../

#DATE=`date "+%y%m%d"`
DATE="230831"
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='Samsung'
GPU_NUM=0
X_F_NUM='F01'
Y_F_NUM='F64'
DATASET_VERSION='v2'

echo "Integrated SET GPU_NUM :"${GPU_NUM}
# echo "X_F_NUM : F"$X_F_NUM ", Y_F_NUM : F"${Y_F_NUM}
BATCH_SIZE=1
echo "BATCH Size : "${BATCH_SIZE}

# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise
# noise estimation
echo $DATE

pge_net_weight_file="230831_PGE_Net_Grayscale_Samsung_SET050607080910_with_SET01020304_x_as_F01_y_as_F64_Noise_est_cropsize_256_vst_MSE.w" 
# pge_net_weight_path="./weights/"$pge_net_weight_file
# if [ ! -f $pge_net_weight_path  ]; then
#     echo $pge_net_weight_path "not exist, Train PGE"
#     CUDA_VISIBLE_DEVICES=$GPU_NUM alert_knock python main_RN2N.py --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
#         --integrate-all-set --individual-noisy-input --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
#         --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
#         --alpha $ALPHA --beta $BETA --batch-size 1 --lr 0.0001 --crop-size 256 
# fi
echo "=== Train FBI with EMSE_AFFINE === integrated SET individual noisy input " $X_F_NUM"<->"$Y_F_NUM"==="
echo "== Apply Median Filter Input : "$APPLY_MEDIAN_FILTER_TARGET" (blank if not apply)=="
CUDA_VISIBLE_DEVICES=$GPU_NUM alert_knock python main_RN2N.py --nepochs 10 \
    --integrate-all-set --individual-noisy-input --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
    --wholedataset-version $DATASET_VERSION \
    --test-wholedataset \
    --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' \
    --model-type 'FBI_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.001 --num-layers 17 \
    --num-filters 64 --crop-size 256 \
    --pge-weight-dir $pge_net_weight_file
