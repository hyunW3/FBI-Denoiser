#! /bin/bash

GPU_NUM=3

# data_type = ['RawRGB', 'SIDD', 'DND', 'FMD']
# data_name = ['fivek', 'CF_MISE', 'CF_FISH', 'TP_MICE']

# Synthetic noise datasets
DATA_TYPE='RawRGB'
DATA_NAME='fivek'
date=230110
echo $date
###############################################
## ALPHA 0.01, 0.0002
ALPHA=0.01
BETA=0.0002

TEST_ALPHA=0.01
TEST_BETA=0.0002  
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64
TEST_ALPHA=0.01
TEST_BETA=0.02  
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64
TEST_ALPHA=0.05
TEST_BETA=0.02
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64


###############################################
## ALPHA 0.01, SIGMA 0.02
ALPHA=0.01
BETA=0.02

TEST_ALPHA=0.01
TEST_BETA=0.0002  
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64
TEST_ALPHA=0.01
TEST_BETA=0.02  
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64
TEST_ALPHA=0.05
TEST_BETA=0.02
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64

###############################################
## ALPHA 0.05, 0.02
ALPHA=0.05
BETA=0.02
TEST_ALPHA=0.01
TEST_BETA=0.0002  
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64
TEST_ALPHA=0.01
TEST_BETA=0.02  
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64
TEST_ALPHA=0.05
TEST_BETA=0.02
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64

###############################################
# Random noise
ALPHA=0
BETA=0

TEST_ALPHA=0.01
TEST_BETA=0.0002 # == sigma of Poisson Gaussian noise
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64

TEST_ALPHA=0.01
TEST_BETA=0.02 # == sigma of Poisson Gaussian noise
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64

TEST_ALPHA=0.05
TEST_BETA=0.02 # == sigma of Poisson Gaussian noise
echo "==== ALPHA = $ALPHA, BETA = $BETA ===="
echo "==== TEST ALPHA = $TEST_ALPHA, TEST BETA = $TEST_BETA ===="
CUDA_VISIBLE_DEVICES=$GPU_NUM python evaluate_pge.py --date $date --seed 0 \
    --noise-type 'Poisson-Gaussian' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --test-alpha $TEST_ALPHA --test-beta $TEST_BETA --batch-size 1 --num-layers 17 --num-filters 64
