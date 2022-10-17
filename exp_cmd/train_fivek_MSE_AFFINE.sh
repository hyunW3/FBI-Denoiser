#!/bin/bash
cd ../

CUDA_VISIBLE_DEVICES=1 python main.py --date 201104 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.01 --beta 0.02 --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 
