#!/bin/bash



## previous one
# alert_knock ./Samsung_N2N.sh 0 F01 F01
# alert_knock ./Samsung_N2N.sh 0 F01 F02
# alert_knock ./Samsung_N2N.sh 0 F02 F02

# alert_knock ./Samsung_N2N.sh 0 F01 F01 &
# alert_knock ./Samsung_N2N.sh 1 F01 F02 &
# alert_knock ./Samsung_N2N.sh 2 F02 F02 &
# alert_knock ./Samsung_N2N.sh 3 F02 F04
# alert_knock ./Samsung_N2N.sh 0 F04 F04 

# ./Samsung_N2N.sh 0 F01 F04
# ./Samsung_N2N.sh 1 F01 F08 
# ./Samsung_N2N.sh 0 F02 F08 &
# ./Samsung_N2N.sh 1 F04 F04
# ./Samsung_N2N.sh 0 F04 F08 &
# ./Samsung_N2N.sh 1 F08 F08
##################  

# self-sup (N2V)
# ./Samsung_N2N.sh 1 F01 F01 N2V FBI_Net &
# ./Samsung_N2N.sh 2 F01 F04 MSE_Affine NAFNet_light &
# ./Samsung_N2N.sh 3 F01 F08 MSE_Affine NAFNet_light &
# # FBI-Net : need PGE-Net
# pge_net_weight_file="230914_PGE_Net_Noise_est_Grayscale_N2N_F01-F01_Samsung_Noise_est_cropsize_256_vst_MSE.w" 
# pge_net_weight_path="./weights/"$pge_net_weight_file
# if [ ! -f $pge_net_weight_path  ]; then
#     ./Samsung_N2N_PGENet.sh 0 F01 F01 
# fi
# # ./Samsung_N2N.sh 2 F01 F01 EMSE_Affine FBI_Net --pge-weight-dir ./weights/230914_PGE_Net_Noise_est_Grayscale_N2N_F01-F01_Samsung_Noise_est_cropsize_256_vst_MSE.w # $pge_net_weight_path
# ./Samsung_N2N.sh 2 F01 F01 EMSE_Affine FBI_Net --pge-weight-dir $pge_net_weight_file
# # ablation
# ./Samsung_N2N.sh 0 F01 F01 MSE_Affine NAFNet_light &
# ./Samsung_N2N.sh 1 F01 F02 MSE_Affine NAFNet_light 

# ./Samsung_N2N.sh 0 F01 F01 MSE_Affine FBI_Net &
# ./Samsung_N2N.sh 1 F01 F02 MSE_Affine FBI_Net &
# ./Samsung_N2N.sh 2 F01 F04 MSE_Affine FBI_Net &
# ./Samsung_N2N.sh 3 F01 F08 MSE_Affine FBI_Net

# comparsion with N2N
# ./Samsung_N2N.sh 1 F02 F02 MSE_Affine FBI_Net
# ./Samsung_N2N.sh 2 F02 F04 MSE_Affine FBI_Net
# ./Samsung_N2N.sh 3 F04 F04 MSE_Affine FBI_Net
./Samsung_N2N.sh 1 F01 F32 MSE FBI_Net # epoch 9
./Samsung_N2N.sh 2 F01 F32 MSE_Affine FBI_Net 