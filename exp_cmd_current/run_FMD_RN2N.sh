#!/bin/bash

BATCH_SIZE=8
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F01 F01 $BATCH_SIZE TP_MICE &
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F02 $BATCH_SIZE TP_MICE 
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F01 F02 $BATCH_SIZE TP_MICE &

BATCH_SIZE=1
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F01 F01 $BATCH_SIZE TP_MICE 
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F02 F02 $BATCH_SIZE TP_MICE &
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F01 F02 $BATCH_SIZE TP_MICE 

# low priority
BATCH_SIZE=8
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F04 $BATCH_SIZE TP_MICE &
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F04 F04 $BATCH_SIZE TP_MICE 

BATCH_SIZE=1
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F04 $BATCH_SIZE TP_MICE &
alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F04 F04 $BATCH_SIZE TP_MICE
# 0514 CF_MICE
# BATCH_SIZE=8
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F01 F01 $BATCH_SIZE CF_MICE &
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F02 $BATCH_SIZE CF_MICE 
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F01 F02 $BATCH_SIZE CF_MICE &

# BATCH_SIZE=1
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F01 F01 $BATCH_SIZE CF_MICE 
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F02 F02 $BATCH_SIZE CF_MICE &
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F01 F02 $BATCH_SIZE CF_MICE 

# # low priority
# BATCH_SIZE=8
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F04 $BATCH_SIZE CF_MICE &
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F04 F04 $BATCH_SIZE CF_MICE 

# BATCH_SIZE=1
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F04 $BATCH_SIZE CF_MICE &
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 0 F04 F04 $BATCH_SIZE CF_MICE

# 0513 CF_FISH
# BATCH_SIZE=8
# # alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F01 F01 $BATCH_SIZE &
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F02 $BATCH_SIZE &
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 3 F01 F02 $BATCH_SIZE 

# BATCH_SIZE=1
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 3 F01 F01 $BATCH_SIZE &
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F01 F02 $BATCH_SIZE 
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 3 F02 F02 $BATCH_SIZE &

# # low priority
# BATCH_SIZE=8
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F04 $BATCH_SIZE 
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 3 F04 F04 $BATCH_SIZE &

# BATCH_SIZE=1
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 1 F02 F04 $BATCH_SIZE 
# alert_knock ./train_FMD_RN2N_MSE_Affine.sh 3 F04 F04 $BATCH_SIZE 