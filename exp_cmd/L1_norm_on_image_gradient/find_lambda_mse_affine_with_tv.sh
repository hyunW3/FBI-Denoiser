#!/bin/bash

# 0530 - 2
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F01 F64 v2 0.005 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F02 F16 v2 0.005 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F04 F08 v2 0.005 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F04 F16 v2 0.005
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F08 F16 v2 0.005 &
# tv와 normal 둘다
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F01 F04 v2 0.005 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F01 F64 v2 0.005 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F32 F64 v2 0.005
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F32 F64 v1 0.005 &

alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F04 v2 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F01 F64 v2 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F32 F64 v2  
# additional
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F08 F16 v1 0.005 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F32 F64 v1

# 0530
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F16 F32 v1 0.005 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F16 F32 v2 0.005 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F08 F32 v1 0.005 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F08 F32 v2 0.005
# 0523
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F01 F08 v2 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F01 F16 v2 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F01 F32 v2 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F02 F04 v2 0.005

# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F02 F08 v2 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input_with_tv.sh 1 F# F16 v2 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input_with_tv.sh 2 F# F16 v1 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input_with_tv.sh 3 F# F64 v2 0.005

# 0521 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F16 F32 v2 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F08 F32 v1 0.005 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F16 F32 v1 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F04 F32 v2 0.005 
# ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input_with_tv.sh 1 F# F32 v2 0.005 & # error
# ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input_with_tv.sh 3 F# F08 v2 0.005 
# ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input_with_tv.sh 1 F# F04 v2 0.005 


# 0512
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F32 v2 0.05 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F08 F32 v2 0.01 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F01 F02 v2 0.05 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F01 F02 v2 0.01 

# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F32 v2 0.005 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F01 F02 v2 0.005




# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F08 F32 v2 0.8 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F32 v2 0.3 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F08 F32 v2 0.1

# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F08 F32 v2 2   &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F32 v2 0.9 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F08 F32 v2 0.8 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F08 F32 v2 0.7 

# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F08 F32 v2 0.6 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F32 v2 0.5 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F08 F32 v2 0.4 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F08 F32 v2 0.3
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F08 F32 v2 1 & 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F32 v2 0.1 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F08 F32 v2 0.01 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F08 F32 v2 0.001 