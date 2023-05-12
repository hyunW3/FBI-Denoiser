#!/bin/bash
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F32 v2 0.05 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F08 F32 v2 0.01 
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F01 F02 v2 0.05 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F01 F02 v2 0.01 

./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F32 v2 0.005 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F01 F02 v2 0.005




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