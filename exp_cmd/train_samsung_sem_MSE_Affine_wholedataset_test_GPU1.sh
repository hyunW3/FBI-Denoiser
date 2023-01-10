#!/bin/bash

./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F02 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F04 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F08 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F16 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F32 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F01 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F04 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F08 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F16 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F32 
