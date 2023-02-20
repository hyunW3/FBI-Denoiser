#!/bin/bash

./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F02 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F04 12345678 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F08 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F16 12345678 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F32 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F01 12345678 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F04 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F08 12345678 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F16 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F32 
