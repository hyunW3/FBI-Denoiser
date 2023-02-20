#!/bin/bash

./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F01 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F02 12345678 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F08 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F16 12345678 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F32 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F01 12345678 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F02 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F04 12345678 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F16 12345678 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F32 12345678 