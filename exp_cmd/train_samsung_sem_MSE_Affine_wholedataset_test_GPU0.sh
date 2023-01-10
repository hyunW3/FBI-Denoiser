#!/bin/bash

./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F16 F01 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F16 F02 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F16 F04 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F16 F08 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F16 F32 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F32 F01 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F32 F02 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F32 F04 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F32 F08 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F32 F16 