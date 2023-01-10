#!/bin/bash

./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F01 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F02 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F08 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F16 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F32 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F01 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F02 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F04 &
wait
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F16 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F32 