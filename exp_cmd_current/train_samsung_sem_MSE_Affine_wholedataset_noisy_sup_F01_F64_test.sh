#!/bin/bash


./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F01 F64 v2 5 6 7 8 9 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F01 F64 v2 5 6 7 8 10 
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F01 F64 v2 5 6 7 9 10 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F01 F64 v2 5 6 8 9 10 
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F01 F64 v2 5 7 8 9 10 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F01 F64 v2 6 7 8 9 10 

