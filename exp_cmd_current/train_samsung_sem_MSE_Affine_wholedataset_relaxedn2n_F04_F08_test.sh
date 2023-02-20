#!/bin/bash


# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F08 v1 1 2 3 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F04 F08 v1 1 2 3 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F08 v1 1 2 4 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F04 F08 v1 1 3 4 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F08 v1 2 3 4 &

./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F04 F08 v2 5 6 7 8 9 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F08 v2 5 6 7 8 10 
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F04 F08 v2 5 6 7 9 10 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F08 v2 5 6 8 9 10 
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F04 F08 v2 5 7 8 9 10 &
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F08 v2 6 7 8 9 10 

