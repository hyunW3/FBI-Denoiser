#!/bin/bash

./train_samsung_sem_NAFNet_light_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F02 v2 &
./train_samsung_sem_NAFNet_light_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F01 F04 v2 &
./train_samsung_sem_NAFNet_light_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F01 F08 v2
./train_samsung_sem_NAFNet_light_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F04 v2 &
./train_samsung_sem_NAFNet_light_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F02 F08 v2 &
./train_samsung_sem_NAFNet_light_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F08 v2


./train_samsung_sem_NAFNet_light_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F01 F64 v2
