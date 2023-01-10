#!/bin/bash

# for test
./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F02 

# # Unsupervised learning(Relaxed Noise2Noise) with MSE_AFFINE 
# knockknock telegram \
#     --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
#     --chat-id 1597147353 \
#     ./train_samsung_sem_MSE_Affine_wholedataset_test_GPU1.sh &
# knockknock telegram \
#     --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
#     --chat-id 1597147353 \
#     ./train_samsung_sem_MSE_Affine_wholedataset_test_GPU2.sh &
# knockknock telegram \
#     --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
#     --chat-id 1597147353 \
#     ./train_samsung_sem_MSE_Affine_wholedataset_test_GPU3.sh 

# Supervised learning with MSE_AFFINE (target F64)
# ./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh 0 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 & 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 & 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F32 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F08 & 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F16 
