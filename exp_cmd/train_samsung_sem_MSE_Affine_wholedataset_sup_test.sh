#!/bin/bash

./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 0 -v v1 -- 2 3 4 &
./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 1 -v v1 -- 1 3 4 &
./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 2 -v v1 -- 1 2 4 &
./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 3 -v v1 -- 1 2 3 &

./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 0 -v v2 -- 6 7 8 9 10 &
./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 1 -v v2 -- 5 7 8 9 10 &
./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 2 -v v2 -- 5 6 8 9 10 &
./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 3 -v v2 -- 5 6 7 9 10 &
./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 0 -v v2 -- 5 6 7 8 10 &
./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230124 \
    --gpu 1 -v v2 -- 5 6 7 8 9 
# for test

# knockknock telegram \
#     --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
#     --chat-id 1597147353 \
#     ./train_samsung_sem_MSE_Affine_wholedataset_test_GPU0.sh 
# # Unsupervised learning(Relaxed Noise2Noise) with MSE_AFFINE 
# knockknock telegram \
#     --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
#     --chat-id 1597147353 \
#     ./train_samsung_sem_MSE_Affine_wholedataset_test_GPU1.sh &
# knockknock telegram \
#     --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
#     --chat-id 1597147353 \
#     ./train_samsung_sem_MSE_Affine_wholedataset_test_GPU2.sh 
# ./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh --date 230111 --gpu 3 -- 1 2 3 4 5 6 7 8 
# # Supervised learning with MSE_AFFINE (target F64)
# #./train_samsung_sem_MSE_Affine_wholedataset_integratedSET.sh 0 12345678 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F64 12345678 & 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F64 12345678 & 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F04 F64 12345678 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F64 1 2 3 4 5 6 7 8 & 
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F16 F64 1 2 3 4 5 6 7 8 &
# ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F32 F64 1 2 3 4 5 6 7 8 
