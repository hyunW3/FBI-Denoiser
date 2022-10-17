#! /bin/bash
cd ../
SET_NUM=$1

knockknock telegram \
    --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
    --chat-id 1597147353 \
    ./train_samsung_sem_EMSE_AFFINE.sh $SET_NUM &

knockknock telegram \
    --token 1531143270:AAFTord-4Bi370ohc39wGyYhjGi7_VjZTwU \
    --chat-id 1597147353 \
    ./train_samsung_sem_MSE_AFFINE.sh $SET_NUM 