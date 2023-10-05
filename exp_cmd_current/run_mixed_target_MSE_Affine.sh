#! /bin/bash

alert_knock ./train_samsung_sem_MSE_Affine_mixed_target.sh 0 'v1' 'FBI_Net'
alert_knock ./train_samsung_sem_MSE_Affine_mixed_target.sh 0 'v1' 'NAFNet_light'
alert_knock ./train_samsung_sem_MSE_Affine_mixed_target.sh 0 'v2' 'FBI_Net' # F01 <-> F02,F04,F08
alert_knock ./train_samsung_sem_MSE_Affine_mixed_target.sh 0 'v2' 'NAFNet_light'