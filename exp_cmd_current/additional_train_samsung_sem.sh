#!/bin/bash

# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 1 F# F32 v2 --apply_median_filter_target &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 2 F# F64 v2 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 3 F# F64 v2 --apply_median_filter_target 

# after checking F64 or F64 median filtered is better, run the following
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F01 F64 v2 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F16 F64 v2 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F01 F64 v2 --apply_median_filter_target &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F16 F64 v2 --apply_median_filter_target 


alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F64 v2 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F64 v2 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F08 F64 v2 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F32 F64 v2 