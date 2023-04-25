#!/bin/bash
# 230407
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F01 F08 v2 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F01 F16 v2 &

alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh      2 F# F02 v2  &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh      3 F# F04 v2  

# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh      0 F# F08 v2  


# 230331
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F02 F32 v2 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F32 v2 
# 230330
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F01 F32 v1 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 2 F# F32 v1 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 3 F# F32 v2  

# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F01 F32 v2 & 
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F01 F02 v2 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F02 F08 v2 

# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F02 F16 v2 & 
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F02 F32 v2 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F04 F16 v2 

# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F04 F32 v2 & 
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F32 v2 &
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F16 F32 v2 

# current running script 230303 23:00
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F08 F16 v2 
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F08 v2 

# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F08 F16 v2 --apply_median_filter_input 

# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F02 F04 v2 &
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F04 F08 v2 --apply_median_filter_input &
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F02 F04 v2 --apply_median_filter_input 
