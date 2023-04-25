#!/bin/bash


alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 2 F# F32 v1 &
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 3 F# F32 v2  


# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 2 F# F16 v1 --apply_median_filter_input &
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 3 F# F08 v1 --apply_median_filter_input

# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 0 F# F16 v2 &
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 2 F# F16 v1  
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 2 F# F08 v2 &
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 3 F# F04 v2 
# ../knock_python.sh ./train_samsung_sem_MSE_Affine_wholedataset_whole_noisy_input.sh 1 F# F02 v2 

 