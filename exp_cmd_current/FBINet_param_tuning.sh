#!/bin/bash

# for lr in 1e-5 1e-4 1e-3
# do
#     for wd in 1e-3 1e-2 1e-1 1
#     do
#         alert_knock ./train_samsung_sem_NAFNet_MSE_Affine_wholedataset_individual_noisy_input.sh 0 $lr $wd 1 &
#         ./train_samsung_sem_NAFNet_MSE_Affine_wholedataset_individual_noisy_input.sh 1 $lr $wd 4 &
#         ./train_samsung_sem_NAFNet_MSE_Affine_wholedataset_individual_noisy_input.sh 2 $lr $wd 16 &
#         wait
#     done
# done

for batch_size in 1 2 4
do
    # for lr in 1e-5 1e-4 1e-3
    for wd in 1e-4 1e-3 1e-2 1e-1 
    do
        ./train_samsung_sem_FBINet_MSE_Affine_wholedataset_individual_noisy_input.sh 1 1e-4 $wd $batch_size &
        ./train_samsung_sem_FBINet_MSE_Affine_wholedataset_individual_noisy_input.sh 3 1e-3 $wd $batch_size 
    done
done

