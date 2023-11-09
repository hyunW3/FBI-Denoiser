#!/bin/bash




# wait_exec 974 echo "F08 F64 training start"


# running
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F02 F64 v2 0.005
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F16 F64 v2 0.005 
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 0 F04 F64 v2                # run on 230601 0915
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F16 F32 v1 # run on 
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F02 F64 v2 # run on
# TODO
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F16 F64 v1 0.005 
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 1 F08 F64 v1 0.005 


alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F08 F64 v2 
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 1 F16 F64 v2 
alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 3 F16 F64 v1 

# done but wandb crash
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 2 F08 F32 v1 0.005 
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 3 F16 F32 v1 0.005  

# done
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input.sh 2 F08 F32 v1 
# alert_knock ./train_samsung_sem_MSE_Affine_wholedataset_individual_noisy_input_with_tv.sh 0 F04 F64 v2 0.005 