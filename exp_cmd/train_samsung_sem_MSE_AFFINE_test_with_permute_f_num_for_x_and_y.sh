#! /bin/bash
# 지금(221219) 돌아가는 것 : (f1,f2),(f1,f4),(f1,f8),(f1,f16),(f1,f32),(f1,f64)
# Input : F2
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F2 F4 1 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F2 F8 2 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F2 F16 3 &
# Input : F4
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F4 F8 0 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F4 F16 1 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F8 F16 2 &
# output : F32
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F1 F32 3 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F2 F32 0 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F4 F32 1 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F8 F32 2 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 7 F16 F32 3 

# Task for SET1 ~ SET4
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 1 F8 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 2 F8 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 3 F8 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 4 F8 F64 
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 1 F16 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 2 F16 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 3 F16 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 4 F16 F64 
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 1 F32 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 2 F32 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 3 F32 F64 &
# ./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 4 F32 F64 
"
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 1 F8 F16 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 2 F8 F16 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 3 F8 F16 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 4 F8 F16 
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 1 F8 F32 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 2 F8 F32 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 3 F8 F32 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 4 F8 F32 
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 1 F16 F32 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 2 F16 F32 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 3 F16 F32 &
./train_samsung_sem_MSE_AFFINE_permute_f_num_for_x_and_y.sh 4 F16 F32 
"