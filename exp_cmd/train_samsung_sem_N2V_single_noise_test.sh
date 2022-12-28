#!/bin/bash
./train_samsung_sem_N2V_single_noise.sh 7 F1 &
./train_samsung_sem_N2V_single_noise.sh 7 F2 &
./train_samsung_sem_N2V_single_noise.sh 7 F4 & 
./train_samsung_sem_N2V_single_noise.sh 7 F8 &
./train_samsung_sem_N2V_single_noise.sh 7 F16 &
./train_samsung_sem_N2V_single_noise.sh 7 F32
