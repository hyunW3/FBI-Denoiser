#!/bin/bash
# test before start
# ./Samsung_N2N_PGENet.sh 0 F01 F64 PGE FBI_Net --test --log-off 
# ./Samsung_N2N.sh 3 F01 F64 EMSE_Affine FBI_Net --test --log-off 
# ./Samsung_N2N.sh 3 F01 F64 N2V FBI_Net --test --log-off 
# ./Samsung_N2N.sh 0 F01 F01 MSE FBI_Net --test --log-off

# # real experiment
# alert_knock ./Samsung_N2N.sh 0 F01 F01 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 0 F01 F02 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 1 F01 F04 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 2 F01 F08 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 3 F01 F16 MSE FBI_Net 

alert_knock ./Samsung_N2N.sh 0 F02 F04 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 1 F01 F32 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 2 F01 F64 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 3 F01 F64 N2V FBI_Net


alert_knock ./Samsung_N2N.sh 0 F04 F04 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 1 F04 F04 MSE NAFNet_light &
# comparsion with N2N
alert_knock ./Samsung_N2N.sh 2 F02 F02 MSE NAFNet_light &
alert_knock ./Samsung_N2N.sh 3 F02 F02 MSE FBI_Net 


alert_knock ./Samsung_N2N.sh 0 F08 F08 MSE FBI_Net &
alert_knock ./Samsung_N2N.sh 1 F08 F08 MSE NAFNet_light &
alert_knock ./Samsung_N2N.sh 2 F16 F16 MSE NAFNet_light &
alert_knock ./Samsung_N2N.sh 3 F16 F16 MSE FBI_Net 

./run_Samsung_FBI-Net.sh 3

# Rest of it 
# alert_knock ./Samsung_N2N.sh 0 F01 F16 MSE FBI_Net