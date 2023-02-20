#! /bin/bash
../knock_python.sh ./train_fbi_net_fivek_EMSE_compare_PGE-Net_and_real.sh 0.01 0.0002
../knock_python.sh ./train_fbi_net_fivek_EMSE_compare_PGE-Net_and_real.sh 0.01 0.02
../knock_python.sh ./train_pge_net_fivek.sh 0.05 0.02 
../knock_python.sh ./train_fbi_net_fivek_EMSE_compare_PGE-Net_and_real.sh 0.05 0.02
