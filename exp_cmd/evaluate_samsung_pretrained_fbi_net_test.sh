#! /bin/bash
./evaluate_samsung_pretrained_fbi_net.sh 7 F1 0 &
./evaluate_samsung_pretrained_fbi_net.sh 7 F2 1 &
./evaluate_samsung_pretrained_fbi_net.sh 7 F4 2 &
./evaluate_samsung_pretrained_fbi_net.sh 7 F8 3 &
./evaluate_samsung_pretrained_fbi_net.sh 7 F16 0 &
./evaluate_samsung_pretrained_fbi_net.sh 7 F32 1