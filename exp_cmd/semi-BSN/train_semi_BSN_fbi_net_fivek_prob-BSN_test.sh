#!/bin/bash

# for test
# ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.5 --test-mode

# prob-BSN
../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.5 
../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.8 
../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.2 

# Next prob-BSN : at train time : random mask, at test time : No mask
../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.5 --test-mode
../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.8 --test-mode
../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.2 --test-mode

# SEED test

# ## prob-BSN
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 3 -seed 1 -a 0.01 -b 0.02 -t prob-BSN -p 0.5 
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 1 -a 0.01 -b 0.02 -t prob-BSN -p 0.8 &
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 3 -seed 1 -a 0.01 -b 0.02 -t prob-BSN -p 0.2


# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 2 -a 0.01 -b 0.02 -t prob-BSN -p 0.5 &
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 3 -seed 2 -a 0.01 -b 0.02 -t prob-BSN -p 0.8 
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek_prob-BSN.sh --date 230202 -g 2 -seed 2 -a 0.01 -b 0.02 -t prob-BSN -p 0.2