#!/bin/bash

## slightly-BSN
../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 0 -a 0.01 -b 0.02 -t slightly-BSN -p 0.01 
../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 0 -a 0.01 -b 0.02 -t slightly-BSN -p 0.001 
../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 0 -a 0.01 -b 0.02 -t slightly-BSN -p 0.0001 
../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 0 -a 0.01 -b 0.02 -t slightly-BSN -p 0.00001

# current sh include this test. 230203 13:33
# ## prob-BSN
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.5 
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.8 
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 0 -a 0.01 -b 0.02 -t prob-BSN -p 0.2


# SEED test

## slightly-BSN
../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 1 -a 0.01 -b 0.02 -t slightly-BSN -p 0.001 
../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 3 -a 0.01 -b 0.02 -t slightly-BSN -p 0.001 

../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 1 -a 0.01 -b 0.02 -t slightly-BSN -p 0.01 
../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 3 -a 0.01 -b 0.02 -t slightly-BSN -p 0.01 


../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 1 -a 0.01 -b 0.02 -t slightly-BSN -p 0.0001 
../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 3 -a 0.01 -b 0.02 -t slightly-BSN -p 0.0001

# current sh include this test. 230203 13:33
# ## prob-BSN
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 1 -a 0.01 -b 0.02 -t prob-BSN -p 0.5 
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 1 -a 0.01 -b 0.02 -t prob-BSN -p 0.8 
# ../knock_python.sh ./train_semi_BSN_fbi_net_fivek.sh --date 230202 -g 0 -seed 1 -a 0.01 -b 0.02 -t prob-BSN -p 0.2