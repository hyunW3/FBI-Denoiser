#!/bin/bash

# ./Samsung_N2N.sh 1 F01 F02 MSE FBI_Net &
# ./Samsung_N2N.sh 2 F01 F04 MSE FBI_Net &
# ./Samsung_N2N.sh 3 F01 F08 MSE FBI_Net 
./Samsung_N2N.sh 1 F01 F02 MSE FBI_Net --average-mode 'median' &
./Samsung_N2N.sh 2 F01 F04 MSE FBI_Net --average-mode 'median' &
./Samsung_N2N.sh 3 F01 F08 MSE FBI_Net --average-mode 'median'