#!/bin/bash

# alert_knock ./Samsung_N2N.sh 0 F01 F01
# alert_knock ./Samsung_N2N.sh 0 F01 F02
# alert_knock ./Samsung_N2N.sh 0 F02 F02

# alert_knock ./Samsung_N2N.sh 0 F01 F01 &
# alert_knock ./Samsung_N2N.sh 1 F01 F02 &
# alert_knock ./Samsung_N2N.sh 2 F02 F02 &
# alert_knock ./Samsung_N2N.sh 3 F02 F04
# alert_knock ./Samsung_N2N.sh 0 F04 F04 

alert_knock ./Samsung_N2N.sh 0 F01 F01 --save-whole-model &
alert_knock ./Samsung_N2N.sh 1 F01 F02 --save-whole-model &
alert_knock ./Samsung_N2N.sh 2 F02 F02 --save-whole-model &
alert_knock ./Samsung_N2N.sh 3 F02 F04 --save-whole-model 
alert_knock ./Samsung_N2N.sh 0 F04 F04 --save-whole-model 