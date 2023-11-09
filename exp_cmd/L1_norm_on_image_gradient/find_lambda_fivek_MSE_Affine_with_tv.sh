#!/bin/bash

alert_knock ./train_fbi_net_synthetic_noise_with_tv.sh 0 0.01 0.02 0.3 &
alert_knock ./train_fbi_net_synthetic_noise_with_tv.sh 1 0.01 0.02 0.8
