#!/bin/bash

../knock_python.sh python 0-whole_process.py --target F16_v2 --gpu 0 --weight ../weights/230303_FBI_Net_Grayscale_Samsung_SET050607080910individual_x_as_F08_y_as_F16_median_filter_input_MSE_Affine_layers_x17_filters_x64_cropsize_256.w &
../knock_python.sh python 0-whole_process.py --target F16_v2 --gpu 1 --weight ../weights/230303_FBI_Net_Grayscale_Samsung_SET050607080910individual_x_as_F08_y_as_F16_MSE_Affine_layers_x17_filters_x64_cropsize_256.w &
../knock_python.sh python 0-whole_process.py --target F08_v2 --gpu 2 --weight ../weights/230303_FBI_Net_Grayscale_Samsung_SET050607080910individual_x_as_F04_y_as_F08_median_filter_input_MSE_Affine_layers_x17_filters_x64_cropsize_256.w &
../knock_python.sh python 0-whole_process.py --target F08_v2 --gpu 3 --weight ../weights/230303_FBI_Net_Grayscale_Samsung_SET050607080910individual_x_as_F04_y_as_F08_MSE_Affine_layers_x17_filters_x64_cropsize_256.w