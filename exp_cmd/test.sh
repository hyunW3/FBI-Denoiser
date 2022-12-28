#!/bin/bash
a=$1
a=${a:1}
echo $a
a=`echo  "l($a) / l(2)" | bc -l`
a=${a::1}
echo $a
a=`expr $a % 4`
echo $a
