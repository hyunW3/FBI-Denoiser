#!/bin/bash
function do_experiment(){
	echo "===== do_experiment start =====" | tee -a $FILENAME
	date >> $FILENAME
	$EXEC_CMD | tee $FILENAME 
}
WAIT_PID=$1
PS_CMD="ps -up "$WAIT_PID
WAIT_CMD=`$PS_CMD`
WAIT_CMD=${WAIT_CMD:(140)}
log_file="./log_wait_exec/"
if [ ! -d $log_file ]; then
	mkdir $log_file
fi
log_file+=$EXEC_CMD
EXEC_CMD=${@:2} # except 1st cmd
FILENAME=$log_file"_wait_to_start_log.txt"
echo " ==== This script log which wait for start\"" $EXEC_CMD 
echo "when " $WAIT_CMD "end === " | tee $FILENAME
for try in {0..0}
do
	while [ True ]; 
	do
	if ! ps -p $WAIT_PID > /dev/null; then	
		echo "PID $WAIT_PID  is not running" | tee -a $FILENAME
		do_experiment
		break;
	else 
		date | tee -a $FILENAME
		sleep 30m;
	fi
	done	
done;

