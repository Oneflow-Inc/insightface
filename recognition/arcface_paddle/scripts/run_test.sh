#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
MODEL=${1:-r50}
BZ_PER_DEVICE=${2:-128}
DTYPE=${3:-'fp16'}
TEST_NUM=${4:-2}
ITER_NUM=${6:-200}

export NODE1=10.11.0.2
export NODE2=10.11.0.3
export NODE3=10.11.0.4
export NODE4=10.11.0.5


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} 0  1    $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 10s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} 0,1  1    $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 10s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} 0,1,2,3  1   $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 10s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} 0,1,2,3,4,5,6,7  1    $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 10s
done