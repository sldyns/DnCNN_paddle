#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
MODE=$2

# MODE be one of ['lite_train_lite_infer']          

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")


if [ ${MODE} = "lite_train_lite_infer" ];then
    # prepare lite data
    rm -rf ./test_tipc/data/BSD68
    rm -rf ./test_tipc/data/Set2
    cd ./test_tipc/data/ && unzip BSD68.zip && unzip Set2.zip && cd ../../
fi