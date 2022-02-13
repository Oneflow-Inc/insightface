#!/usr/bin/bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
MODEL=${1:-"r50"}
BZ_PER_DEVICE=${2:-96}
ITER_NUM=${3:-120}
GPUS=${4:-0,1,2,3,4,5,6,7}
NODE_NUM=${5:-1}
DTYPE=${6:-"fp32"}
TEST_NUM=${7:-1}
DATASET=${8:-retina}
MODEL_PARALLEL=${9:-"True"}
is_static=${10:-"True"}
sample_ratio=${11:-1.0}
num_classes=${12:-100000}
config_file=${13:-configs/ms1mv3_r50.py}

a=`expr ${#GPUS} + 1`
gpu_num_per_node=`expr ${a} / 2`
gpu_num=`expr ${gpu_num_per_node} \* ${NODE_NUM}`
total_bz=`expr ${BZ_PER_DEVICE} \* ${gpu_num}`

if [ "$DTYPE" = "fp16" ] ; then
    fp16=True
else
    fp16=False
fi

case $MODEL in
    "r50") LOSS=arcface ;;
    "y1") LOSS=arcface ;;
esac

set -ex



log_folder=20211130-logs-${MODEL}-${DTYPE}/insightface/bz${BZ_PER_DEVICE}/${NODE_NUM}n${gpu_num_per_node}g
mkdir -p $log_folder
log_file=$log_folder/${MODEL}_b${BZ_PER_DEVICE}_${DTYPE}_$TEST_NUM.log

if [ ${NODE_NUM} -eq 1 ] ; then
    node_ip=localhost:${gpu_num_per_node}
else
    echo "Not a valid node."
fi

export CUDA_VISIBLE_DEVICES=${GPUS}


echo "Begin time: "; date;

if [ "$MODEL_PARALLEL" = "False" ] ; then
    echo "Use data patallel mode"
    log_dir=./logs/arcface_paddle_${backbone}_${mode}_${dtype}_r${sample_ratio}_bz${BZ_PER_DEVICE}_${num_nodes}n${gpu_num_per_node}g_id${test_id}
	
    python -m paddle.distributed.launch --gpus=${GPUS} --log_dir=${log_dir} tools/train.py \
        --config_file configs/ms1mv3_r50.py \
        --is_static ${is_static} \
        --num_classes 100000 \
        --batch_size ${BZ_PER_DEVICE} \
        --fp16 ${fp16} \
        --sample_ratio ${sample_ratio} \
        --log_interval_step 50 \
        --train_unit 'step' \
        --train_num ${ITER_NUM} \
        --warmup_num 0 \
        --use_synthetic_dataset True \
        --model_parallel False \
        --do_validation_while_train False 2>&1 | tee ${log_file}	
else
        echo "Use model patallel mode"
    log_dir=./logs/arcface_paddle_${backbone}_${mode}_${dtype}_r${sample_ratio}_bz${BZ_PER_DEVICE}_${num_nodes}n${gpu_num_per_node}g_id${test_id}
	
    python -m paddle.distributed.launch --gpus=${GPUS} --log_dir=${log_dir} tools/train.py \
        --config_file configs/ms1mv3_r50.py \
        --is_static ${is_static} \
        --num_classes 100000 \
        --batch_size ${BZ_PER_DEVICE} \
        --fp16 ${fp16} \
        --sample_ratio ${sample_ratio} \
        --log_interval_step 50 \
        --train_unit 'step' \
        --train_num ${ITER_NUM} \
        --warmup_num 0 \
        --use_synthetic_dataset True \
        --model_parallel True \
        --do_validation_while_train False 2>&1 | tee ${log_file}		   
fi


echo "Writting log to $log_file"
echo "End time: "; date;


