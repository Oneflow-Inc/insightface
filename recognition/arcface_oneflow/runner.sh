#!/usr/bin/bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
MODEL=${1:-"r50"}
BZ_PER_DEVICE=${2:-56}
ITER_NUM=${3:-400}
GPUS=${4:-7}
NODE_NUM=${5:-1}
DTYPE=${6:-"fp32"}
TEST_NUM=${7:-1}
DATASET=${8:-retina}
MODEL_PARALLEL=${9:-"True"}

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
    "r100") LOSS=arcface ;;
    "y1") LOSS=arcface ;;
esac



log_folder=20211130-logs-${MODEL}-${DTYPE}/insightface/bz${BZ_PER_DEVICE}/${NODE_NUM}n${gpu_num_per_node}g
mkdir -p $log_folder
log_file=$log_folder/${MODEL}_b${BZ_PER_DEVICE}_${DTYPE}_$TEST_NUM.log

if [ ${NODE_NUM} -eq 1 ] ; then
    node_ip=localhost:${gpu_num_per_node}
else
    echo "Not a valid node."
fi

export CUDA_VISIBLE_DEVICES=${GPUS}
sed -i "s/\(default.per_batch_size = \)\S*/\default.per_batch_size = ${BZ_PER_DEVICE}/" config.py


echo "Begin time: "; date;

if [ "$MODEL_PARALLEL" = "False" ] ; then
    echo "Use data patallel mode"
    python -m torch.distributed.launch \
    --nproc_per_node=${gpu_num_per_node} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=1234 train.py configs/speed.py \
    --train_num ${ITER_NUM} \
    --batch_size ${BZ_PER_DEVICE} \
    --model_parallel False \
    --fp16 ${fp16}  2>&1 | tee ${log_file} 
else
    echo "Use model patallel mode"
    python -m torch.distributed.launch \
    --nproc_per_node=${gpu_num_per_node} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=1234 train.py configs/speed.py \
    --train_num ${ITER_NUM} \
    --batch_size ${BZ_PER_DEVICE} \
    --model_parallel True \
    --fp16 ${fp16}  2>&1 | tee ${log_file}  
     
fi

echo "Writting log to $log_file"
echo "End time: "; date;


