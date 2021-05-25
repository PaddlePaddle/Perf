#!/bin/bash

export PYTHONPATH=/workspace/models/
export DATA_DIR=/workspace/models/bert_data/

export CUDA_VISIBLE_DEVICES=0

batch_size=${1:-32}
num_gpus=${2:-1}
use_amp=${3:-"True"}
max_steps=${4:-500}
logging_steps=${5:-20}

if [ $num_gpus = 1 ]; then
   CMD="python ./run_pretrain.py"
else
   unset CUDA_VISIBLE_DEVICES
   CMD="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 ./run_pretrain.py"
fi

$CMD \
   --model_type bert \
   --model_name_or_path bert-base-uncased \
   --max_predictions_per_seq 20 \
   --batch_size $batch_size   \
   --learning_rate 1e-4 \
   --weight_decay 1e-2 \
   --adam_epsilon 1e-6 \
   --warmup_steps 10000 \
   --input_dir $DATA_DIR \
   --output_dir ./tmp2/ \
   --logging_steps $logging_steps \
   --save_steps 50000 \
   --max_steps $max_steps \
   --use_amp $use_amp

