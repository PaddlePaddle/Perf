#!/bin/bash

set -x

batch_size=$1  # batch size per gpu
num_gpus=$2    # number of gpu
precision=$3   # fp32 | fp16
gradient_accumulation_steps=$(expr 67584 \/ $batch_size \/ $num_gpus)
train_batch_size=$(expr 67584 \/ $num_gpus)   # total batch_size per gpu
train_steps=${4:-20}    # max train steps

# NODE_RANK主要用于多机，单机可以不用这行。
export NODE_RANK=`python get_mpi_rank.py`
# 防止checkpoints冲突
rm -rf results/checkpoints

# run pre-training
bash scripts/run_pretraining.sh $train_batch_size 6e-3 $precision $num_gpus 0.2843 $train_steps 200 false true true $gradient_accumulation_steps
