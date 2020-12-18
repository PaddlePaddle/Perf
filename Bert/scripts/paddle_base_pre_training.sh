#!/bin/bash
#!/bin/bash

export PYTHONPATH=/workspace/models/PaddleNLP
export DATA_DIR=/workspace/models/bert_data/
export CUDA_VISIBLE_DEVICES=7

batch_size=${1:-32}
use_amp=${2:-"True"}
max_steps=${3:-500}
logging_steps=${4:-20}

python3.7 ./run_pretrain_single.py \
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
    --use_amp $use_amp\
    --enable_addto True
