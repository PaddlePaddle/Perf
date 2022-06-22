#!/bin/bash

 export PYTHONPATH=/workspace/models/PaddleNLP
 export DATA_DIR=/workspace/models/bert_data/
 export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 # 设置以下环境变量为您所用训练机器的IP地址
 export TRANER_IPS="10.10.0.1,10.10.0.2,10.10.0.3,10.10.0.4"
 export PADDLE_WITH_GLOO=0

 batch_size=${1:-32}
 use_amp=${2:-"True"}
 max_steps=${3:-500}
 logging_steps=${4:-20}

 CMD="python3.7 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 --ips $TRAINER_IPS ./run_pretrain.py"

 $CMD \
            --max_predictions_per_seq 80
            --learning_rate 5e-5
            --weight_decay 0.0
            --adam_epsilon 1e-8
            --warmup_steps 0
            --output_dir ./tmp2/
            --logging_steps 10
            --save_steps 20000
            --input_dir=$DATA_DIR
            --model_type bert
            --model_name_or_path bert-base-uncased
            --batch_size ${batch_size}
            --use_amp ${use_amp}
            --gradient_merge_steps $(expr 67584 \/ $batch_size \/ 8)"