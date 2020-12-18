#!/bin/bash

export FLAGS_fraction_of_gpu_memory_to_use=0.8
export FLAGS_cudnn_batchnorm_spatial_persistent=1
export FLAGS_max_inplace_grad_add=8
# 设置以下环境变量为您所用训练机器的IP地址
export TRANER_IPS="10.10.0.1,10.10.0.2,10.10.0.3,10.10.0.4"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_WITH_GLOO=0

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --ips ${TRANER_IPS} ./tools/static/train.py -c ResNet50_32gpu_fp32_bs96.yaml > paddle_gpu32_fp32_bs96.txt 2>&1
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --ips ${TRANER_IPS} ./tools/static/train.py -c ResNet50_32gpu_amp_bs128.yaml > paddle_gpu32_amp_bs128.txt 2>&1
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --ips ${TRANER_IPS} ./tools/static/train.py -c ResNet50_32gpu_amp_bs208.yaml > paddle_gpu32_amp_bs208.txt 2>&1
