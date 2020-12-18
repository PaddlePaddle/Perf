#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.8
export FLAGS_cudnn_batchnorm_spatial_persistent=1
export FLAGS_max_inplace_grad_add=8

python ./tools/static/train.py -c ResNet50_1gpu_fp32_bs96.yaml > paddle_gpu1_fp32_bs96.txt 2>&1

python ./tools/static/train.py -c ResNet50_1gpu_amp_bs128.yaml > paddle_gpu1_amp_bs128.txt 2>&1

python ./tools/static/train.py -c ResNet50_1gpu_amp_bs208.yaml > paddle_gpu1_amp_bs208.txt 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" ./tools/static/train.py -c ResNet50_8gpu_fp32_bs96.yaml > paddle_gpu8_fp32_bs96.txt 2>&1

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" ./tools/static/train.py -c ResNet50_8gpu_amp_bs128.yaml > paddle_gpu8_amp_bs128.txt 2>&1

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" ./tools/static/train.py -c ResNet50_8gpu_amp_bs208.yaml > paddle_gpu8_amp_bs208.txt 2>&1