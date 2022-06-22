#!/bin/bash
# 设置以下环境变量为您所用训练机器的IP地址
export TRANER_IPS="10.10.0.1,10.10.0.2,10.10.0.3,10.10.0.4"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash clas_fp32.sh 256 32 ${TRANER_IPS} > paddle_gpu32_fp32_bs256.txt 2>&1

bash clas_fp16.sh 256 32 ${TRANER_IPS} > paddle_gpu32_pure_fp16_bs256.txt 2>&1