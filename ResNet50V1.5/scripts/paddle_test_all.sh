#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

bash clas_fp32.sh 256 1 > paddle_gpu1_fp32_bs256.txt 2>&1

bash clas_fp16.sh 256 1 > paddle_gpu1_pure_fp16_bs256.txt 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash clas_fp32.sh 256 8 > paddle_gpu8_fp32_bs256.txt 2>&1

bash clas_fp16.sh 256 8  > paddle_gpu8_pure_fp16_bs256.txt 2>&1