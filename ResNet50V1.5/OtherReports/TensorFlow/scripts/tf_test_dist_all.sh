#!/bin/bash
set -x

export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=0
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=xgbe0
export IBV_DRIVERS=mlx5

rm -rf /tmp/result/*
cd  ${HOME_WORK_DIR}/workspace/DeepLearningExamples/TensorFlow/Classification/ConvNets
mkdir -p data/tfrecords/log

mpirun="/usr/local/openmpi-3.1.0/bin/orterun --allow-run-as-root -tag-output \
   --bind-to none \
   -timestamp-output --hostfile ${TRAIN_WORKSPACE}/hostfile \
   -mca btl_tcp_if_exclude docker0,lo,matrixdummy0,matrix0 \
   -x PATH -x LD_LIBRARY_PATH \
   -x NCCL_IB_GID_INDEX \
   -x NCCL_IB_DISABLE \
   -x NCCL_IB_CUDA_SUPPORT \
   -x NCCL_P2P_DISABLE \
   -x NCCL_DEBUG \
   -x NCCL_SOCKET_IFNAME \
   -x IBV_DRIVERS"

echo "runing fp32 128"
$mpirun  python main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 128 \
	--data_dir=./data/tfrecords/ \
	--results_dir=/tmp/result/gpu8_fp32_bs128

echo "runing fp32 256"
$mpirun  python main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 256 \
	--data_dir=./data/tfrecords/ \
	--results_dir=/tmp/result/gpu8_fp32_bs256

echo "runing fp16 128"
$mpirun  python main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 128 \
	--data_dir=./data/tfrecords/ \
	--results_dir=/tmp/result/gpu8_amp_bs128 \
	--use_tf_amp \
	--use_static_loss_scaling \
	--loss_scale=128

echo "runing fp16 256"
$mpirun  python main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 256 \
	--data_dir=./data/tfrecords/ \
	--results_dir=/tmp/result/gpu8_amp_bs208 \
	--use_tf_amp \
	--loss_scale=128
