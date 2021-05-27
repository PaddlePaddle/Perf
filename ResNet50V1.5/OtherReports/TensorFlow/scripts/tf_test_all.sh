#!/bin/bash

rm -rf /tmp/result/*

python ./main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 128 \
	--data_dir=/data/tfrecords/ \
	--results_dir=/tmp/result/gpu1_fp32_bs128 > /log/tf_gpu1_fp32_bs128.txt

python ./main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 128 \
	--data_dir=/data/tfrecords/ \
	--results_dir=/tmp/result/gpu1_amp_bs128 \
	--use_tf_amp \
	--use_static_loss_scaling \
	--loss_scale=128 > /log/tf_gpu1_amp_bs128.txt

python ./main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 254 \
	--data_dir=/data/tfrecords/ \
	--results_dir=/tmp/result/gpu1_amp_bs254 \
	--use_tf_amp \
	--use_static_loss_scaling \
	--loss_scale=128 > /log/tf_gpu1_amp_bs254.txt

mpiexec --allow-run-as-root --bind-to socket -np 8 python3 main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 128 \
	--data_dir=/data/tfrecords/ \
	--results_dir=/tmp/result/gpu8_fp32_bs128 > /log/tf_gpu8_fp32_bs128.txt

mpiexec --allow-run-as-root --bind-to socket -np 8 python3 main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 128 \
	--data_dir=/data/tfrecords/ \
	--results_dir=/tmp/result/gpu8_amp_bs128 \
	--use_tf_amp \
	--use_static_loss_scaling \
	--loss_scale=128 > /log/tf_gpu8_amp_bs128.txt

mpiexec --allow-run-as-root --bind-to socket -np 8 python3 main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 254 \
	--data_dir=/data/tfrecords/ \
	--results_dir=/tmp/result/gpu8_amp_bs254 \
	--use_tf_amp \
	--use_static_loss_scaling \
	--loss_scale=128 > /log/tf_gpu8_amp_bs254.txt
