<!-- omit in toc -->
# NGC TensorFlow Bert 性能复现

此处给出了基于 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) 实现的 Bert Base Pre-Training 任务的详细复现流程，包括执行环境、TensorFlow版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境介绍](#一环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [二、环境搭建](#二环境搭建)
  - [1. 单机（单卡、8卡）环境搭建](#1-单机单卡8卡环境搭建)
  - [2. 多机（32卡）环境搭建](#2-多机32卡环境搭建)
- [三、测试步骤](#三测试步骤)
  - [1. 单机（单卡、8卡）测试](#1-单机单卡8卡测试)
  - [2. 多机（32卡）测试](#2-多机32卡测试)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
  - [1.单机（单卡、8卡）日志](#1单机单卡8卡日志)


## 一、环境介绍

### 1.物理机环境

我们使用了同一个物理机环境，对 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) 的 Bert 模型进行了测试，详细物理机配置，见[Paddle Bert Base 性能测试](../../README.md#1.物理机环境)。

- 单机（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - Driver Version: 470.83.01
  - 内存：630 GB
 

- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

NGC TensorFlow 的代码仓库提供了自动构建 Docker 镜像的的 [shell 脚本](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/docker/build.sh)，

- **镜像版本**: `nvcr.io/nvidia/tensorflow:20.06-tf1-py3`
- **TensorFlow 版本**: `1.15.2+nv`
- **CUDA 版本**: `11.0`
- **cuDnn 版本**: `8.0.1`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

我们遵循了 NGC TensorFlow 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：

- **拉取代码**

  ```bash
  git clone https://github.com/NVIDIA/DeepLearningExamples
  cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
  # 本次测试是在如下版本下完成的：
  git checkout 4a15e9146a6516941ba3ae146621a5c94e4bc431
  ```

- **构建镜像**

  ```bash
  bash scripts/docker/build.sh   # 构建镜像
  bash scripts/docker/launch.sh  # 启动容器
  ```

  我们将 `launch.sh` 脚本中的 `docker` 命令换为了 `nvidia-docker` 启动的支持 GPU 的容器，其他均保持不变，脚本如下：
  ```bash
  #!/bin/bash

  CMD=${@:-/bin/bash}
  NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

  nvidia-docker run --name=test_tf_bert -it \
      --net=host \
      --shm-size=1g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
      -v $PWD:/workspace/bert \
      -v $PWD/results:/results \
      bert $CMD
  ```

- **准备数据**

  NGC TensorFlow 提供单独的数据下载和预处理脚本 [data/create_datasets_from_start.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/data/create_datasets_from_start.sh)。在容器中执行如下命令，可以下载和制作 `wikicorpus_en` 的 tfrecord 数据集。

  ```bash
  bash data/create_datasets_from_start.sh wiki_only
  ```

  由于数据集比较大，且容易受网速的影响，上述命令执行时间较长。因此，为了更方便复现竞品的性能数据，我们提供了已经处理好的 tfrecord 格式[样本数据集](https://bert-data.bj.bcebos.com/benchmark_sample%2Ftfrecord.tar.gz)。

  下载后的数据集需要放到容器中`/workspace/bert/data/`目录下，并修改[scripts/run_pretraining_lamb_phase1.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb_phase1.sh#L81)的第81行的数据集路径,如：

  ```bash
  # 解压数据集
  tar -xzvf benchmark_sample_tfrecord.tar.gz
  # 放到 data/目录下
  mv benchmark_sample_tfrecord bert/data/tfrecord
  # 修改 run_pretraining_lamb_phase1 L81 行数据集路径
  INPUT_FILES="$DATA_DIR/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/training"
  EVAL_FILES="$DATA_DIR/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/test"
  ```

### 2. 多机（32卡）环境搭建

- IB配置(可选）
请参考[这里](../../../utils/ib.md)

- MPI配置
请参考[这里](../../../utils/mpi.md)

## 三、测试步骤

为了更准确的测试 NGC TensorFlow 在 `NVIDIA DGX-1 (8x V100 32GB)` 的性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了的性能测试。

官方提供的 [scripts/run_pretraining_lamb.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb.sh) 执行脚本中，默认配置的是两阶段训练。我们此处统一仅执行 **第一阶段训练**，并根据日志中的输出的数据计算吞吐。因此我们注释掉了[scripts/run_pretraining_lamb.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb.sh#L60)的60行：

```bash
# RUN PHASE 2
# bash scripts/run_pretraining_lamb_phase2.sh $SCRIPT_ARGS |& tee -a $LOGFILE
```

**重要的配置参数：**

- **train_batch_size_phase1**: 用于指定每张卡上的 batch_size 数目
- **precision**: 用于指定精度训练模式，fp32 或 fp16
- **use_xla**: 是否开启 XLA 加速，我们统一开启此选项
- **num_gpus**: 用于指定 GPU 卡数
- **bert_model**: 用于指定 Bert 模型，我们统一指定为 **base**

### 1. 单机（单卡、8卡）测试

为了更方便地测试不同 batch_size、num_gpus、precision组合下的 Pre-Training 性能，我们单独编写了 `run_benchmark.sh` 脚本，并放在`scripts`目录下。

- **shell 脚本内容如下：**

  ```bash
  #!/bin/bash

  set -x

  batch_size=$1  # batch size per gpu
  num_gpus=$2    # number of gpu
  precision=$3   # fp32 | fp16
  num_accumulation_steps_phase1=$(expr 67584 \/ $batch_size \/ $num_gpus)
  train_steps=${4:-200}        # max train steps
  bert_model=${5:-"base"}      # base | large

  # run pre-training
  bash scripts/run_pretraining_lamb.sh $batch_size 64 8 7.5e-4 5e-4 $precision true $num_gpus 2000 200 $train_steps 200 $num_accumulation_steps_phase1 512 $bert_model
  ```
  > 注：由于原始 global_batch_size=65536 对于 batch_size=48/96 时出现除不尽情况。因此我们按照就近原则，选取 67584 作为 global_batch_size.<br>
  > 计算公式：global_batch_size = batch_size_per_gpu * num_gpu * num_accumulation_steps


- **单卡启动脚本：**

  若测试单机单卡 batch_size=96、FP32 的训练性能，执行如下命令：

  ```bash
  bash scripts/run_benchmark.sh 96 1 fp32
  ```

- **8卡启动脚本：**

  若测试单机8卡 batch_size=96、FP16 的训练性能，执行如下命令：

  ```bash
  bash scripts/run_benchmark.sh 96 8 fp16
  ```

### 2. 多机（32卡）测试

基础配置和上文所述的单机配置相同，多机这部分主要侧重于多机和单机的差异部分。

NGC TensorFlow BERT使用MPI管理作业进程，内部使用Horovod作为分布式通信框架。

- 我们需要改动原始[`mpi命令`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb_phase1.sh#L68)为我们的`$mpirun`命令请参考[这里](https://github.com/PaddlePaddle/Perf/blob/master/utils/mpi.md#%E9%80%9A%E4%BF%A1%E6%A1%86%E6%9E%B6%E5%8F%AF%E4%BB%A5%E4%BB%8Empi%E4%B8%AD%E8%8E%B7%E5%8F%96%E4%BF%A1%E6%81%AF)
- 另外把[mpi](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb_phase1.sh#L92)改为`$mpirun`

另：多机测试时，是否使用`gradient accumulation`对 性能影响很大。为了方便区分，我们同时测试了打开（默认打开）和关闭（W/O AccGrad）两种情况。关闭的测试方法为：设置[`num_accumulation_steps_phase1`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb_phase1.sh#L30)为"false",同时设置[`allreduce_post_accumulation`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb_phase1.sh#L109) 为 "False"。

## 四、测试结果

|卡数 | FP32(BS=32) | AMP(BS=96)|
|:-----:|:-----:|:-----:|
|1 | 156.33 | 530.28 |
|8 | 1231.74 | 4181.32 |
|32 | 4379.4 | 16554.3|
|32<sup>[W/O AccGrad]</sup> | 2943.8 | 12767.2|

## 五、日志数据
### 1.单机（单卡、8卡）日志

- [单卡 bs=96、FP32](./logs/tf_bert_pretraining_lamb_base_fp32_bs96_gpu1.log)
- [单卡 bs=96、AMP](./logs/tf_bert_pretraining_lamb_base_fp16_bs96_gpu1.log)
- [8卡 bs=96、FP32](./logs/tf_bert_pretraining_lamb_base_fp32_bs96_gpu8.log)
- [8卡 bs=96、AMP](./logs/tf_bert_pretraining_lamb_base_fp16_bs96_gpu8.log)
- [32卡 bs=32、FP32、GradAcc](./logs/tf_bert_pretraining_lamb_base_fp32_bs32_gpu32.log)
- [32卡 bs=48、FP32、GradAcc](./logs/tf_bert_pretraining_lamb_base_fp32_bs48_gpu32.log)
- [32卡 bs=64、AMP、GradAcc](./logs/tf_bert_pretraining_lamb_base_fp16_bs64_gpu32.log)
- [32卡 bs=96、AMP、GradAcc](./logs/tf_bert_pretraining_lamb_base_fp16_bs96_gpu32.log)
- [32卡 bs=32、FP32 no GradAcc](./logs/tf_bert_pretraining_lamb_base_without_gradacc_fp32_bs32_gpu32.log)
- [32卡 bs=48、FP32 no GradAcc](./logs/tf_bert_pretraining_lamb_base_without_gradacc_fp32_bs48_gpu32.log)
- [32卡 bs=64、AMP no GradAcc](./logs/tf_bert_pretraining_lamb_base_without_gradacc_fp16_bs64_gpu32.log)
- [32卡 bs=96、AMP no GradAcc](./logs/tf_bert_pretraining_lamb_base_without_gradacc_fp16_bs96_gpu32.log)
