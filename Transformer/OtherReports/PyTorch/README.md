<!-- omit in toc -->
# NGC PyTorch Transformer 性能复现


此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) 实现的 Transformer 任务的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

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

我们使用了同一个物理机环境，对 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) 的 Transformer 模型进行了测试，详细物理机配置，见[Paddle Transformer 性能测试](../../README.md#1.物理机环境)。

- 单机V100（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - Driver Version: 515.57
  - 内存：630 GB
- 单机A100（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：NVIDIA A100-SXM4-40GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 160
  - Driver Version: 515.48.07
  - 内存：1510 GB
- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

NGC PyTorch 的代码仓库提供了自动构建 Docker 镜像的 [Dockerfile](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/Dockerfile)，

- **镜像版本**: `nvcr.io/nvidia/pytorch:22.06-py3`
- **PyTorch 版本**: `1.13.0a0+340c412`
- **CUDA 版本**: `11.4`
- **cuDnn 版本**: `8.4`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#quick-start-guide) 教程搭建了测试环境，主要过程如下：

- **拉取代码**

    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/Translation/Transformer
    # 本次测试是在如下版本下完成的：
    git checkout cfdbf4eda13bafa6c56abd9d0f94aceb01280d55
    ```

- **构建镜像**

    ```bash
    docker build . -t your.repository:transformer   # 构建镜像
    nvidia-docker run -it --rm --ipc=host your.repository:transformer bash  # 启动容器
    ```

- **准备数据**

    NGC PyTorch 提供单独的数据下载和预处理脚本 [scripts/run_preprocessing.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/scripts/run_preprocessing.sh)。在容器中执行如下命令，可以下载和制作WMT14英德数据集。

    ```bash
    bash scripts/run_preprocessing.sh
    ```

### 2. 多机（32卡）环境搭建

- IB配置(可选）
请参考[这里](../../../utils/ib.md)

- MPI配置
请参考[这里](../../../utils/mpi.md)

## 三、测试步骤

为了更准确的测试 NGC PyTorch 在 `NVIDIA DGX-1 (8x V100 32GB)` 上的性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了的性能测试。

根据官方提供的 [train.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/train.py) 脚本中执行计算吞吐。

**重要的配置参数：**

- **max-tokens**: 单卡 batch_size
- **amp**: 用于指定是否开启amp训练。
- **amp-devel**: O1代表amp，O2代表fp16。

### 1. 单机（单卡、8卡）测试

为了更方便地测试不同 batch_size、num_gpus、precision组合下的性能，我们编写了 `run_benchmark.sh` 脚本:

``` bash
num_trainers=${OMPI_COMM_WORLD_SIZE:-1}
num_gpu=$1
batch=$2
total_cards=$((num_trainers*num_gpu))
if [[ $3 == 'fp32' ]];then
    appends=""
elif [[ $3 == 'fp16' ]];then
    appends='--amp'
else
    echo "unexpect fp32 or fp16"
    exit
fi

# RANK主要用于多机，单机可以不用这行
export RANK=${OMPI_COMM_WORLD_RANK:-0}

distribute="--nnodes ${num_trainers} --node_rank ${RANK}  \
    --master_addr ${MASTER_ADDR:-127.0.0.1} --master_port ${MASTER_PORT:-8421}"
envs='--distributed-init-method env://'


python  -m torch.distributed.launch --nproc_per_node=${num_gpu}  ${distribute} train.py \
  /data/wmt14_en_de_joined_dict \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas 0.9 0.997 \
  --adam-eps 1e-9 \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000  \
  --lr 0.000846 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens ${batch} \
  --seed 1 \
  --max-epoch 40 \
  --no-epoch-checkpoints \
  --fuse-layer-norm \
  --log-interval 10 \
  ${envs} \
  --save-dir /workspace/checkpoint \
  ${appends} 

```


- **单卡启动脚本：**

    若测试单机单卡 batch_size=5120、FP32 的训练性能，执行如下命令：

    ```bash
    bash scripts/run_benchmark.sh 1 5120 fp32
    ```

- **8卡启动脚本：**

    若测试单机8卡 batch_size=5120、FP16 的训练性能，执行如下命令：

    ```bash
    bash scripts/run_benchmark.sh 8 5120 fp16
    ```


### 2. 多机（32卡）测试
基础配置和上文所述的单机配置相同，多机这部分主要侧重于多机和单机的差异部分。
我们需要把环境变量`${MASTER_ADDR}`  `${MASTER_PORT}`传递给`run_pretraining.sh`脚本，即可在单机的基础上完成多机的启动。

- **多机启动脚本**

	`$mpirun`命令请参考[这里](../../../utils/mpi.md#需要把集群节点环境传给通信框架)

	```
	# fp32
	echo "begin run bs:2560 fp32 on 8 gpus"
	$mpirun bash ./run_benchmark.sh 8 2560 fp32

    # amp
	echo "begin run bs:5120 amp on 8 gpus"
	$mpirun bash ./run_benchmark.sh 8 5120 amp
 
    # fp16
	echo "begin run bs:5120 fp16 on 8 gpus"
	$mpirun bash ./run_benchmark.sh 8 5120 fp16

	# add more test
	```

## 四、测试结果

### V100 (单位： tokens/s)

|卡数 | FP32(BS=5120) | FP16(BS=5120) |
|:-----:|:-----:|:-----:|
|1 | 9037.34  | 32406.5  | 
|8 | 65075  | 209138 |
|32 | 166352.6 | 385625.7 | 

### A100 (单位： |tokens/s)

|卡数 | FP32(BS=5120) | FP16(BS=5120) |
|:-----:|:-----:|:-----:|
|1 | 38604.2  | 44544  | 
|8 | 257124  | 286728 |

## 五、日志数据
### 1.日志
- [V100-单机单卡、FP32](./logs/V100_LOG/transformer_pytorch_bs5120_fp32_gpu1)
- [V100-单机八卡、FP32](./logs/V100_LOG/transformer_pytorch_bs5120_fp32_gpu8)
- [V100-单机单卡、FP16](./logs/V100_LOG/transformer_pytorch_bs5120_fp16_gpu1)
- [V100-单机八卡、FP16](./logs/V100_LOG/transformer_pytorch_bs5120_fp16_gpu8)
- [V100-4机32卡、FP32](./logs/V100_LOG/pytorch_gpu32_fp32_bs2560)
- [V100-4机32卡、FP16](./logs/V100_LOG/pytorch_gpu32_fp16_bs5120)
- [V100-4机32卡、AMP ](./logs/V100_LOG/pytorch_gpu32_amp_bs5120)
- [A100-单机单卡、FP32](./logs/A100_LOG/transformer_pytorch_bs5120_fp32_gpu1)
- [A100-单机八卡、FP32](./logs/A100_LOG/transformer_pytorch_bs5120_fp32_gpu8)
- [A100-单机单卡、FP16](./logs/A100_LOG/transformer_pytorch_bs5120_fp16_gpu1)
- [A100-单机八卡、FP16](./logs/A100_LOG/transformer_pytorch_bs5120_fp16_gpu8)