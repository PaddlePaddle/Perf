<!-- omit in toc -->
# NGC PyTorch DeepLabV3P 性能复现


此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) 实现的 [mmseg-DeepLabV3P](https://github.com/open-mmlab/mmsegmentation) 任务的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境介绍](#一环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [二、环境搭建](#二环境搭建)
  - [1. 单机（单卡、8卡）环境搭建](#1-单机单卡8卡环境搭建)
- [三、测试步骤](#三测试步骤)
  - [1. 单机（单卡、8卡）测试](#1-单机单卡8卡测试)
  - [2. 多机（32卡）测试](#2-多机32卡测试)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
  - [1.日志](#1日志)


## 一、环境介绍

### 1.物理机环境

我们使用了同一个物理机环境，对 [NGC PyTorch](https://github.com/open-mmlab/mmsegmentation) 的 模型进行了测试，详细物理机配置，见[Paddle DeepLabV3P 性能测试](../../README.md#1.物理机环境)。

- 单机（单卡、8卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB
 
- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

NGC PyTorch 的代码仓库提供了自动构建 Docker 镜像的 [Dockerfile](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/Dockerfile)，

- **镜像版本**: `nvcr.io/nvidia/pytorch:20.06-py3`
- **PyTorch 版本**: `1.6.0a0+9907a3e`
- **CUDA 版本**: `11.0`
- **cuDnn 版本**: `8.0.1`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide) 教程搭建了测试环境，主要过程如下：

- **拉取镜像**

```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
    # 本次测试是在如下版本下完成的：
    git checkout 99b1c898cead5603c945721162270c2fe077b4a2
```

- **构建镜像**

```bash
    bash scripts/docker/build.sh   # 构建镜像
```
在此基础上，我们修改了一部分代码，以达到AI-Rank要求的输出和更方便的配置，并在不影响原始代码性能和精度的前提下修复了原始代码的[bug](https://github.com/open-mmlab/mmsegmentation/pull/522), 最终使用的代码是[sljlp/mmsegmentation](https://github.com/sljlp/mmsegmentation)

- **从[mmsegmentation](https://github.com/sljlp/mmsegmentation)拉取模型代码**

```bash
    cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
    git clone https://github.com/sljlp/mmsegmentation.git -b benchmark
    cd mmsegmentation
    #添加官方库
    git remote add upstream https://github.com/open-mmlab/mmsegmentation.git
    git merge upstream/master
    
```

- **启动镜像**
```bash
    bash scripts/docker/launch.sh  # 启动容器
```
    我们将 `launch.sh` 脚本中的 `docker` 命令换为了 `nvidia-docker` 启动的支持 GPU 的容器，同时将`BERT`(即`$pwd`)目录替换为`mmsegmentation`目录，其他均保持不变，脚本如下：
```bash
    #!/bin/bash

    CMD=${1:-/bin/bash}
    NV_VISIBLE_DEVICES=${2:-"all"}
    DOCKER_BRIDGE=${3:-"host"}

    nvidia-docker run --name test_bert_torch -it  \
    --net=$DOCKER_BRIDGE \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e LD_LIBRARY_PATH='/workspace/install/lib/' \
    -v $PWD/mmsegmentation:/workspace/mmsegmentation \
    -v $PWD/mmsegmentation/results:/results \
    mmsegmentation $CMD
```  

- **准备数据**

   DeepLabV3P 模型是基于 [Cityscapes 数据集](https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar) 进行的训练的。
```   bash
    # 下载cityscapes  
    wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar  

    # 解压数据集
    tar -xzvf cityscapes.tar

    # checksum
    md5sum cityscapes.tar
    输出：cityscapes.tar 37724b19b6e5d41f9f147936d60b3c29

    # 放到 data/ 目录下
    mv cityscapes PaddleSeg/data/
```

## 三、测试步骤

为了更准确的测试 NGC PyTorch 在 `NVIDIA DGX-1 (8x V100 32GB)` 上的性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了的性能测试。

根据官方提供的 [train.py](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/train.py) 脚本中执行计算吞吐。

**重要的配置参数：**

- **batch_size**: 单卡 batch_size
- **fop16**: 用于指定是否开启amp训练。

### 1. 单机（单卡、8卡）测试

为了更方便地测试不同 batch_size、num_gpus、precision组合下的性能，我们修改了tools/train.py的接受参数列表，同时编写了`run_benchmark.sh` 脚本:

``` bash
cd mmsegmentation

if [ $2 = fp16 ]
then FP16="--fp16"
else FP16=
fi

if [ $1 = 1 ]
then
export CUDA_VISIBLE_DEVICES=0
cp configs/_base_/models/deeplabv3plus_r50-d8.py configs/_base_/models/deeplabv3plus_r50-d8.tmp.py
sed -i 's/SyncBN/BN/g' configs/_base_/models/deeplabv3plus_r50-d8.py

python3 tools/train.py \
    configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py \
    --no-validate \
    --gpus 1 \
    --max_iters 40 \
    --log_iters 4 \
    --batch_size 8 \
    --num_workers 8 \
	$FP16

fi

if [ $1 = 1 ]
then mv configs/_base_/models/deeplabv3plus_r50-d8.tmp.py configs/_base_/models/deeplabv3plus_r50-d8.py
fi

if [ $1 -gt 1 ]
then
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CONFIG_FILE=configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py
total_gpus=$1
NUM_GPUS=8
NUM_NODES=$((total_gpus/NUM_GPUS))
python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
               --nnodes=${NUM_NODES} --node_rank=$NODE_RANK --master_addr=$MASTER_NODE \
               --master_port=$MASTER_PORT \
               tools/train.py ${CONFIG_FILE} --launcher pytorch --no-validate \
	       --max_iters 10 --log_iters 2 --batch_size 8 --num_workers 8 $FP16
fi

```


- **单卡启动脚本：**

    若测试单机单卡 batch_size=8、FP32 的训练性能，执行如下命令：

    ```bash
    bash run_benchmark.sh 1 fp32
    ```

- **8卡启动脚本：**

    若测试单机8卡 batch_size=8、FP16 的训练性能，执行如下命令：

    ```bash
    bash run_benchmark.sh 8 fp16
    ```

### 2. 多机（32卡）测试
基础配置和上文所述的单机配置相同，多机这部分主要侧重于多机和单机的差异部分。
我们需要配置变量`${MASTER_ADDR}`  `${MASTER_PORT}`, 即可在单机的基础上完成多机的启动。

- **多机启动脚本**

	```bash
	# 4机 32 卡 fp16
	bash run_benchmark.sh 32 fp16

	```

## 四、测试结果

> 单位： samples/sec

|卡数 | FP32(BS=8) | AMP(BS=8) |
|:-----:|:-----:|:-----:|
|1 | 15  | 4.3  | 
|8 | 75  | 16.8  |
|32 | 243 | 61.0 | 

## 五、日志数据
### 1.日志
- [单机单卡、FP32](./logs/pytorch/deeplabv3p_c1_fp32.log)
- [单机八卡、FP32](./logs/pytorch/deeplabv3p_c8_fp32.log)
- [单机单卡、AMP](./logs/pytorch/deeplabv3p_c1_fp16.log)
- [单机八卡、AMP](./logs/pytorch/deeplabv3p_c8_fp16.log)
- [4机32卡、FP32](./logs/pytorch/deeplabv3p_c32_fp32.log)
- [4机32卡、AMP ](./logs/pytorch/deeplabv3p_c32_fp16.log)
