# NGC PyTorch ResNet50V1.5 性能测试

此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) 实现的 ResNet50V1.5 任务的详细复现流程，包括环境介绍、环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC PyTorch ResNet50V1.5 性能测试](#ngc-pytorch-resnet50v15-性能测试)
  - [一、环境介绍](#一环境介绍)
    - [1.物理机环境](#1物理机环境)
    - [2.Docker 镜像](#2docker-镜像)
  - [二、环境搭建](#二环境搭建)
    - [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
    - [2.多机（32卡）环境搭建](#2多机32卡环境搭建)
  - [三、测试步骤](#三测试步骤)
    - [1.单机（单卡、8卡）测试](#1单机单卡8卡测试)
    - [2.多机（32卡）测试](#2多机32卡测试)
  - [四、测试结果](#四测试结果)
  - [五、日志数据](#五日志数据)

## 一、环境介绍

### 1.物理机环境

我们使用了与Paddle测试完全相同的物理机环境：

- 单机（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - Driver Version: 515.57
  - 内存：630 GB


- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

我们使用 NGC PyTorch 的代码仓库提供的Dockerfile制作镜像：

- Docker: nvcr.io/nvidia/pytorch:21.03-py3
- PyTorch：1.9.0a0+df837d0
- 模型代码：[NVIDIA/DeepLearningExamples/PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)
- CUDA：11
- cuDNN：8.0.1

## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

单机环境的搭建，我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：


- 下载NGC PyTorch repo,并进入目录

   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/PyTorch/Classification/ConvNets
   # 本次测试是在如下版本下完成的：
   git checkout cfdbf4eda13bafa6c56abd9d0f94aceb01280d55
   ```

- 制作Docker镜像

   ```bash
   docker build . -t nvidia_rn50_pytorch
   ```

- 启动Docker

   ```bash
   # 假设imagenet数据放在<path to data>目录下
   nvidia-docker run --rm -it -v <path to data>:/imagenet --ipc=host nvidia_rn50_pytorch
   ```

### 2.多机（32卡）环境搭建

- IB配置(可选）
请参考[这里](../../../utils/ib.md)
	
- MPI配置
请参考[这里](../../../utils/mpi.md)

## 三、测试步骤

### 1.单机（单卡、8卡）测试

对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现，对其提供的代码未做改动，并严格按照NGC测试使用的参数配置测试。其公开的测试报告请见：[《ResNet50 v1.5 For PyTorch》](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)

- 下载我们编写的测试脚本，并执行该脚本

   ```bash
   cd scripts/
   bash pytorch_test_all.sh
   ```

- 执行后将得到如下日志文件：

   ```
   /log/pytorch_gpu1_amp_bs256.txt
   /log/pytorch_gpu1_fp32_bs256.txt
   /log/pytorch_gpu8_amp_bs256.txt
   /log/pytorch_gpu8_fp32_bs256.txt
   ```

在NGC报告的[Training performance benchmark](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#training-performance-benchmark)小节，提供了其测试的参数配置。因此，我们提供的`pytorch_test_all.sh`是参考了其文档中的配置。

### 2.多机（32卡）测试
基础配置和上文所述的单机配置相同，多机这部分主要侧重于多机和单机的差异部分。

为了方便测试，我们封装了一下NGC的启动脚本

```
#!/bin/bash
set -xe

batch_size=$1  # batch size per gpu
num_gpus=$2    # number of gpu
precision=$3   # --amp or ""
train_steps=${4:-100}    # max train steps

export NODE_RANK=`python get_mpi_rank.py`

export env_path=/workspace/DeepLearningExamples/PyTorch/Classification/ConvNets
cd ${env_path}

python ./multiproc.py \
   --master_addr ${MASTER_NODE} \
   --master_port ${MASTER_PORT} \
   --nnodes ${NUM_NODES}  \
   --nproc_per_node ${num_gpus} \
   --node_rank ${NODE_RANK} \
./main.py --arch resnet50 \
	${precision} -b ${batch_size} \
	--training-only \
	-p 1 \
	--raport-file benchmark.json \
	--epochs 1 \
	--prof ${train_steps} ./data/imagenet
```

然后使用一个脚本测试多组实验

```
# fp32
echo "begin run 256 fp32 on 32 gpus"
$mpirun bash ./run_benchmark.sh  256 32 ""

# fp16
echo "begin run 256 fp16 on 32 gpus"
$mpirun bash ./run_benchmark.sh  256 8 "--amp"
```

其中mpi的使用参考[这里](../../../utils/mpi.md#需要把集群节点环境传给通信框架) 


## 四、测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=256) | AMP(BS=256) |
|:-----:|:-----:|:-----:|
|1 | 369.376 | 777.358  | 
|8 | 2824.18 | 5841.2  | 
|32 |10523.32| 21259.81 | 

> 关于torch数据，按照官方文档反复重测了多次未达到官方的标准。若了解相关原因，欢迎issue我们。 <br>


## 五、日志数据
- [1卡 FP32 BS=256 日志](./logs/pytorch_gpu1_fp32_bs256.txt)
- [1卡 AMP BS=256 日志](./logs/pytorch_gpu1_amp_bs256.txt)
- [8卡 FP32 BS=256 日志](./logs/pytorch_gpu8_fp32_bs256.txt)
- [8卡 AMP BS=256 日志](./logs/pytorch_gpu8_amp_bs256.txt)
- [32卡 FP32 BS=256 日志](./logs/pytorch_gpu32_fp32_bs256.txt)
- [32卡 AMP BS=256 日志](./logs/pytorch_gpu32_amp_bs256.txt)
