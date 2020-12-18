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
  - 系统：CentOS Linux release 7.5.1804
  - GPU：Tesla V100-SXM2-16GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 38
  - Driver Version: 450.80.02
  - 内存：432 GB

- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

我们使用 NGC PyTorch 的代码仓库提供的Dockerfile制作镜像：

- Docker: nvcr.io/nvidia/pytorch:20.07-py3
- PyTorch：1.6.0a0+9907a3e
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
   git checkout 99b1c898cead5603c945721162270c2fe077b4a2
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

> 我们会后续补充此内容。

## 三、测试步骤

### 1.单机（单卡、8卡）测试

对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现，对其提供的代码未做改动，并严格按照NGC测试使用的参数配置测试。其公开的测试报告请见：[《ResNet50 v1.5 For PyTorch》](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)

- 下载我们编写的测试脚本，并执行该脚本
   ```bash
   wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/OtherReports/PyTorch/scripts/pytorch_test_all.sh
   bash pytorch_test_all.sh
   ```

- 执行后将得到如下日志文件：
   ```
   /log/pytorch_gpu1_fp32_bs128.txt
   /log/pytorch_gpu1_amp_bs128.txt
   /log/pytorch_gpu1_amp_bs256.txt
   /log/pytorch_gpu8_fp32_bs128.txt
   /log/pytorch_gpu8_amp_bs128.txt
   /log/pytorch_gpu8_amp_bs248.txt
   ```

在NGC报告的[Training performance benchmark](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#training-performance-benchmark)小节，提供了其测试的参数配置。因此，我们提供的`pytorch_test_all.sh`是参考了其文档中的配置。

### 2.多机（32卡）测试

> 我们会后续补充此内容。


## 四、测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=128) | AMP(BS=128) | AMP(BS=256)|
|:-----:|:-----:|:-----:|:-----:|
|1 | 364.3 | 828.7 | 841.6 |
|8 | 2826.8 | 6014.7 | 6230.1(BS=248) |
|32 | - | - | - |

> 其中，AMP 8卡在BatchSize=256时会OOM，因此下调BatchSize为248

## 五、日志数据
- [1卡 FP32 BS=128 日志](./logs/pytorch_gpu1_fp32_bs128.txt)
- [1卡 AMP BS=128 日志](./logs/pytorch_gpu1_amp_bs128.txt)
- [1卡 AMP BS=256 日志](./logs/pytorch_gpu1_amp_bs256.txt)
- [8卡 FP32 BS=128 日志](./logs/pytorch_gpu8_fp32_bs128.txt)
- [8卡 AMP BS=128 日志](./logs/pytorch_gpu8_amp_bs128.txt)
- [8卡 AMP BS=248 日志](./logs/pytorch_gpu8_amp_bs248.txt)
