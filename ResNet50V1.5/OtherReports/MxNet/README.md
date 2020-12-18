# NGC MxNet ResNet50V1.5 性能测试

此处给出了基于 [NGC MxNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) 实现的 ResNet50V1.5 任务的详细复现流程，包括环境介绍、环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC MxNet ResNet50V1.5 性能测试](#ngc-mxnet-resnet50v15-性能测试)
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

我们使用 NGC MxNet 的代码仓库提供的Dockerfile制作镜像：

- Docker: nvcr.io/nvidia/mxnet:19.07-py3
- MxNet：1.5.0
- 模型代码：[NVIDIA/DeepLearningExamples/MxNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5)
- CUDA：10.1
- cuDNN：7.6.1


## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

单机环境的搭建，我们遵循了 NGC MxNet 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：

- 下载NGC MxNet repo,并进入目录
   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/MxNet/Classification/RN50v1.5
   # 本次测试是在如下版本下完成的：
   git checkout 99b1c898cead5603c945721162270c2fe077b4a2
   ```

- 制作Docker镜像
   ```bash
   docker build . -t nvidia_rn50_mxnet
   ```

- 启动Docker
   ```bash
   # 假设imagenet数据放在<path to data>目录下
   # 假设MxNet转化后的数据放在<path to rec data>
   nvidia-docker run --rm -it --ipc=host -v <path to data>:/ILSVRC2012 -v <path to mxnet data>:/data/imagenet/train-val-recordio-passthrough nvidia_rn50_mxnet
   ```

- 制作MxNet需要的rec格式数据
   ```bash
   ./scripts/prepare_imagenet.sh /ILSVRC2012 /data/imagenet/train-val-recordio-passthrough
   ```

### 2.多机（32卡）环境搭建

> 我们会后续补充此内容。

## 三、测试步骤

### 1.单机（单卡、8卡）测试

对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现，对其提供的代码未做改动，并严格按照NGC测试使用的参数配置测试。其公开的测试报告请见：[《ResNet50 v1.5 for MXNet》](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5)

- 下载我们编写的测试脚本，并执行该脚本
   ```bash
   wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/OtherReports/MxNet/scripts/mxnet_test_all.sh
   bash mxnet_test_all.sh
   ```

- 执行后将得到如下日志文件：
   ```bash
   /log/mxnet_gpu1_gpu8_fp32_bs96.txt
   /log/mxnet_gpu1_gpu8_amp_bs128.txt
   /log/mxnet_gpu1_gpu8_amp_bs192.txt
   ```

在NGC报告的[Training performance: NVIDIA DGX-2 (16x V100 32G)](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5#training-performance-nvidia-dgx-2-16x-v100-32g)小节，提供了其测试的参数配置。因此，我们提供的`mxnet_test_all.sh`是参考了其文档中的配置。

### 2.多机（32卡）测试

> 我们会后续补充此内容。

## 四、测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=96) | AMP(BS=128) | AMP(BS=192)|
|:-----:|:-----:|:-----:|:-----:|
|1 | 387.1 | 1380.6 | 1447.6 |
|8 | 2998.1 | 9218.9 | 9765.6 |
|32 | - | - | - |

## 五、日志数据
- [1卡、8卡 FP32 BS=96 日志](./logs/mxnet_gpu1_gpu8_fp32_bs96.txt)
- [1卡、8卡 AMP BS=128 日志](./logs/mxnet_gpu1_gpu8_amp_bs128.txt)
- [1卡、8卡 AMP BS=192 日志](./logs/mxnet_gpu1_gpu8_amp_bs192.txt)
