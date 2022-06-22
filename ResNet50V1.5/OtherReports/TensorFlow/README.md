# NGC TensorFlow ResNet50V1.5 性能测试

此处给出了基于 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) 实现的 ResNet50V1.5 任务的详细复现流程，包括环境介绍、环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC TensorFlow ResNet50V1.5 性能测试](#ngc-tensorflow-resnet50v15-性能测试)
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
  - Driver Version: 470.83.01
  - 内存：630 GB

- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

我们使用 NGC TensorFlow 的代码仓库提供的Dockerfile制作镜像：

- Docker: nvcr.io/nvidia/tensorflow:20.12-tf1-py3
- TensorFlow：1.15.4+nv
- 模型代码：[NVIDIA/DeepLearningExamples/TensorFLow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5)
- CUDA：11
- cuDNN：8.1.1

## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

单机环境的搭建，我们遵循了 NGC TensorFlow 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：

- 以ImageNet2012数据集为基础制作TF_Record格式的数据。

  这部分不在本报告中详细展开，可参考NGC提供的[文档](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide)制作。

- 下载NGC TensorFlow repo,并进入目录

   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/TensorFlow/Classification/ConvNets
   # 本次测试是在如下版本下完成的：
   git checkout 4a15e9146a6516941ba3ae146621a5c94e4bc431
   ```

- 制作Docker镜像

   ```bash
   docker build . -t nvidia_rn50_tf
   ```

- 启动Docker

   ```bash
   # 假设制作好的TF_Record数据放在<path to tfrecords data>目录下
   nvidia-docker run --rm -it -v <path to tfrecords data>:/data/tfrecords --ipc=host nvidia_rn50_tf
   ```

### 2.多机（32卡）环境搭建

- IB配置(可选）
请参考[这里](../../../utils/ib.md)
	
- MPI配置
请参考[这里](../../../utils/mpi.md)

## 三、测试步骤

### 1.单机（单卡、8卡）测试

对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现，对其提供的代码未做改动，并严格按照NGC测试使用的参数配置测试。其公开的测试报告请见：[《ResNet-50 v1.5 for TensorFlow》](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5)

- 修改`/workspace/rn50v15_tf/runtime/runner.py`文件:

修改前代码：
  ```
  config.inter_op_parallelism_threads = max(2, (multiprocessing.cpu_count() // max(hvd.size(), 8) - 2))
  ```
修改后代码：
  ```
  config.inter_op_parallelism_threads = 4
  ```

- 下载我们编写的测试脚本，并执行该脚本

   ```bash
   cd scripts/
   bash tf_test_all.sh
   ```

- 执行后将得到如下日志文件：

   ```bash
   /log/tf_gpu1_amp_bs256.txt
   /log/tf_gpu1_fp32_bs256.txt
   /log/tf_gpu8_amp_bs256.txt
   /log/tf_gpu8_fp32_bs256.txt
   ```

由于NGC TensorFlow的测试使用的是`training_perf.sh`，因此我们提供的`tf_test_all.sh`是参考了`training_perf.sh`的参数设置方法。

### 2.多机（32卡）测试

使用[`$mpirun`](../../../utils/mpi.md#通信框架可以从MPI中获取信息) 命令启动多机训练进程，例如:

```
echo "runing fp32 256"
$mpirun  python main.py --mode=training_benchmark \
	--use_xla \
	--warmup_steps 200 \
	--num_iter 500 \
	--iter_unit batch \
	--batch_size 256 \
	--data_dir=./data/tfrecords/ \
	--results_dir=./results/gpu32_fp32_bs256
```

## 四、测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=256) | AMP(BS=256)|
|:-----:|:-----:|:-----:|
|1 | 414.73  | 1173.38 |
|8 | 3275.93 | 9310.31 |
|32 |12671.9 | 33317.67 |

## 五、日志数据
- [1卡 FP32 BS=256 日志](./logs/tf_gpu1_fp32_bs256.txt)
- [1卡 AMP BS=256 日志](./logs/tf_gpu1_amp_bs256.txt)
- [8卡 FP32 BS=256 日志](./logs/tf_gpu8_fp32_bs256.txt)
- [8卡 AMP BS=256 日志](./logs/tf_gpu8_amp_bs256.txt)
- [32卡 FP32 BS=256 日志](./logs/tf_gpu32_fp32_bs256.txt)
- [32卡 AMP BS=256 日志](./logs/tf_gpu32_amp_bs256.txt)
