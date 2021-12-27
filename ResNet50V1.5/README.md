<!-- omit in toc -->
# Paddle ResNet50V1.5 性能测试

此处给出了Paddle ResNet50V1.5的详细测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在同等执行环境下，业内几个知名框架在ResNet50V1.5模型下的性能数据，进行对比。

其他深度学习框架的 ResNet50V1.5 训练性能数据测试流程，请参考：[OtherReports](./OtherReports)。

<!-- omit in toc -->
## 目录
- [一、测试说明](#一测试说明)
- [二、环境介绍](#二环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [三、环境搭建](#三环境搭建)
- [四、测试步骤](#四测试步骤)
  - [1.单机（单卡、8卡）测试](#1单机单卡8卡测试)
  - [2.多机（32卡）测试](#2多机32卡测试)
- [五、测试结果](#五测试结果)
  - [1.Paddle训练性能](#1paddle训练性能)
  - [2.与业内其它框架对比](#2与业内其它框架对比)
- [六、日志数据](#六日志数据)

## 一、测试说明

我们统一使用 **吞吐能力** 作为衡量性能的数据指标。**吞吐能力** 是业界公认的、最主流的框架性能考核指标，它直接体现了框架训练的速度。

Resnet50V1.5 作为计算机视觉领域极具代表性的模型。在测试性能时，我们以 **单位时间内能够完成训练的图片数量（images/sec）** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。

测试中，我们选择如下3个维度，测试吞吐性能：

- **卡数**

   本次测试关注1卡、8卡、32卡情况下，模型的训练吞吐性能。选择的物理机是单机8卡配置。
   因此，1卡、8卡测试在单机下完成。32卡在4台机器下完成。

- **FP32/AMP**

   FP32 和 AMP 是业界框架均支持的两种精度训练模式，也是衡量框架性能的混合精度量化训练的重要维度。
   本次测试分别对 FP32 和 AMP 两种精度模式进行了测试。

- **BatchSize**

   本次测试，结合各框架具体情况，BatchSize选用如下：

   | 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet |
   |:-----:|:-----:|:-----:|:-----:|:-----:|
   | FP32 | 128 | 128 | 128 | 96 |
   | AMP | 128 | 128 | 128 | 128 |
   | AMP | 208 | 254 | 254 | 192 |

关于其它一些参数的说明：
- **DALI**

   DALI 能够提升数据加载的速度，防止数据加载成为训练的瓶颈。因此，本次测试全部在打开 DALI 模式下进行。

- **XLA**

   本次测试的原则是测试 Resnet50V1.5 在 Paddle 下的最好性能表现，同时对比其与其它框架最好性能表现的优劣。

   因此，对于支持 XLA 的框架，我们默认打开 XLA 模式，以获得该框架最好的吞吐性能数据。

## 二、环境介绍
### 1.物理机环境

- 单机（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 40
  - Driver Version: 460.27.04
  - 内存：502 GB


- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

Paddle Docker的基本信息如下：

- Docker: `paddlepaddle/paddle-benchmark:2.2.1-cuda11.2-cudnn8-runtime-ubuntu16.04`
- Paddle：2.2.0.post112
- 模型代码：[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
- CUDA：11.2
- cuDNN：8.1.1

## 三、环境搭建

- 拉取docker
  ```bash
  paddlepaddle/paddle-benchmark:2.2.1-cuda11.2-cudnn8-runtime-ubuntu16.04
  ```

- 启动docker
  ```bash
  # 假设imagenet数据放在<path to data>目录下
  nvidia-docker run --shm-size=64g -it -v <path to data>:/data 
  paddlepaddle/paddle-benchmark:2.2.1-cuda11.2-cudnn8-runtime-ubuntu16.04 /bin/bash
  ```

- 拉取PaddleClas
  ```bash
  git clone https://github.com/PaddlePaddle/PaddleClas.git
  cd PaddleClas
  # 本次测试是在如下版本下完成的：
  git checkout 0f35f706b6345eed0c17870c33bfd5158e771b8f
  ```

- 多机网络部署
  ```bash
  InfiniBand 100 Gb/sec
  nvidia-smi topo -m
  GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  CPU Affinity
  GPU0     X      NV2     NV2     NV1     NV1     NODE    NODE    NODE    NODE    0-19
  GPU1    NV2      X      NV1     NV1     NODE    NV2     NODE    NODE    NODE    0-19
  GPU2    NV2     NV1      X      NV2     NODE    NODE    NV1     NODE    NODE    0-19
  GPU3    NV1     NV1     NV2      X      NODE    NODE    NODE    NV2     NODE    0-19
  GPU4    NV1     NODE    NODE    NODE     X      NV2     NV2     NV1     NODE    0-19
  GPU5    NODE    NV2     NODE    NODE    NV2      X      NV1     NV1     NODE    0-19
  GPU6    NODE    NODE    NV1     NODE    NV2     NV1      X      NV2     NODE    0-19
  GPU7    NODE    NODE    NODE    NV2     NV1     NV1     NV2      X      NODE    0-19
  mlx5_0  NODE    NODE    NODE    NODE    NODE    NODE    NODE    NODE     X

  Legend:

    X    = Self
    SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
    NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
    PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
    PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
    PIX  = Connection traversing at most a single PCIe bridge
    NV#  = Connection traversing a bonded set of # NVLinks
    ```

- 数据部署
  ImageNet数据集位于/data目录，具有如下目录结构：
  ```shell
  /data
   |---train
   |   |---n01440764
   |   |   |---n01440764_10026.jpeg
   |   |   |...
   |   |...
   |---train_list.txt
  ```
  其中，train子目录下包含训练数据集，train_list.txt文件记录各训练数样本的路径和类别标签，内容如下：
  ```shell
  train/n01440764/n01440764_10026.jpeg 0
  train/n01440764/n01440764_10027.jpeg 0
  train/n01440764/n01440764_10029.jpeg 0
  train/n01440764/n01440764_10040.jpeg 0
  ... ...
  ```

## 四、测试步骤

### 1.单机（单卡、8卡）测试

- 下载我们编写的测试脚本，并执行该脚本
  ```bash
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/paddle_test_all.sh
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_1gpu_fp32_bs128.yaml
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_1gpu_amp_bs128.yaml
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_1gpu_amp_bs208.yaml
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_8gpu_fp32_bs128.yaml
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_8gpu_amp_bs128.yaml
  bash paddle_test_all.sh
  ```


- 执行后将得到如下日志文件：
   ```bash
   ./paddle_gpu1_fp32_bs128.txt
   ./paddle_gpu1_amp_bs128.txt
   ./paddle_gpu1_amp_bs208.txt
   ./paddle_gpu8_fp32_bs128.txt
   ./paddle_gpu8_amp_bs128.txt
   ```

### 2.多机（32卡）测试
- 下载我们编写的测试脚本，并执行该脚本(每台机器均需要执行)
  ```bash
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/paddle_test_multi_node_all.sh
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_32gpu_fp32_bs96.yaml
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_32gpu_amp_bs128.yaml
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_32gpu_amp_bs208.yaml
  bash paddle_test_multi_node_all.sh
  ```

- 执行后将得到如下日志文件：
   ```bash
   ./paddle_gpu32_fp32_bs128.txt
   ./paddle_gpu32_amp_bs128.txt
   ./paddle_gpu32_fp32_bs256.txt
   ./paddle_gpu32_amp_bs256.txt
   ```

## 五、测试结果

### 1.Paddle训练性能


- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=128) | AMP(BS=128) | AMP(BS=208)|
|:-----:|:-----:|:-----:|:-----:|
|1 | 339.59 | 1340.47  | 1412.91 |
|8 | 2711.29 | 9926.42  | —— |
|32 | 11366.6 | 29715.2 | 34302.5 |

以上数据是根据PaddleClas日志数据，去掉warmup step后，求平均得出。

### 2.与业内其它框架对比

说明：
- 同等执行环境下测试
- 单位：`images/sec`
- 对于支持 `DALI/XLA` 的框架，以下测试为开启 `DALI/XLA` 的数据

结果：
- FP32测试

  | 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet |
  |:-----:|:-----:|:-----:|:-----:|:-----:|
  | GPU=1,BS=128 | 339.59 | 408.70  | 358.30 | 360.73  <sup>[BS=96]</sup> |
  | GPU=8,BS=128 | 2711.29 ] | 3132.27  | 2768.51  | 2955.30  <sup>[BS=96]</sup> |
  | GPU=32,BS=128 | 11366.6<sup>[BS=96]</sup> | 11622.9 | 10393.2 | - |

- AMP测试

  | 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet |
  |:-----:|:-----:|:-----:|:-----:|:-----:|
  | GPU=1,BS=128 | 1340.47  | 1100.88 | 752.39  | 1277.3   |
  | GPU=1,BS=256 | 1412.91 <sup>[BS=208]</sup> | 1166.72 <sup>[BS=254]</sup> |763.54 <sup>[BS=254]</sup> | 1355.00  <sup>[BS=192]</sup> |
  | GPU=8,BS=128 | 9926.42  | 8434.95  | 5587.61  | 9449.3  |
  | GPU=8,BS=256 | - | 9229.42 <sup>[BS=208]</sup>  | 5687.99 <sup>[BS=244]</sup> | 10318.00 <sup>[BS=192]</sup> |
  | GPU=32,BS=128 | 29715.2 | 27528.0 | 17940.7 | - |
  | GPU=32,BS=256 | 34302.5 | 33695.0 | 21588.1 | - |

> 以上测试，由于显存限制，下调了部分测试的BatchSize，并在表格中注明 <br>
> Pytorch AMP 8卡在BatchSize=256时会OOM，因此下调BatchSize为244, AMP 单卡在BatchSize=256时会OOM，因此下调BatchSize为254

## 六、日志数据
- [1卡 FP32 BS=128 日志](./logs/paddle_gpu1_fp32_bs128.txt)
- [1卡 AMP BS=128 日志](./logs/paddle_gpu1_amp_bs128.txt)
- [1卡 AMP BS=208 日志](./logs/paddle_gpu1_amp_bs208.txt)
- [8卡 FP32 BS=128 日志](./logs/paddle_gpu8_fp32_bs128.txt)
- [8卡 AMP BS=128 日志](./logs/paddle_gpu8_amp_bs128.txt)
- [32卡 FP32 BS=96 日志](./logs/paddle_gpu32_fp32_bs96.txt)
- [32卡 AMP BS=128 日志](./logs/paddle_gpu32_amp_bs128.txt)
- [32卡 AMP BS=208 日志](./logs/paddle_gpu32_amp_bs208.txt)

