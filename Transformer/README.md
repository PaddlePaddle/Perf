<!-- omit in toc -->
# Paddle Transformer 性能测试

此处给出了基于 Paddle 框架实现的 Transformer 任务的训练性能详细测试报告，包括执行环境、Paddle 版本、环境搭建、复现脚本、测试结果和测试日志。

相同环境下，其他深度学习框架的 Transformer 训练性能数据测试流程，请参考：[OtherReports](./OtherReports)。

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
  - [1.单机（单卡、8卡）日志](#1单机单卡8卡日志)



## 一、测试说明

我们统一使用了 **吞吐能力** 作为衡量性能的数据指标。**吞吐能力** 是业界公认的、最主流的框架性能考核指标，它直接体现了框架训练的速度。

Transformer 模型是机器翻译领域极具代表性的模型。在测试性能时，我们以 **words/sec** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。

测试中，我们选择如下3个维度，测试吞吐性能：

- **卡数**

   本次测试关注1卡、8卡、32卡情况下，模型训练的吞吐性能。选择的物理机是单机8卡配置。
   因此，1卡、8卡测试在单机下完成。32卡在4台机器下完成。

- **训练精度**

   FP32/AMP/FP16 是业界框架常用的精度训练模式，也是衡量框架性能的混合精度量化训练的重要维度。
   本次测试分别对 FP32/AMP/FP16 精度模式进行了测试。


- **BatchSize**

   本次测试，结合各框架具体情况，BatchSize(max_tokens)选用如下：
   | 参数 | PaddlePaddle | NGC PyTorch |
   |:-----:|:-----:|:-----:|
   | FP32 | 2560 | 2560 |
   | AMP | 5120 | 5120 |
   | FP16 | 5120 | 5120 |

  
关于其它一些参数的说明：

- **XLA**

   本次测试的原则是测试 Transformer 在 Paddle 下的最好性能表现，同时对比其与其它框架最好性能表现的优劣。

   因此，对于支持 XLA 的框架，我们默认打开 XLA 模式，以获得该框架最好的吞吐性能数据。

## 二、环境介绍
### 1.物理机环境

  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

- **镜像版本**: `hub.baidubce.com/paddlepaddle/paddle-benchmark:cuda10.1-cudnn7-runtime-ubuntu16.04`
- **Paddle 版本**: `develop+0f1fde51021e1c9deae099ee0c875c53128687b4`
- **模型代码**：[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop)
- **CUDA 版本**: `10.1`
- **cuDnn 版本:** `7.6.5`


## 三、环境搭建

各深度学习框架在公开的 Github 仓库中给出了详细的docker镜像和构建脚本，具体搭建流程请参考：[此处](./OtherReports)。

如下是 Paddle 测试环境的具体搭建流程:

- **拉取代码**
  ```bash
  git clone https://github.com/PaddlePaddle/models.git
  cd models && git checkout 643fd690044c249a5108d8c979f4025af50962e1
  ```


- **构建镜像**

   ```bash
   # 拉取镜像
   docker pull hub.baidubce.com/paddlepaddle/paddle-benchmark:cuda10.1-cudnn7-runtime-ubuntu16.04

   # 创建并进入容器
   nvidia-docker run --name=test_transformer_paddle -it \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v $PWD:/workspace/models \
    hub.baidubce.com/paddlepaddle/paddle-benchmark:cuda10.1-cudnn7-runtime-ubuntu16.04 /bin/bash
   ```

- **安装依赖**
   ```bash
   # 安装 PaddleNLP 中依赖库
   pip3.7 install -r PaddleNLP/requirements.txt
   ```

- **准备数据**

   Transformer 模型是基于 [WMT'14 EN-DE 数据集](https://dumps.wikimedia.org/) 进行的训练的。在代码的reader.py中会自动下载数据集。

## 四、测试步骤

transformer测试目录位于`/workspace/models/PaddleNLP/benchmark/transformer`。详细的测试方法在该目录已写明。
根据测试的精度，需要调整configs/transformer.base.yaml中的参数。
| 精度 | batch_size | use_amp | use_pure_fp16 |
|:-----:|:-----:|:-----:|:-----:|
| FP32 | 2560 | False | False |
| AMP | 5120 | True | False |
| FP16 | 5120 | True | False |


## 五、测试结果

### 1.Paddle训练性能

- 训练吞吐率(words/sec)如下:

   |卡数 | FP32(BS=2560) | AMP(BS=5120) | FP16(BS=5120) |
   |:-----:|:-----:|:-----:|:-----:|
   |1 | |  |  |
   |8 |  |  |  |
   |32 |  |  |  |

### 2.与业内其它框架对比

- 说明：
  - 同等执行环境下测试
  - 单位：`words/sec`
  - BatchSize FP32下选择 2560, AMP、FP16下选择 5120


- FP32测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=2560 |  |  |
  | GPU=8,BS=2560 |  |  |
  | GPU=32,BS=2560 | 194040.4 | 166352.6 |


- AMP测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=5120 |  |  |
  | GPU=8,BS=5120 |  |  |
  | GPU=32,BS=5120 | 613864.5 | 385625.7 |


- FP16测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=5120 |  |  |
  | GPU=8,BS=5120 |  |  |
  | GPU=32,BS=5120 | 678315.9 | 590188.7 |


## 六、日志数据
### 1.单机（单卡、8卡）日志
- [4机32卡、FP32](./logs/paddle_gpu32_fp32_bs2560)
- [4机32卡、FP16](./logs/paddle_gpu32_fp16_bs5120)
- [4机32卡、AMP ](./logs/paddle_gpu32_amp_bs5120)
