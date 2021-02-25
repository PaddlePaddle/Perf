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

在测试性能时，我们以 **words/sec** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。

测试中，我们选择如下3个维度，测试吞吐性能：

- **卡数**

   本次测试关注1卡、8卡、32卡情况下，模型训练的吞吐性能。选择的物理机是单机8卡配置。
   因此，1卡、8卡测试在单机下完成。32卡在4台机器下完成。

- **FP32/AMP**

   FP32 和 AMP 是业界框架均支持的两种精度训练模式，也是衡量框架性能的混合精度量化训练的重要维度。
   本次测试分别对 FP32 和 AMP 两种精度模式进行了测试。


- **BatchSize**

   本次测试，结合各框架具体情况，BatchSize 选用如下：

   | 参数 | PaddlePaddle | NGC PyTorch |
   |:-----:|:-----:|:-----:|
   | FP32 | 2560 | 2560 |
   | AMP | 5120 | 5120 |

关于其它一些参数的说明：

- **XLA**

   本次测试的原则是测试 Transformer 在 Paddle 下的最好性能表现，同时对比其与其它框架最好性能表现的优劣。

   因此，对于支持 XLA 的框架，我们默认打开 XLA 模式，以获得该框架最好的吞吐性能数据。

- **优化器**

   在 Transformer 中，各个框架使用的优化器略有不同。NGC PyTorch 均支持 LAMBOptimizer，PaddlePaddle 默认使用的是 AdamOptimizer。LAMBOptimizer 优化器由于支持**梯度聚合策略**，在多机参数更新时，通信开销更低，性能会比原生的 AdamOptimizer 优化器更好一些。

   此处我们以各个框架默认使用的优化器为准，并测试模型的吞吐性能。Paddle 后续也会支持性能更优的 LAMBOptimizer 优化器。

## 二、环境介绍
### 1.物理机环境

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

- **镜像版本**: `hub.baidubce.com/paddlepaddle/paddle-benchmark:cuda10.1-cudnn7-runtime-ubuntu16.04`
- **Paddle 版本**: `develop+613c46bc0745c8069c55686aef4adc775f9e27d1`
- **模型代码**：[PaddleNLP](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP)
- **CUDA 版本**: `10.1`
- **cuDnn 版本:** `7.6.5`


## 三、环境搭建

各深度学习框架在公开的 Github 仓库中给出了详细的docker镜像和构建脚本，具体搭建流程请参考：[此处](./OtherReports)。

如下是 Paddle 测试环境的具体搭建流程:

- **拉取代码**
  ```bash
  git clone https://github.com/PaddlePaddle/models.git
  cd models && git checkout 5b4aef8ecef2c6f9a4ec81652a4138c623a754ba
  ```


- **构建镜像**

   ```bash
   # 拉取镜像
   docker pull hub.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82

   # 创建并进入容器
   nvidia-docker run --name=test_transformer_paddle -it \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v $PWD:/workspace/models \
    hub.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82 /bin/bash
   ```

- **安装依赖**
   ```bash
   # 安装 PaddleNLP 中依赖库
   pip3.7 install -r PaddleNLP/requirements.txt
   # 还需要额外安装两个依赖库
   pip3.7 install attrdict
   pip3.7 install seqeval
   ```

- **准备数据**

   训练脚本会自动下载数据集到目录 `/root/.paddlenlp/datasets/machine_translation/WMT14ende/WMT14.en-de.tar.gz`
   ```

## 四、测试步骤

### 1.单机单卡测试

- **FP32 启动命令：**
```
export CUDA_VISIBLE_DEVICES=0 & nohup python3.7 train.py > ./logs/transformer_bs2560_fp32_gpu1.log 2>&1 &
```

需要修改 `../configs/transformer.big.yaml` 的参数 `use_amp = False`。

- **AMP 启动命令：**
```
export CUDA_VISIBLE_DEVICES=0 & nohup python3.7 train.py > ./logs/transformer_bs5120_fp16_gpu1.log 2>&1 &
```

需要修改 `../configs/transformer.big.yaml` 的参数 `use_amp = True`。

## 五、测试结果

### 1.Paddle训练性能

- 训练吞吐率(sequences/sec)如下:

   |卡数 | FP32(BS=2560) | AMP(BS=5120) |
   |:-----:|:-----:|:-----:|
   |1 | ~ | ~ |

### 2.与业内其它框架对比

- 说明：
  - 同等执行环境下测试
  - 单位：`words/sec`
  - BatchSize FP32下统一选择 2560、AMP下统一选择 5120


- FP32测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|:-----:|
  | GPU=1,BS=2560 | ~ | ~ |


- AMP测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=5120 | ~ | ~ |
  

## 六、日志数据
### 1.单机（单卡、8卡）日志

- [单卡 bs=2560、FP32](./logs/transformer_bs2560_fp32_gpu1.log)
- [单卡 bs=5120、AMP](./logs/transformer_bs5120_amp_gpu1.log)
