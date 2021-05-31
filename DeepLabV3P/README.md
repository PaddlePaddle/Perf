<!-- omit in toc -->
# Paddle DeepLabV3P 性能测试

此处给出了基于 Paddle 框架实现的 DeepLabV3P 任务的训练性能详细测试报告，包括执行环境、Paddle 版本、环境搭建、复现脚本、测试结果和测试日志。

相同环境下，其他深度学习框架的 DeepLabV3P 训练性能数据测试流程，请参考：[OtherReports](./OtherReports)。

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

DeepLabV3P 模型是图像分割领域极具代表性的模型。在测试性能时，我们以 **samples/sec** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。

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
   | FP32 | 2 | 2 |
   | AMP | 4 | 4 |

## 二、环境介绍
### 1.物理机环境

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

- **镜像版本**: `paddlepaddle/paddle-benchmark:cuda11.0-cudnn8-runtime-ubuntu16.04-gcc82`
- **Paddle 版本**: `develop`
- **模型代码**：[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/benchmark)
- **CUDA 版本**: `11.0`
- **cuDnn 版本:** `8.0.5`


## 三、环境搭建

各深度学习框架在公开的 Github 仓库中给出了详细的docker镜像和构建脚本，具体搭建流程请参考：[此处](./OtherReports)。

如下是 Paddle 测试环境的具体搭建流程:

- **拉取代码**
  ```bash
  git clone https://github.com/PaddlePaddle/PaddleSeg.git -b benchmark
  cd PaddleSeg && git checkout 318dd24cf3c7788f3fe88c7ccb910a09e6f469e5
  ```


- **构建镜像**

   ```bash
   # 拉取镜像
   docker pull paddlepaddle/paddle-benchmark:cuda11.0-cudnn8-runtime-ubuntu16.04-gcc82

   # 创建并进入容器
   nvidia-docker run --name=test_transformer_paddle -it \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v $PWD:/workspace/models \
    paddlepaddle/paddle-benchmark:cuda11.0-cudnn8-runtime-ubuntu16.04-gcc82 /bin/bash
   ```

- **安装依赖**
   ```bash
   # 安装 PaddleSeg 中依赖库
   pip install -r requirements.txt
   pip install paddleseg
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

## 四、测试步骤

DeepLabV3P测试目录位于`/workspace/models/`。详细的测试方法在该目录已写明。
根据测试的精度，需要调整命令行参数。

| 精度 | batch_size | fp16 |
|:-----:|:-----:|:-----:|
| FP32 | 2 | NO |
| AMP | 4 | YES | 


## 五、测试结果

### 1.Paddle训练性能

- 训练吞吐率(samples/sec)如下:

   |卡数 | FP32(BS=2) | AMP(BS=4) 
   |:-----:|:-----:|:-----:|
   |1 | -- | -- |
   |8 | --   | -- |
   |32 | -- | -- | 
### 2.与业内其它框架对比

- 说明：
  - 同等执行环境下测试
  - 单位：`samples/sec`
  - BatchSize FP32下选择 2, AMP下选择 4


- FP32测试

  | 参数 | [PaddlePaddle](./PaddleSeg) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=2 | -- | --  |
  | GPU=8,BS=2 | --  | --  |
  | GPU=32,BS=2 | -- | -- |


- AMP测试

  | 参数 | [PaddlePaddle](./PaddleSeg) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=4 | --  | --  |
  | GPU=8,BS=4 | --  | --  |
  | GPU=32,BS=4 | -- | -- |


## 六、日志数据
### 1.单机（单卡、8卡）日志
- [单机单卡、FP32](./logs/paddle_gpu1_fp32_bs2)
- [单机八卡、FP32](./logs/paddle_gpu8_fp32_bs2)
- [4机32卡、FP32](./logs/paddle_gpu32_fp32_bs2)
- [单机单卡、AMP](./logs/paddle_gpu1_amp_bs4)
- [单机八卡、AMP](./logs/paddle_gpu8_amp_bs4)
- [4机32卡、AMP ](./logs/paddle_gpu32_amp_bs4)
