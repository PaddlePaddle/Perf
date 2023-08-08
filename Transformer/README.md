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
   本次测试分别对 FP32/FP16 精度模式进行了测试。


- **BatchSize**

   本次测试，结合各框架具体情况，BatchSize(max_tokens)选用如下：
   | 参数 | PaddlePaddle | NGC PyTorch |
   |:-----:|:-----:|:-----:|
   | FP32 | 5120 | 5120 |
   | FP16 | 5120 | 5120 |

  
关于其它一些参数的说明：

- **XLA**

   本次测试的原则是测试 Transformer 在 Paddle 下的最好性能表现，同时对比其与其它框架最好性能表现的优劣。

   因此，对于支持 XLA 的框架，我们默认打开 XLA 模式，以获得该框架最好的吞吐性能数据。

## 二、环境介绍
### 1.物理机环境

- 单机V100（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - Driver Version: 525.60.11
  - 内存：630 GB
- 单机A100（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：NVIDIA A100-SXM4-40GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 160
  - Driver Version: 525.60.13
  - 内存：1510 GB
- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

- **镜像版本**: `paddlepaddle/paddle-benchmark:2.5.0-cuda11.2-cudnn8-runtime-ubuntu16.04`
- **Paddle 版本**: `2.5.0.post112`
- **模型代码**：[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
- **CUDA 版本**: `11.2`
- **cuDnn 版本:** `8.2`


## 三、环境搭建

各深度学习框架在公开的 Github 仓库中给出了详细的docker镜像和构建脚本，具体搭建流程请参考：[此处](./OtherReports)。

如下是 Paddle 测试环境的具体搭建流程:

- **拉取代码**
  ```bash
  git clone https://github.com/PaddlePaddle/PaddleNLP.git
  cd PaddleNLP && git checkout 537665a0af2e8a8ef3e539d2d0fae810c3a12ce1
  ```


- **构建镜像**

   ```bash
   # 拉取镜像
   docker pull paddlepaddle/paddle-benchmark:2.5.0-cuda11.2-cudnn8-runtime-ubuntu16.04

   # 创建并进入容器
   nvidia-docker run --name=test_transformer_paddle -it \
    --net=host \
    --shm-size=30g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v $PWD:/workspace/ \
    paddlepaddle/paddle-benchmark:2.5.0-cuda11.2-cudnn8-runtime-ubuntu16.04 /bin/bash
   ```

- **安装依赖**
   ```bash
   # 安装 PaddleNLP 中依赖库
   pip install -r requirements.txt
   pip install paddlenlp
   ```

- **准备数据**

   Transformer 模型是基于 [WMT'14 EN-DE 数据集](https://dumps.wikimedia.org/) 进行的训练的。在代码的reader.py中会自动下载数据集。

## 四、测试步骤

transformer测试目录位于`/PaddleNLP/tests`。
根据测试的精度，需要调整/workspace/configs/transformer.big.yaml中的参数。
| 精度 | batch_size  | use_pure_fp16 |
|:-----:|:-----:|:-----:|
| FP32 | 5120  | False |
| FP16 | 5120  | True |
### 1.单机（单卡、8卡）测试
为了更方便地复现我们的测试结果，我们提供了一键测试 benchmark 数据的脚本 run_benchmark.sh ，需放在/PaddleNLP/tests目录下。
- **测试脚本**
   ```bash
   #!/bin/bash   
    base_batch_size=${1:-"2"}       
    fp_item=${2:-"fp32"}            # (必选) fp32|pure_fp16
    device_num=${3:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    max_iter=${4:-500}              
    batch_size=${base_batch_size}   
    static_scripts="../examples/machine_translation/transformer/static/"
    config_file="transformer.big.yaml"

    if [ ${fp_item} == "pure_fp16" ]; then
        sed -i "s/^use_amp.*/use_amp: True/g" ${static_scripts}/../configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: True/g" ${static_scripts}/../configs/${config_file}
    elif [ ${fp_item} == "fp32" ]; then
        sed -i "s/^use_amp.*/use_amp: False/g" ${static_scripts}/../configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: False/g" ${static_scripts}/../configs/${config_file}
    else
        echo " The fp_item should be fp32 pure_fp16 "
    fi
    sed -i "s/^max_iter.*/max_iter: ${max_iter}/g" ${static_scripts}/../configs/${config_file}
    sed -i "s/^batch_size:.*/batch_size: ${base_batch_size}/g" ${static_scripts}/../configs/${config_file}

    train_cmd="--config ${static_scripts}/../configs/${config_file} --train_file ${static_scripts}/../train.en ${static_scripts}/../train.de --dev_file ${static_scripts}/../dev.en ${static_scripts}/../dev.de --vocab_file ${static_scripts}/../vocab_all.bpe.33712 --unk_token <unk> --bos_token <s> --eos_token <e> --benchmark"
    if [[ ${device_num} = "N1C1" ]];then
        export CUDA_VISIBLE_DEVICES=0;
        sed -i "s/^is_distributed.*/is_distributed: False/g" ${static_scripts}/../configs/${config_file}
        train_cmd="python -u ${static_scripts}/train.py ${train_cmd}" 
    else
        sed -i "s/^is_distributed.*/is_distributed: True/g" ${static_scripts}/../configs/${config_file}
        rm -rf ./mylog
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES \
                  ${static_scripts}/train.py --distributed ${train_cmd}" 
    fi
    echo "train_cmd: ${train_cmd} "
    timeout 25m ${train_cmd} 

   ```
- **单卡启动脚本**  

  若测试单机单卡 batch_size=5120 FP32 的训练性能，执行如下命令：  

    ```bash
        bash  run_benchmark.sh 5120 fp32 N1C1
    ```
- **8卡启动脚本**  

  若测试单机8卡 batch_size=5120 FP16 的训练性能，执行如下命令：  
    ```bash
        bash  run_benchmark.sh 5120 pure_fp16 N1C8
    ```

## 五、测试结果

### 1.Paddle训练性能

- V100 上训练吞吐率(words/sec)如下:

   |卡数 | FP32(BS=5120) | FP16(BS=5120) |
   |:-----:|:-----:|:-----:|
   |1 | 8809.798 | 33082.12 (O2) | 
   |8 | 63436.463   | 225027.075 (O2) | 
   |32 | 194040.4 | 678315.9 |

- A100 上训练吞吐率(words/sec)如下:

   |卡数 | FP32(BS=5120) | FP16(BS=5120) |
   |:-----:|:-----:|:-----:|
   |1 | 40723.75 | 71401.051 (O2) | 
   |8 | 279494.693 | 437351.427 (O2) | 

### 2.与业内其它框架对比

- 说明：
  - 同等执行环境下测试
  - 单位：`words/sec`
  - BatchSize FP32、FP16下选择 5120


- V100 FP32测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=5120 | 8809.798 | 9040.48  |
  | GPU=8,BS=5120 | 63436.463  | 65010.7  |
  | GPU=32,BS=5120 | 183830.0 | 166352.6 |


- V100 FP16测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=5120 | 33082.12 (O2) | 32525.6  |
  | GPU=8,BS=5120 | 225027.075 (O2) | 208959  |
  | GPU=32,BS=5120 | 682820.5 | 590188.7 |

- A100 FP32测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=5120 | 40723.75 | 37458.4  |
  | GPU=8,BS=5120 | 279494.693 | 255375   |


- A100 FP16测试

  | 参数 | [PaddlePaddle](./Transformer) | [NGC PyTorch](./Transformer/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|
  | GPU=1,BS=5120 | 71401.051 (O2) | 43989.5 |
  | GPU=8,BS=5120 | 437351.427 (O2)| 286005 |


## 六、日志数据
### 1.单机（单卡、8卡）日志
- [V100-单机单卡、FP32](./logs/V100_LOG/PaddleNLP_transformer_big_bs5120_fp32_DP_N1C1_log)
- [V100-单机八卡、FP32](./logs/V100_LOG/PaddleNLP_transformer_big_bs5120_fp32_DP_N1C8_log)
- [V100-单机单卡、FP16](./logs/V100_LOG/PaddleNLP_transformer_big_bs5120_pure_fp16_DP_N1C1_log)
- [V100-单机八卡、FP16](./logs/V100_LOG/PaddleNLP_transformer_big_bs5120_pure_fp16_DP_N1C8_log)
- [V100-4机32卡、FP32](./logs/V100_LOG/paddle_gpu32_fp32_bs2560)
- [V100-4机32卡、FP16](./logs/V100_LOG/paddle_gpu32_fp16_bs5120)
- [V100-4机32卡、AMP ](./logs/V100_LOG/paddle_gpu32_amp_bs5120)
- [A100-单机单卡、FP32](./logs/A100_LOG/PaddleNLP_transformer_big_bs5120_fp32_DP_N1C1_log)
- [A100-单机八卡、FP32](./logs/A100_LOG/PaddleNLP_transformer_big_bs5120_fp32_DP_N1C8_log)
- [A100-单机单卡、FP16](./logs/A100_LOG/PaddleNLP_transformer_big_bs5120_pure_fp16_DP_N1C1_log)
- [A100-单机八卡、FP16](./logs/A100_LOG/PaddleNLP_transformer_big_bs5120_pure_fp16_DP_N1C8_log)
