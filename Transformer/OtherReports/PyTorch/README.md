<!-- omit in toc -->
# NGC PyTorch Transformer 性能复现


此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) 实现的 Transformer 任务的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境介绍](#一环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [二、环境搭建](#二环境搭建)
  - [1. 单机（单卡、8卡）环境搭建](#1-单机单卡8卡环境搭建)
  - [2. 多机（32卡）环境搭建](#2-多机32卡环境搭建)
- [三、测试步骤](#三测试步骤)
  - [1. 单机（单卡、8卡）测试](#1-单机单卡8卡测试)
  - [2. 多机（32卡）测试](#2-多机32卡测试)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
  - [1.单机（单卡、8卡）日志](#1单机单卡8卡日志)


## 一、环境介绍

### 1.物理机环境

我们使用了同一个物理机环境，对 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) 的 Transformer 模型进行了测试，详细物理机配置，见[Paddle Transformer 性能测试](../../README.md#1.物理机环境)。

### 2.Docker 镜像

NGC PyTorch 的代码仓库提供了自动构建 Docker 镜像的的 [shell 脚本](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/scripts/docker/build.sh)，

- **镜像版本**: `nvcr.io/nvidia/pytorch:20.06-py3`
- **PyTorch 版本**: `1.6.0a0+9907a3e`
- **CUDA 版本**: `11.0`
- **cuDnn 版本**: `8.0.1`

## 二、环境搭建

### 1. 单机单卡环境搭建

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#quick-start-guide) 教程搭建了测试环境，主要过程如下：

- **拉取代码**

    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/Translation/Transformer
    # 本次测试是在如下版本下完成的：
    git checkout 99b1c898cead5603c945721162270c2fe077b4a2
    ```

- **构建镜像**

    ```bash
    bash scripts/docker/build.sh   # 构建镜像
    bash scripts/docker/launch.sh  # 启动容器
    ```

- **准备数据**

    NGC PyTorch 提供单独的数据下载和预处理脚本 [scripts/run_preprocessing.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/scripts/run_preprocessing.sh)。在容器中执行如下命令，可以下载和制作 `WMT14 English-German` 数据集。

    ```bash
    bash scripts/run_preprocessing.sh
    ```

## 三、测试步骤

### 1. 单机单卡测试

- **FP32 训练命令：**

    若测试单机单卡 batch_size=2560、FP32 的训练性能，执行如下命令：

    ```
    RESULTS_DIR='/results'
    CHECKPOINTS_DIR='/results/checkpoints'
    STAT_FILE=${RESULTS_DIR}/run_log.json
    mkdir -p $CHECKPOINTS_DIR

    python /workspace/translation/train.py \
      /data/wmt14_en_de_joined_dict \
        --arch transformer_wmt_en_de_big_t2t \
        --share-all-embeddings \
        --optimizer adam \
        --adam-betas '(0.9, 0.997)' \
        --adam-eps "1e-9" \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 0.0 \
        --warmup-updates 4000 \
        --lr 0.0006 \
        --min-lr 0.0 \
        --dropout 0.1 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 2560 \
        --seed 1 \
        --max-epoch 1 \
        --fuse-layer-norm \
        --log-interval 500 \
        --save-dir ${CHECKPOINTS_DIR} \
        --stat-file ${STAT_FILE} \
    ```

- **AMP O2 训练命令：**

    若测试单机单卡 batch_size=5120、AMP O2 的训练性能，执行如下命令：

    ```
    RESULTS_DIR='/results'
    CHECKPOINTS_DIR='/results/checkpoints'
    STAT_FILE=${RESULTS_DIR}/run_log.json
    mkdir -p $CHECKPOINTS_DIR

    python /workspace/translation/train.py \
      /data/wmt14_en_de_joined_dict \
      --arch transformer_wmt_en_de_big_t2t \
      --share-all-embeddings \
      --optimizer adam \
      --adam-betas '(0.9, 0.997)' \
      --adam-eps "1e-9" \
      --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt \
      --warmup-init-lr 0.0 \
      --warmup-updates 4000 \
      --lr 0.0006 \
      --min-lr 0.0 \
      --dropout 0.1 \
      --weight-decay 0.0 \
      --criterion label_smoothed_cross_entropy \
      --label-smoothing 0.1 \
      --max-tokens 5120 \
      --seed 1 \
      --max-epoch 1 \
      --fuse-layer-norm \
      --amp \
      --amp-level O2 \
      --log-interval 500 \
      --save-dir ${RESULTS_DIR} \
      --stat-file ${STAT_FILE} \
    ```

## 四、测试结果

> 单位： tokens/sec

|卡数 | FP32(BS=2560) | AMP O2(BS=5120) |
|:-----:|:-----:|:-----:|
|1 | 7893.1 | 30523.5 |

## 五、日志数据
### 1.单机（单卡、8卡）日志

- [单卡 bs=2560、FP32](./logs/transformer.pyt_transformer_fp32_bs2560_gpu1.log)
- [单卡 bs=5120、AMP O2](./logs/transformer.pyt_transformer_amp_O2_bs5120_gpu1.log)
