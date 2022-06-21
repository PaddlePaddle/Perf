<!-- omit in toc -->
# Paddle Perf——Paddle框架性能测试


本 repo 用于公开 PaddlePaddle 开源实现的各个学术界、工业界前沿模型，在训练期间的性能数据，同时提供了各模型性能测试的详细复现流程，以供参考。

同时，我们也在相同的硬件执行环境下，按照业内其它知名深度学习框架公开的代码和教程，测试了对应模型的性能数据，并记录具体日志和数据。

<!-- omit in toc -->
## 目录

- [一、测试模型](#一测试模型)
  - [1.计算机视觉](#1计算机视觉)
  - [2.自然语言处理](#2自然语言处理)
- [二、供对比的业内深度学习框架](#二供对比的业内深度学习框架)
  - [1. NGC TensorFlow 1.15](#1-ngc-tensorflow-115)
  - [2. NGC PyTorch](#2-ngc-pytorch)
  - [3. NGC MxNet](#3-ngc-mxnet)
- [三、测试结果](#三测试结果)
  - [1. ResNet50V1.5](#1-resnet50v15)
  - [2. Bert Base Pre-Training](#2-bert-base-pre-training)

## 一、测试模型

目前我们公开了**计算机视觉**和**自然语言处理**领域的两个典型模型的性能对比数据：

### 1.计算机视觉
- [ResNet50V1.5](./ResNet50V1.5)

### 2.自然语言处理
- [Bert Base Pre-Training](./Bert)

我们将持续开展性能测试工作，后续将逐步公开更多性能数据，敬请期待。

## 二、供对比的业内深度学习框架

我们选择了 NGC 优化后的 TensorFlow、PyTorch、MxNet，作为性能的参考。

对这些框架的性能测试，我们选用相同的物理机执行，并严格参照各框架官网公布的测试方法进行复现。

### 1. [NGC TensorFlow 1.15](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags)

- 代码库：[DeepLearningExamples/TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow)

### 2. [NGC PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)

- 代码库：[DeepLearningExamples/PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch)
### 3. [NGC MxNet](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet/tags)

- 代码库：[DeepLearningExamples/MxNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet)


## 三、测试结果

说明：

- 本次测试选用`8 * V100-SXM2-16GB`物理机做单机单卡、单机8卡测试；选用4台`8 * V100-SXM2-32GB`物理机做32卡测试。
- 测试中，我们尽可能复现不同框架的最好性能，因此以下测试结果默认打开了各个框架的各种加速功能/选项，如：
   - 对于支持 `DALI/XLA` 的框架，以下测试为开启 `DALI/XLA` 的数据

### 1. ResNet50V1.5
> 详细数据请见[《Paddle ResNet50V1.5 性能测试报告》](./ResNet50V1.5)

- **单位**：`images/sec`

- FP32测试

  | 参数 | [PaddlePaddle](./ResNet50V1.5) | [NGC TensorFlow 1.15](./ResNet50V1.5/OtherReports/TensorFlow) | [NGC PyTorch](./ResNet50V1.5/OtherReports/PyTorch) | [NGC MXNet](./ResNet50V1.5/OtherReports/MxNet) |
  |:-----:|:-----:|:-----:|:-----:|:-----:|
  | GPU=1,BS=256 | 389.12  | 414.73 | 363.77 | 380.55 |
  | GPU=8,BS=256 | 2946.3 | 3275.93 | 2787.43 | 3002.8 |
  | GPU=32,BS=128 | 11366.6 | 11622.9 | 10393.2 | - |

- AMP测试

  | 参数 | [PaddlePaddle](./ResNet50V1.5) | [NGC TensorFlow 1.15](./ResNet50V1.5/OtherReports/TensorFlow) | [NGC PyTorch](./ResNet50V1.5/OtherReports/PyTorch) | [NGC MXNet](./ResNet50V1.5/OtherReports/MxNet) |
  |:-----:|:-----:|:-----:|:-----:|:-----:|
  | GPU=1,BS=256 | 1444.94 | 1173.38 | 773.30 | 1359.4 |
  | GPU=8,BS=256 | 10649.9 | 9310.31 | 5811.26 | 10494 |
  | GPU=32,BS=256 | 34302.5 | 33695.0 | 21588.1 | -<sup>[BS=192]</sup> |

  > 关于torch数据，按照官方文档反复重测了多次未达到官方的标准。若了解相关原因，欢迎issue我们。 <br>

### 2. Bert Base Pre-Training
> 详细数据请见[《Paddle Bert Base 性能测试报告》](./Bert)

- **max_seq_len**: `128`
- **单位**：`sequences/sec`

- FP32测试

  | 参数 | [PaddlePaddle](./Bert) | [NGC TensorFlow 1.15](./Bert/OtherReports/TensorFlow) | [NGC PyTorch](./Bert/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|:-----:|
  | GPU=1,BS=96 | 161.15  | 156.33  | 153.56 |
  | GPU=8,BS=96 | 1288.50   | 1231.74 | 1228.24 |
  | GPU=32,BS=32 | 4862.5 | 4379.4 | 3994.1 |
  | GPU=32,BS=48 | 4948.4 | 4723.5 | 3974.0 |

- AMP测试

  | 参数 | [PaddlePaddle](./Bert) | [NGC TensorFlow 1.15](./Bert/OtherReports/TensorFlow) | [NGC PyTorch](./Bert/OtherReports/PyTorch) |
  |:-----:|:-----:|:-----:|:-----:|
  | GPU=1,BS=96 | 653.97 | 530.28  | 630.61 |
  | GPU=8,BS=96 | 5234.17  | 4181.32 | 5044.04 |
  | GPU=32,BS=64 | 18432.2 | 14773.4 | 15941.1 |
  | GPU=32,BS=96 | 18480.0 | 16554.3 | 16311.6 |
