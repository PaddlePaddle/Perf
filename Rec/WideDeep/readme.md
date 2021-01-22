<!-- omit in toc -->

# Paddle Wide&Deep 性能测试

此处给出了Paddle Wide&Deep 的详细测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在同等执行环境下，业内几个知名框架在Wide&Deep模型下的性能数据，进行对比。


<!-- omit in toc -->
## 目录
- [一、测试说明](#一测试说明)
- [二、环境介绍](#二环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [三、环境搭建](#三环境搭建)
- [四、测试步骤](#四测试步骤)
  - [1.多机（32机）测试](#32机16线程测试)
- [五、测试结果](#五测试结果)
  - [1.Paddle训练性能](#1paddle训练性能)
  - [2.与业内其它框架对比](#2与业内其它框架对比)
- [六、日志数据](#六日志数据)

## 一、测试说明

我们统一使用 **吞吐能力** 作为衡量性能的数据指标。**吞吐能力** 是业界公认的、最主流的框架性能考核指标，它直接体现了框架训练的速度。

Wide&Deep 作为推荐系统领域经典的代表性的模型。在测试性能时，我们以 **单位时间内能够完成训练的样本行数量（lines/sec）** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。

测试中，我们选择4/8/16/32四种集群节点下，测试吞吐性能：

- **节点数**

   本次测试关注4机、8机、16机、32机情况下，模型的训练吞吐性能。
   每台服务器均使用16线程进行训练。


关于其它一些参数的说明：
- **ASYNC**

   AYSNC（异步训练） 能够提升参数服务器下模型的训练速度且在实际场景下对模型效果影响有限。因此，本次测试全部在打开 ASYNC 模式下进行。


## 二、环境介绍
### 1.物理机环境

- 多机(单台服务器配置)
  - 系统：CentOS release 6.3
  - CPU：Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz 
  - 内存：256 GB


### 2.Docker 镜像

Paddle Docker的基本信息如下：

- Docker: hub.baidubce.com/paddlepaddle/paddle-benchmark:cuda10.1-cudnn7-runtime-ubuntu16.04
- Paddle：release 2.0
- 模型代码：[PaddleRec](https://github.com/PaddlePaddle/PaddleRec)


## 三、环境搭建

- 拉取docker
  ```bash
  docker pull hub.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.0-cudnn7
  ```

- 启动docker
  ```bash
  # 假设数据放在<path to data>目录下
  docker run -it -v <path to data>:/data hub.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.0-cudnn7 /bin/bash
  ```

- 拉取PaddleRec
  ```bash
  git clone https://github.com/PaddlePaddle/PaddleRec.git
  cd PaddleRec
  # 本次测试是在如下版本下完成的：
  git checkout b0904fd250715b3c040c88881395bad06eea9be6
  ```

- 数据部署
  训练使用开源的Criteo数据集，下载地址为：https://fleet.bj.bcebos.com/ctr_data.tar.gz，将数据挂载进入Docker后，
  数据集位于/data目录，具有如下目录结构：
  ```shell
  /data
   |---raw_data
   |   |---part-55
   |   |---part-56
   |   |...
   |---test_data
   |   |---part-226
   |   |---part-236
  ```
  其中，raw_data子目录下包含训练数据集，文件名称为part-*，test_data子目录包含测试数据集，文件名称为part-*

## 四、测试步骤

### 2.多机（32机）测试
- 执行多机测试前，需要预先配置好多机的环境变量，即每台机器都要配置当前运行所需的环境参数
  参数服务器相关的环境变量配置列表：

  | 环境变量 | 说明 | 示例 |
  |:-----:|:-----:|:------:|
  | PADDLE_PSERVERS_IP_PORT_LIST | PSERVER的IP:PORT列表 | "127.0.0.1:67001,127.0.0.1:67002" |
  | PADDLE_TRAINERS_NUM | 当前TRAINER的个数 | "20" |
  | TRAINING_ROLE | 目前两种角色 "TRAINER", "PSERVER" |
  | PADDLE_TRAINER_ID | 每个TRAINER的id | 12 |
  | PADDLE_PORT | 每个PSERVER服务的端口 | 67001 |
  | POD_IP | 每个PSERVER服务的IP | "127.0.0.1" |

- 下载我们编写的测试脚本，并执行该脚本(每台机器均需要执行)
  ```bash
  wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/ResNet50V1.5/scripts/ResNet50_32gpu_amp_bs208.yaml
  bash paddle_test_multi_node_all.sh
  ```

- 执行后PSERVER端结束时输出日志类似如下：
   ```bash
   server.cpp:1037] Server[paddle::distributed::BrpcPsService] is serving on port=62004.
   server.cpp:1040] Check out http://IP:67002 in web browser.
   ```

- 执行后WORKER端结束时输出日志类似如下：
   ```bash
   epoch 1 using time 2980.92606091, auc: 0.877906736262
   distributed training finished.
   server.cpp:1095] Server[paddle::distributed::DownpourPsClientService] is going to quit
   ```

## 五、测试结果

### 1.Paddle训练性能


- 训练吞吐率(lines/sec)如下（数据是所有节点训练速度的加和）:

| 节点数 | 吞吐 |
|:-----:|:-----:|
|4 | 285495 | 
|8 | 533848 | 
|16 | 1024367 |
|32 | 928801 | 

### 2.与业内其它框架对比

说明：
- 同等执行环境下测试
- 单位：`lines/sec`
- 对于支持 `ASYNC(异步训练)` 的框架，以下测试为开启 `ASYNC` 的数据

结果：
- 测试(WORKER和SERVER数量相等)

  | 参数 | PaddlePaddle | TensorFlow 1.12 |
  |:-----:|:-----:|:-----:|
  | 4x4   | <sup>285495</sup>  | <sup>21757</sup> |
  | 8x8   | <sup>533848</sup>  | <sup>23187</sup> |
  | 16x16 | <sup>1024367</sup>  | <sup>22500</sup> |
  | 32x32 | <sup>928801</sup> | <sup>22853</sup> |


