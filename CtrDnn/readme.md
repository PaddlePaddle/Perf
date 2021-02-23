<!-- omit in toc -->

# Paddle CTR-DNN 性能测试

此处给出了Paddle CTR-DNN 的详细测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在同等执行环境下，业内几个知名框架在DNN模型下的性能数据，进行对比。


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

CTR-DNN 作为推荐系统领域早期代表性的模型。在测试性能时，我们以 **单位时间内能够完成训练的样本行数量（example/sec）** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。测试中，我们选择4/8/16/32四种集群节点下，使用同构的CPU服务器，基于参数服务器模式测试吞吐性能：

- **数据集**

训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。


- **节点数**

   本次测试关注4机、8机、16机、32机情况下，模型的训练吞吐性能。
   每台服务器均使用16线程进行训练。


关于其它一些参数的说明，这些参数可以在`benchmark.yaml`中找到：

- **sync_mode = async**

   AYSNC（异步训练） 能够提升参数服务器下模型的训练速度且在实际场景下对模型效果影响有限。因此，本次测试全部在打开 ASYNC 模式下进行。

- **split_file_list = False**
 
   参数服务器使用数据并行模式进行训练，若每台服务器上的数据是已经经过切分的，则配置该选项为`False`，若每台服务器都挂载了完整的数据目录，则设置该选项为`True`，进行均匀的数据切分。


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

- 安装PaddlePaddle

  paddle 版本应大于 realease/2.0
  ```bash
  pip instsall -U paddlepaddle
  ```

- 拉取PaddleRec
  ```bash
  git clone https://github.com/PaddlePaddle/PaddleRec.git
  cd PaddleRec
  # 本次测试是在如下版本下完成的：
  git checkout 10c1bd7a89881fae1336a2075cbe793856771c72
  ```

- 进入Dnn模型目录
  ```bash
  # cd PaddleRec
  cd models/rank/dnn
  ```

  Dnn目录有以下文件是benchmark测试相关：

  - static_model.py       *模型组网*
  - benchmark.yaml        *通用的benchmark超参配置*
  - benchmark_reader.py   *benchmark适配的数据读取处理脚本*
  - benchmark_data.sh     *下载数据的脚本文件*

- 数据部署
  训练使用开源的Criteo数据集，下载地址为：https://paddlerec.bj.bcebos.com/benchmark/criteo_benchmark_data.tar.gz
  
  亦可直接使用模型目录下的 `benchmark_data.sh`脚本执行数据下载

  ```bash
  sh benchmark_data.sh
  ```

  将数据放置到PaddleRec/models/rank/dnn目录，数据集应具有如下目录结构：

  ```shell
  /dnn
   |---train_data
   |   |---part-55
   |   |---part-56
   |   |...
   |---test_data
   |   |---part-226
   |   |---part-236
  ```
  其中，train_data子目录下包含训练数据集，文件名称为part-*，test_data子目录包含测试数据集，文件名称为part-*, 训练数据共有1024个Part，测试数据有10个Part

## 四、测试步骤

### 多机32机 16线程 Async + DataSet 模式 测试
- 执行多机测试前，需要在每台机器上都执行上述步骤: 
  1. 配置docker环境；
  2. 安装Paddle；
  3. 克隆PaddleRec；
  4. 下载数据

- 确保各个机器之间的联通正常，通过IP可互相访问

- 通过`benchmark.yaml`配置训练模式及超参：

  运行16线程 Async + DataSet 模式，需要配置如下超参

  ```yaml
  runner:
    epochs: 15

    sync_mode: "async"  # sync / async /geo / heter
    thread_num: 16

    reader_type: "QueueDataset"  # DataLoader / QueueDataset / RecDataset
    pipe_command: "python benchmark_reader.py"
    dataset_debug: False
    split_file_list: False
  ```

  > 如果每个机器都挂载了全量的数据，配置`split_file_list: True`

- 启动单机训练，进行验证

  ```bash
  python -u ../../../tools/static_ps_trainer.py -m benchmark.yaml
  ```

- 通过`fleetrun`命令启动分布式训练，在每台机器上都需要执行以下命令：

  ```bash
  # pwd = PaddleRec/models/rank/dnn
  fleetrun --servers="ip1:port1,ip2:port2,...,ip32:port32" --workers="ip1:port1,ip2:port2,...,ip32:port32" ../../../tools/static_ps_trainer.py -m benchmark.yaml
  ```

- 执行后PSERVER端结束时输出日志类似如下：

   日志位于`PaddleRec/models/rank/dnn/log/serverlog.*` 

   ```bash
   server.cpp:1037] Server[paddle::distributed::BrpcPsService] is serving on port=62004.
   server.cpp:1040] Check out http://IP:67002 in web browser.
   ```

- 执行后WORKER端结束时输出日志类似如下：

   日志位于`PaddleRec/models/rank/dnn/log/workerlog.*` 

   ```bash
   epoch 1 using time 2980.92606091, auc: 0.877906736262
   distributed training finished.
   server.cpp:1095] Server[paddle::distributed::DownpourPsClientService] is going to quit
   ```

### 多机32机 16线程 Async + DataLoader 模式 测试

- 通过`benchmark.yaml`配置训练模式及超参：

  运行16线程 Async + DataLoader 模式，需要配置如下超参

  ```yaml
  runner:
    epochs: 15

    sync_mode: "async"  # sync / async /geo / heter
    thread_num: 16

    reader_type: "DataLoader"  # DataLoader / QueueDataset / RecDataset
    split_file_list: False
  ```

  > 如果每个机器都挂载了全量的数据，配置`split_file_list: True`

- 环境搭建、数据准备、运行命令、日志查看步骤与DataSet模式一致

## 五、测试结果

### 1.Paddle训练性能


- 训练吞吐率(example/sec)如下（数据是所有节点训练速度的加和）:

| 节点数 | 线程数 |DataSet吞吐 | DataLoader吞吐 |
|:-----:|:-----:|:-----:|:-----:|
|4 | 16 |378599 | 42378 | 
|8 | 16 |736955 | 84910 | 
|16 | 16 |1448210 |176246 | 
|32 | 16 |2553503 | 352492 | 

### 2.与业内其它框架对比

说明：
- 同等执行环境下测试
- 单位：`example/sec`
- 对于支持 `ASYNC(异步训练)` 的框架，以下测试为开启 `ASYNC` 的数据

结果：

WORKER和SERVER数量相等，每个节点上分别启动一个Worker进程与一个Server进程

> 注：Tensorflow性能数据是在与Paddle相同的机器配置环境下测得，使用了TF推荐的分布式API，运行多次取平均值。

> 若您可以测得更好的数据，欢迎联系我们，提供测试方法，更新数据
  

| 节点数 | PaddlePaddle | TensorFlow 1.12 |
|:-----:|:-----:|:-----:|
| 4   | <sup>378599</sup>  | <sup>13507</sup> |
| 8   | <sup>736955</sup>  | <sup>22862</sup> |
| 16 | <sup>1448210</sup>  | <sup>10703</sup> |
| 32 | <sup>2553503</sup> | <sup>21407</sup> |

## 六、日志数据
### 1.paddlepaddle 4机、8机、16机、32机日志

- [CtrDnn DataSet 4机](./logs/CtrDnn_DataSet_16Thread_4Node)
- [CtrDnn DataSet 8机](./logs/CtrDnn_DataSet_16Thread_8Node)
- [CtrDnn DataSet 16机](./logs/CtrDnn_DataSet_16Thread_16Node)
- [CtrDnn DataSet 32机](./logs/CtrDnn_DataSet_16Thread_32Node)
- [CtrDnn DataLoader 4机](./logs/CtrDnn_DataLoader_16Thread_4Node)
- [CtrDnn DataLoader 8机](./logs/CtrDnn_DataLoader_16Thread_8Node)
- [CtrDnn DataLoader 16机](./logs/CtrDnn_DataLoader_16Thread_16Node)
- [CtrDnn DataLoader 32机](./logs/CtrDnn_DataLoader_16Thread_32Node)


