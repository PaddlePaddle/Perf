<!-- omit in toc -->

# Paddle Word2Vec 性能测试

此处给出了Paddle Word2Vec 的详细测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在同等执行环境下，业内几个知名框架在word2vec模型下的性能数据，进行对比。


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

Word2Vec 作为推荐系统领域早期代表性的模型。在测试性能时，我们以 **单位时间内能够完成训练的单词数量（word/sec）** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。测试中，我们选择4/8/16/32四种集群节点下，使用同构的CPU服务器，基于参数服务器模式测试吞吐性能：

- **数据集**

  全量训练集使用1 Billion Word Language Model Benchmark的训练集，该训练集一共包含30294863个文本。

  全量测试集共包含19558个测试样例，每个测试样例由4个词组合构成，依次记为word_a, word_b, word_c, word_d。组合中，前两个词word_a和word_b之间的关系等于后两个词word_c和word_d之间的关系，例如:
  ```
  Beijing China Tokyo Japan
  write writes go goes
  ```
  所以word2vec的测试任务实际上是一个常见的词类比任务，我们希望通过公式emb(word_b) - emb(word_a) + emb(word_c)计算出的词向量和emb(word_d)最相近。最终整个模型的评分用成功预测出word_d的数量来衡量。

- **节点数**

   本次测试关注4机、8机、16机、32机情况下，模型的训练吞吐性能。
   每台服务器均使用16线程进行训练。


关于其它一些参数的说明，这些参数可以在`benchmark.yaml`中找到：

- **sync_mode = async**

   AYSNC（异步训练） 能够提升参数服务器下模型的训练速度且在实际场景下对模型效果影响有限。因此，本次测试全部在打开 ASYNC 模式下进行。

- **sync_mode = geo**

   GEO异步训练（异步训练） 能够提升参数服务器下模型的训练速度且在Word2Vec下验证不影响模型效果。因GEO-ASYNC是飞桨独有的训练模式，本次测试会新增飞桨的吞吐在 GEO-ASYNC 模式下进行。

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

- 进入Word2Vec模型目录
  ```bash
  # cd PaddleRec
  cd models/recall/word2vec/benchmark
  ```

  word2vec/benchmark目录有以下文件是benchmark测试相关：

  - static_model.py       *模型组网*
  - benchmark.yaml        *通用的benchmark超参配置*
  - benchmark_reader.py   *benchmark适配的数据读取处理脚本*
  - benchmark_data.sh     *下载数据的脚本文件*
  - w2v_infer.py          *训练效果验证脚本

- 数据部署
  训练使用开源的one_billion数据集，下载地址为：https://paddlerec.bj.bcebos.com/benchmark/word2vec_benchmark_data.tar.gz
  
  亦可直接使用模型目录下的 `benchmark_data.sh`脚本执行数据下载

  ```bash
  sh benchmark_data.sh
  ```

  将数据放置到PaddleRec/models/recall/word2vec/benchmark目录，数据集应具有如下目录结构：

  ```shell
  /word2vec/benchmark
   |---train_data
   |   |---part-55
   |   |---part-56
   |   |...
   |---test_data
   |   |---questions-words.txt
   |---dict
   |   |---word_count_dict.txt
   |   |---word_id_dict.txt
  ```
  其中，train_data子目录下包含训练数据集，文件名称为part-*，test_data子目录包含测试数据集，文件名称为questions-words.txt, 训练数据共有1024个Part，测试数据有1个Part，dict包括两个训练及预测所需的字典文件

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

- 通过`fleetrun`命令启动分布式训练，在每台机器上都需要执行以下命令：：

  ```bash
  # pwd = PaddleRec/models/recall/word2vec/benchmark
  fleetrun --servers="ip1:port1,ip2:port2,...,ip32:port32" --workers="ip1:port1,ip2:port2,...,ip32:port32" ../../../../tools/static_ps_trainer.py -m benchmark.yaml
  ```

- 执行后PSERVER端结束时输出日志类似如下：

   日志位于`PaddleRec/models/recall/word2vec/benchmark/log/serverlog.*` 

   ```bash
   server.cpp:1037] Server[paddle::distributed::BrpcPsService] is serving on port=62004.
   server.cpp:1040] Check out http://IP:67002 in web browser.
   ```

- 执行后WORKER端结束时输出日志类似如下：

   日志位于`PaddleRec/models/recall/word2vec/benchmarkp/log/workerlog.*` 

   ```bash
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

    reader_type: "QueueDataset"  # DataLoader / QueueDataset / RecDataset
    pipe_command: "python benchmark_reader.py"
    dataset_debug: False
    split_file_list: False
  ```

  > 如果每个机器都挂载了全量的数据，配置`split_file_list: True`

- 环境搭建、数据准备、运行命令、日志查看步骤与DataSet模式一致

### 多机32机 16线程 GEO + DataSet 模式 测试

- 通过`benchmark.yaml`配置训练模式及超参：

  运行16线程 Async + DataLoader 模式，需要配置如下超参

  ```yaml
  runner:
    epochs: 15

    sync_mode: "geo"  # sync / async /geo / heter
    geo_step: 400
    thread_num: 16

    reader_type: "QueueDataset"  # DataLoader / QueueDataset / RecDataset
    pipe_command: "python benchmark_reader.py"
    dataset_debug: False
    split_file_list: False
  ```

  > 如果每个机器都挂载了全量的数据，配置`split_file_list: True`

- 环境搭建、数据准备、运行命令、日志查看步骤与DataSet模式一致

### 多机32机 16线程 GEO + DataLoader 模式 测试

- 通过`benchmark.yaml`配置训练模式及超参：

  运行16线程 Async + DataLoader 模式，需要配置如下超参

  ```yaml
  runner:
    epochs: 15

    sync_mode: "geo"  # sync / async /geo / heter
    geo_step: 400
    thread_num: 16

    reader_type: "DataLoader"  # DataLoader / QueueDataset / RecDataset
    split_file_list: False
  ```

  > 如果每个机器都挂载了全量的数据，配置`split_file_list: True`

- 环境搭建、数据准备、运行命令、日志查看步骤与DataSet模式一致

## 五、测试结果

### 1.Paddle训练性能


- Async模式训练吞吐率(word/sec)如下（数据是所有节点训练速度的加和）:


  | 节点数 | 线程数 |DataSet吞吐 | DataLoader吞吐 |
  |:-----:|:-----:|:-----:|:-----:|
  |4 | 16 |123873 | 139154 | 
  |8 | 16 |195847 | 221268 | 
  |16 | 16 |306993 |324421 | 
  |32 | 16 |476067 | 534432 | 

- GEO模式训练吞吐率(word/sec)如下（数据是所有节点训练速度的加和）:


  | 节点数 | 线程数 |DataSet吞吐 | DataLoader吞吐 |
  |:-----:|:-----:|:-----:|:-----:|
  |4 | 16 |272754 | 168633 | 
  |8 | 16 |512492 | 327173 | 
  |16 | 16 |960883 |625802 | 
  |32 | 16 |1503222 | 1100679 | 

### 2.与业内其它框架对比

说明：
- 同等执行环境下测试
- 单位：`word/sec`
- 对于支持 `ASYNC(异步训练)` 的框架，以下测试为开启 `ASYNC` 的数据

结果：

WORKER和SERVER数量相等，每个节点上分别启动一个Worker进程与一个Server进程

> 注：Tensorflow性能数据是在与Paddle相同的机器配置环境下测得，使用了TF推荐的分布式API，运行多次取平均值。

> 若您可以测得更好的数据，欢迎联系我们，提供测试方法，更新数据

  | 节点 | PaddlePaddle：Async | PaddlePaddle：GEO |TensorFlow 1.12 |
  |:-----:|:-----:|:-----:|:-----:|
  | 4   | <sup>139154</sup>  | <sup>272754</sup> |<sup>16296</sup> |
  | 8   | <sup>221268</sup>  | <sup>512492</sup> |<sup>39625</sup> |
  | 16 | <sup>324421</sup>  | <sup>960883</sup> |<sup>71436</sup> |
  | 32 | <sup>534432</sup> | <sup>1503222</sup> |<sup>114037</sup> |



