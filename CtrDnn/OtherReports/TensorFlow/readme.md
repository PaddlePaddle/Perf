数据下载
```bash
wget --no-check-certificate https://paddlerec.bj.bcebos.com/benchmark/tf_criteo.tar.gz
```

运行环境
- 使用Tensorflow官方1.12-CPU-Docker
- 在每台机器上配置分布式训练环境变量

 | 环境变量 | 说明 | 示例 |
   |:-----:|:-----:|:------:|
   | PADDLE_PSERVERS_IP_PORT_LIST | PSERVER的IP:PORT列表 | "127.0.0.1:67001,127.0.0.1:67002" |
   | PADDLE_WORKERS_IP_PORT_LIST | WORKER的IP:PORT列表 | "127.0.0.1:67001,127.0.0.1:67002" |
   | PADDLE_TRAINERS_NUM | 当前TRAINER的个数 | "32" |
   | TRAINING_ROLE | 目前两种角色 "TRAINER", "PSERVER" |"PSERVER"|
   | PADDLE_TRAINER_ID | 每个TRAINER的id | 12 |
   | PADDLE_PORT | 每个PSERVER服务的端口 | 67001 |
   | POD_IP | 每个PSERVER服务的IP | "127.0.0.1" |


运行命令：
```bash
python ctr_dnn_distribute.py --sync_mode=False --is_local=False
```