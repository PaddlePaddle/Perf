# MPI Cluster搭建和使用
## 前言
考虑到我们做实验一般会一次启动多组子实验，我们用MPI来做这种小规模的实验的管理。这个场景下，MPI的优势在于

- 轻量管理，测试简单   
   特别是当我们去跑竞品实验时，MPI是比较好的调度粘合剂。

- 保证一个实验中的多个子实验不会相互`串了`.   
MPI的world会因为一个rank的失败而整体退出，当一个脚本里边包含多个子实验时，这个特性有效防止多个子实验之间串行(hang)
	
## MPI Cluster搭建
需要在多机之间增加SSH互信

- 可以参考[这里](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/)的前三步
-  可以通过`mpirun hostname`来检查是否搭建成功
-  如果需要改变sshd的端口，可以通过启动ssh的server     
 
```
export SSHD_PORT=xxxx
/etc/init.d/ssh start
``` 


##  MPI Cluster使用简介
### 需要把集群节点环境传给通信框架
如MPI管理的进程中调用了Pytorch的Launch

```
# 下边的环境变量需要根据集群的环境进行修改
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=xgbe0
# 注意这个变量，小坑，主要是让IB找到对应的设备驱动
# 你的机器上可能不用写
export IBV_DRIVERS=mlx5
	
# 获得rank=0的ip
IFS=',' read -r -a node_array <<< "${NODES_IPS}"
export MASTER_NODE=${node_array[0]}
# 获得端口，主要用于torch的master port
export MASTER_PORT=${NODE_PORT}
export NUM_NODES=${NODES_NUM}
echo "runing on ${MASTER_NODE}:${MASTER_PORT}-${NUM_NODES}"
	
# 传给mpi的变量需要根据自己的环境自己设定
#  不熟悉 mpi的同学尤其要注意环境变量
# slaver节点上的mpi启动时，其环境变量按道理应该继承当前机器环境，但经过测试未必是这样，还是直接手动指定来的保险
mpirun="/usr/local/openmpi-3.1.0/bin/orterun --allow-run-as-root -tag-output \
   -timestamp-output --hostfile ${WORKSPACE}/hostfile \
   -mca btl_tcp_if_exclude docker0,lo,matrixdummy0,matrix0 \
   -x PATH -x LD_LIBRARY_PATH \
   -x NCCL_IB_GID_INDEX \
   -x NCCL_DEBUG \
   -x NCCL_SOCKET_IFNAME \
   -x IBV_DRIVERS -x MASTER_NODE -x MASTER_PORT -x NUM_NODES -x WORKSPACE"
```

其中hostfile是一个配置文件, 注意其中的`slots=1`，表示单机启动一个MPI进程

```
hostname1 slots=1
hostname2 slots=1
```


### 通信框架可以从MPI中获取信息
如MPI管理进程，进程使用Horovod作为通信框架

```
# 下边的环境变量需要根据集群的环境进行修改
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=xgbe0
# 注意这个变量，小坑，主要是让IB找到对应的设备驱动
# 你的机器上可能不用写
export IBV_DRIVERS=mlx5
		
# 传给mpi的变量需要根据自己的环境自己设定
# 不熟悉 mpi的同学尤其要注意环境变量
# slaver节点上的mpi启动时，其环境变量按道理应该继承当前机器环境，但经过测试未必是这样，还是直接手动指定来的保险
mpirun="/usr/local/openmpi-3.1.0/bin/orterun --allow-run-as-root -tag-output \
   -timestamp-output --hostfile ${WORKSPACE}/hostfile \
   -mca btl_tcp_if_exclude docker0,lo,matrixdummy0,matrix0 \
   -x PATH -x LD_LIBRARY_PATH \
   -x NCCL_IB_GID_INDEX \
   -x NCCL_DEBUG \
   -x NCCL_SOCKET_IFNAME \
   -x IBV_DRIVERS -x WORKSPACE"
```

其中hostfile为

```
hostname1 slots=8
hostname2 slots=8
```
注意其中的`slots=8`，表示单机启动8(卡数个）个MPI进程
   
	
#### 获得当前机器的mpirank，原作者在[这里](https://gist.github.com/serihiro/33f8f775cd8ba524d7b20d08d170e69c)，拷贝如下
	
```
from mpi4py import MPI
	
comm = MPI.COMM_WORLD
print(comm.Get_rank())
```
使用方式：
	
```
rank=`python get_mpi_rank.py`
```
