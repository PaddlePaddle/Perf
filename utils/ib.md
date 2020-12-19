# IB的安装

## 简介
多机情况下NGC对网络通信的性能提升很明显。
不过由于NGC BERT一般采用梯度多步聚合的方式，对网络延时的要求降低，所以当使用梯度聚合时，IB可选。

## 安装
你可以直接从[这里](https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed)下载驱动，但需要注意Host机器的IB驱动版本要大于等于Docker中的镜像版本，我们实验中采用版本是`MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64`
	
安装方式：
	
```
apt-get update && apt-get install lsb-core -y
# 自行修改HOME_WORK_DIR
tar zxf MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64.tgz -C ${HOME_WORK_DIR}/ && \
    cd ${HOME_WORK_DIR}/MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64 && \
    ./mlnxofedinstall --user-space-only --force --without-neohost-backend && \
    rm -rf ${HOME_WORK_DIR}/MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64*
```
	
注：
	
- 安装过程中会提示很多依赖包没有安装的warning`dpatch libelf1 libmnl0 libltdl-dev lsof ..`等，经测试，其实不用安装
- 另外，NGC的原版镜像中貌似已经在`/opt/mellanox/`目录下设置了`change_mofed_version.sh`脚本来切换IB driver(主要就是几个 .so拷贝到对应的位置)，但不太理解为何把`5.0-2.1.8`软链到了`4.6-1.0.1/`，也不太理解为何没有安装rdma的驱动。Anymore，这些都可以通过前边的驱动安装直接覆盖掉