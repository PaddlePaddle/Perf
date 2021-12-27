#!/bin/bash

base_batch_size=$1
num_gpus=$2

train_cmd="-c ppcls/configs/ImageNet/ResNet/ResNet50.yaml
           -o Global.print_batch_step=10
           -o DataLoader.Train.sampler.batch_size=${base_batch_size}
           -o Global.eval_during_train=False
           -o DataLoader.Train.dataset.image_root=./data
           -o DataLoader.Train.dataset.cls_label_path=./data/train_list.txt
           -o fuse_elewise_add_act_ops=True
           -o enable_addto=True
           -o DataLoader.Train.loader.num_workers=8
           -o Global.epochs=1"

if [ ${num_gpus} == 1 ]; then
    train_cmd="python -u ppcls/static/train.py "${train_cmd}
else
     train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 ppcls/static/train.py "${train_cmd}" -o Global.use_dali=True"
fi