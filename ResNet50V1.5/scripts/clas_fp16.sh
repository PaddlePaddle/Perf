#!/bin/bash

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.8
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_conv_workspace_size_limit=4000
export FLAGS_cudnn_batchnorm_spatial_persistent=1
export FLAGS_max_inplace_grad_add=8

base_batch_size=$1
num_gpus=$2
TRANER_IPS=$3


use_pure_fp16=False
sed -i "s/output_fp16.*/output_fp16: False/g" ppcls/configs/ImageNet/ResNet/ResNet50_fp16.yaml

train_cmd="-c ppcls/configs/ImageNet/ResNet/ResNet50_amp_O2_ultra.yaml
           -o Global.epochs=1
           -o DataLoader.Train.sampler.batch_size=${base_batch_size}
           -o Global.eval_during_train=False
           -o DataLoader.Train.dataset.image_root=./data
           -o DataLoader.Train.dataset.cls_label_path=./data/train_list.txt
           -o DataLoader.Train.loader.num_workers=8
           -o Global.print_batch_step=10
           -o Global.device=gpu
           -o Global.image_shape=[4,224,224]
           -o AMP.use_pure_fp16=${use_pure_fp16}
           "

if [[ ${num_gpus} == 1 ]]; then
    train_cmd="python -u ppcls/static/train.py "${train_cmd}
elif [[ ${num_gpus} == 8 ]]; then
     train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 ppcls/static/train.py "${train_cmd}" -o Global.use_dali=True"
else
     train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 --ips="${TRANER_IPS}" ppcls/static/train.py "${train_cmd}" -o Global.use_dali=True"
fi
echo ${train_cmd}
${train_cmd}
