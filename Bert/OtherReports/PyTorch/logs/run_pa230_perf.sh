bash scripts/run_benchmark.sh 96 1 fp32 > torchbert_base_seq128_bs96_fp32_gpu1 2>&1  
mv /workspace/bert/results/checkpoints /workspace/bert/results/checkpoints_torchbert_base_seq128_bs96_fp32_gpu1
bash scripts/run_benchmark.sh 96 8 fp32 > torchbert_base_seq128_bs96_fp32_gpu8 2>&1  
mv /workspace/bert/results/checkpoints /workspace/bert/results/checkpoints_torchbert_base_seq128_bs96_fp32_gpu8

# 若测试单机8卡 batch_size=96、FP16 的训练性能，执行如下命令：
bash scripts/run_benchmark.sh 96 1 fp16 > torchbert_base_seq128_bs96_fp16_gpu1 2>&1  
mv /workspace/bert/results/checkpoints /workspace/bert/results/checkpoints_torchbert_base_seq128_bs96_fp16_gpu1

bash scripts/run_benchmark.sh 96 8 fp16 > torchbert_base_seq128_bs96_fp16_gpu8 2>&1
mv /workspace/bert/results/checkpoints /workspace/bert/results/checkpoints_torchbert_base_seq128_bs96_fp16_gpu8
