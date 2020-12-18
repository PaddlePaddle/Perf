#/bin/bash

python benchmark.py -n 1,8 -b 96 --dtype float32 -o benchmark_report_fp32.json -i 500 -e 3 -w 1 --num-examples 32000 --mode train > /log/mxnet_gpu1_gpu8_fp32_bs96.txt 2>&1

python benchmark.py -n 1,8 -b 128 --dtype float16 -o benchmark_report_fp16.json -i 500 -e 3 -w 1 --num-examples 32000 --mode train > /log/mxnet_gpu1_gpu8_amp_bs128.txt 2>&1

python benchmark.py -n 1,8 -b 192 --dtype float16 -o benchmark_report_fp16.json -i 500 -e 3 -w 1 --num-examples 32000 --mode train > /log/mxnet_gpu1_gpu8_amp_bs192.txt 2>&1
