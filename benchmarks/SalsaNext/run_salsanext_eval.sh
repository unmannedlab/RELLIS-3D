#!/bin/sh
export CUDA_VISIBLE_DEVICES="1"
cd ./train/tasks/semantic
python infer2.py -d /home/usl/Datasets/rellis -l /home/usl/Datasets -s test -m /home/usl/Code/Peng/data_collection/benchmarks/SalsaNext/train/tasks/semantic/logs/logs/2020-10-14-17_37rellis