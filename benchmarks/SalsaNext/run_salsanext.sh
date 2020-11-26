#!/bin/sh
export CUDA_VISIBLE_DEVICES="0,1"
cd ./train/tasks/semantic;  ./train.py -d /home/usl/Datasets/rellis  -ac ./config/arch/salsanext_ouster.yml -dc ./config/labels/rellis.yaml -n rellis -l ./logs -p ""