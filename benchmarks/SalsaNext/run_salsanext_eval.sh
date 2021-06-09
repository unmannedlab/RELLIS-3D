#!/bin/sh
export CUDA_VISIBLE_DEVICES="1"
cd ./train/tasks/semantic
python infer2.py -d /path/to/Datasets/rellis -l /path/to/save -s test -m /path/to/salsanext_best/2020-10-14-17:37rellis