#!/bin/bash
export PYTHONPATH=/home/usl/Code/Peng/data_collection/benchmarks/GSCNN-master/:$PYTHONPATH
echo $PYTHONPATH
python train.py --dataset rellis --bs_mult 3 --lr 0.001 --exp final