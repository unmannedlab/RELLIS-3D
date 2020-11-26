#!/bin/bash
export PYTHONPATH=/home/usl/Code/Peng/data_collection/benchmarks/GSCNN-master/:$PYTHONPATH
echo $PYTHONPATH
python train.py --dataset rellis --bs_mult 3 --lr 0.001 --exp final \
                --checkpoint_path /home/usl/Code/Peng/data_collection/benchmarks/GSCNN-master/logs/ckpt/final/testing/best_epoch_84_mean-iu_0.46839.pth \
                --mode test \
                --test_sv_path /home/usl/Datasets/prediction