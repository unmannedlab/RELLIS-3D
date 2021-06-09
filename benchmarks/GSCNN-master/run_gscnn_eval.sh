#!/bin/bash
export PYTHONPATH=/home/usl/Code/PengJiang/RELLIS-3D/benchmarks/GSCNN-master/:$PYTHONPATH
echo $PYTHONPATH
python train.py --dataset rellis --bs_mult 3 --lr 0.001 --exp final \
                --checkpoint_path /home/usl/Downloads/best_epoch_84_mean-iu_0.46839.pth \
                --mode test \
                --viz \
                --data-cfg  /home/usl/Code/Peng/data_collection/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml \
                --test_sv_path /home/usl/Datasets/prediction