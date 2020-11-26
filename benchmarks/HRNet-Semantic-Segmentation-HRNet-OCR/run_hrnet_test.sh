#!/bin/bash
export PYTHONPATH=/home/usl/Code/Peng/data_collection/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/:$PYTHONPATH
python tools/test.py --cfg experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET val.lst \
                     OUTPUT_DIR /home/usl/Datasets \
                     TEST.MODEL_FILE /home/usl/Code/Peng/data_collection/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/output/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484/best.pth
