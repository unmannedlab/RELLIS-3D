# RELLIS-3D Benchmarks

The HRNet, SalsaNext and KPConv can use environment file ```requirement.txt```. GSCNN need use file ```gscnn_requirement.txt```.

## Image Semantic Segmenation 
**Note: New script for evaluate the results is available for [point cloud](https://github.com/unmannedlab/RELLIS-3D/blob/main/utils/Evaluate_pt.ipynb) and [image](https://github.com/unmannedlab/RELLIS-3D/blob/main/utils/Evaluate_img.ipynb)**

**Note: if you used your script the evalutate the image segmentation model, the id of save predition is from 0 to num_classes. You may need to convert the id back or convert the ground truth using this [script](https://github.com/unmannedlab/RELLIS-3D/blob/main/utils/label_convert.py).**


### HRNet+OCR
The HRNext+OCR is a fork from [https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)

To evaluate the dataset:
```
cd /path/to/code/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR
export PYTHONPATH=/path/to/code/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/:$PYTHONPATH
python tools/test.py --cfg experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET val.lst \
                     OUTPUT_DIR /path/for/save/prediction \
                     TEST.MODEL_FILE /path/to/code/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/output/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484/best.pth
```
Add dataset path to ```ROOT``` in ```experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484.yaml``` 

The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from [onedrive](https://onedrive.live.com/?authkey=%21AKvqI6pBZlifgJk&cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21116&parId=F7FD0B7F26543CEB%21105&action=locate) or [https://github.com/HRNet/HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification).

Dowload pre-trained model ([Download 751MB](https://drive.google.com/file/d/137Lfw6HcDmdEReu_R7Q_I-zmRvvqFys3/view?usp=sharing))

To retrain the HRNet on RELLIS-3D:
```
export PYTHONPATH=/path/to/code/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/:$PYTHONPATH
echo $PYTHONPATH
PY_CMD="python -m torch.distributed.launch --nproc_per_node=2"
$PY_CMD tools/train.py --cfg experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484.yaml
```
Add dataset path to ```ROOT``` in ```experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484.yaml```


### GSCNN
The GSCNN is a fork from [https://github.com/nv-tlabs/GSCNN](https://github.com/nv-tlabs/GSCNN)

To evaluate the dataset:
```
cd /path/to/code/benchmarks/GSCNN-master
export PYTHONPATH=/path/to/code/benchmarks/GSCNN-master/:$PYTHONPATH
python train.py --dataset rellis --bs_mult 3 --lr 0.001 --exp final \
                --checkpoint_path /path/to/pre-trained/chk_file \
                --mode test \
                --test_sv_path /path/for/save/prediction
```
Add dataset path to ```__C.DATASET.RELLIS_DIR``` in ```benchmarks/GSCNN-master/config.py``` 
Dowload pre-trained model ([Download 1GB](https://drive.google.com/file/d/1Z8OlstkdzDrY9k-yxMQmVB192ac8j4MD/view?usp=sharing))


To retrain the GSCNN on RELLIS-3D:
```
export PYTHONPATH=/path/to/code/benchmarks/GSCNN-master/:$PYTHONPATH
python train.py --dataset rellis --bs_mult 3 --lr 0.001 --exp final
```
Add dataset path to ```__C.DATASET.RELLIS_DIR``` in ```benchmarks/GSCNN-master/config.py``` 
The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from [here](https://drive.google.com/file/d/1OfKQPQXbXGbWAQJj2R82x6qyz6f-1U6t/view).

## LiDAR Semantic Segmenation

### SalsaNext

The SalsaNext is a fork from [https://github.com/Halmstad-University/SalsaNext](https://github.com/Halmstad-University/SalsaNext)

To evaluate the dataset:
```
#!/bin/sh
export CUDA_VISIBLE_DEVICES="1"
cd /path/to/code/benchmarks/SalsaNext/train/tasks/semantic  
python infer2.py -d /path/to/RELLIS-3D -l /path/for/save/prediction -s test -m /path/to/pre-trained/model/folder
```
Dowload pre-trained model ([Download 157MB](https://drive.google.com/file/d/1DxuzlnFKnU8EpSuODRywJUrJlieUUheg/view?usp=sharing))

To retrain the SalsaNext on RELLIS-3D:
```
export CUDA_VISIBLE_DEVICES="0,1"
cd /path/to/code/benchmarks/SalsaNext/train/tasks/semantic  
./train.py -d /path/to/RELLIS-3D  -ac ./config/arch/salsanext_ouster.yml -dc ./config/labels/rellis.yaml -n rellis -l ./logs -p ""
```

### KPConv
The KPConv is a fork from [https://github.com/HuguesTHOMAS/KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch)

To evaluate the dataset:
```
cd /path/to/code/benchmarks/KPConv-PyTorch-master
export PYTHONPATH=/path/to/code/benchmarks/KPConv-PyTorch-master/:$PYTHONPATH
python test_models.py
```
Configure ```benchmarks/KPConv-PyTorch-master/test_models.py``` before evaluation.
```
chosen_log = '/path/to/pretrained/model/folder'
config.sv_path = "/path/to/save/prediction"
config.data_path = "/path/to/RELLIS-3D"
```
Dowload pre-trained model ([Download 1GB](https://drive.google.com/file/d/1Exrt4yWDhgucx_vr08hAuXTcaLUcpXXm/view?usp=sharing))


To retrain the KPConv on RELLIS-3D:
To evaluate the dataset:
```
cd /path/to/code/benchmarks/KPConv-PyTorch-master
export PYTHONPATH=/path/to/code/benchmarks/KPConv-PyTorch-master/:$PYTHONPATH
python train_Rellis.py
```
Configure ```benchmarks/KPConv-PyTorch-master/train_Rellis.py``` before training.
```
data_path = "/path/to/RELLIS-3D"
```
