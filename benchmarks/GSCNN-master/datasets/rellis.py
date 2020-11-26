"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from collections import defaultdict
import math
import logging
import datasets.cityscapes_labels as cityscapes_labels
import json
from config import cfg
import torchvision.transforms as transforms
import datasets.edge_utils as edge_utils

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
num_classes = 19
ignore_label = 0
root = cfg.DATASET.RELLIS_DIR
list_paths = {'train':'train.lst','val':"val.lst",'test':'test.lst'}


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class Rellis(data.Dataset):

    def __init__(self, mode, joint_transform=None, sliding_crop=None,
                 transform=None, target_transform=None, dump_images=False,
                 cv_split=None, eval_mode=False, 
                 eval_scales=None, eval_flip=False):
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        self.root = root
        if eval_scales != None:
            self.eval_scales = [float(scale) for scale in eval_scales.split(",")]
        self.list_path = list_paths[mode]
        self.img_list = [line.strip().split() for line in open(root+self.list_path)]
        self.files = self.read_files()
        if len(self.files) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.54218053, 0.64250553, 0.56620195], [0.54218052, 0.64250552, 0.56620194])
        self.label_mapping = {0: 0,
                              1: 0,
                              3: 1,
                              4: 2,
                              5: 3,
                              6: 4,
                              7: 5,
                              8: 6,
                              9: 7,
                              10: 8,
                              12: 9,
                              15: 10,
                              17: 11,
                              18: 12,
                              19: 13,
                              23: 14,
                              27: 15,
                              29: 1,
                              30: 1,
                              31: 16,
                              32: 4,
                              33: 17,
                              34: 18}

    def _eval_get_item(self, img, mask, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool)+1):
            imgs = []
            if flip :
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w,h = img.size
                target_w, target_h = int(w * scale), int(h * scale) 
                resize_img =img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(tensor_img)
            return_imgs.append(imgs)
        return return_imgs, mask
        
    def read_files(self):
        files = []
        # if 'test' in self.mode:
        #     for item in self.img_list:
        #         image_path = item
        #         name = os.path.splitext(os.path.basename(image_path[0]))[0]
        #         files.append({
        #             "img": image_path[0],
        #             "name": name,
        #         })
        # else:
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files

    def convert_label(self, label, inverse=False):
        
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        img_name = item["name"]
        img_path = self.root + item['img']
        label_path = self.root + item["label"]

        img = Image.open(img_path).convert('RGB')

        mask = np.array(Image.open(label_path))
        mask = mask[:, :]
        

        mask_copy = self.convert_label(mask)

        if self.eval_mode:
            return self._eval_get_item(img, mask_copy, self.eval_scales, self.eval_flip), img_name


        mask = Image.fromarray(mask_copy.astype(np.uint8))
        # Image Transformations
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.mode == 'test':
            return img, mask, img_name, item['img']

        _edgemap = mask.numpy()
        _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)

        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)

        edgemap = torch.from_numpy(_edgemap).float()
        
	# Debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)
        return img, mask, edgemap, img_name

    def __len__(self):
        return len(self.files)


def make_dataset_video():
    img_dir_name = 'leftImg8bit_demoVideo'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit/demoVideo')
    items = []
    categories = os.listdir(img_path)
    for c in categories[1:]:
        c_items = [name.split('_leftImg8bit.png')[0] for name in
                   os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = os.path.join(img_path, c, it + '_leftImg8bit.png')
            items.append(item)
    return items


class CityScapesVideo(data.Dataset):

    def __init__(self, transform=None):
        self.imgs = make_dataset_video()
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imgs)

