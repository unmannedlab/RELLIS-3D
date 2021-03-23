import os

import cv2
import numpy as np
from PIL import Image
import yaml
import argparse
from tqdm import tqdm


def convert_label(label, label_mapping, inverse=False):
    temp = label.copy()
    if inverse:
        for v,k in label_mapping.items():
            temp[label == k] = v
    else:
        for k, v in label_mapping.items():
            temp[label == k] = v
    return temp

def convert_color(label, color_map):
        temp = np.zeros(label.shape + (3,)).astype(np.uint8)
        for k,v in color_map.items():
            temp[label == k] = v
        return temp

def save_output(label_dir, output_dir, config_path):
    config_dict = yaml.safe_load(open(config_path, 'r'))
    color_map = config_dict['color_map']
    learning_map = {0: 0,
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

    label_list = os.listdir(label_dir)
    color_dir = os.path.join(output_dir,'color')
    id_dir = os.path.join(output_dir,'id')
    if not os.path.exists(color_dir):
        os.makedirs(color_dir)
    if not os.path.exists(id_dir):
        os.makedirs(id_dir)

    for label_path in tqdm(label_list):
        label = np.array(Image.open(os.path.join(label_dir, label_path)))
        label = label[:, :,0]
        label = convert_label(label, learning_map, True)
        color_label = convert_color(label, color_map)
        id_label = Image.fromarray(label)
        id_label.save(os.path.join(id_dir, label_path))
        color_label = Image.fromarray(color_label,'RGB')
        color_label.save(os.path.join(color_dir, label_path.replace("png",'jpg')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('label_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--config_path',default='./benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml')
    args = parser.parse_args()
    save_output(args.label_dir,args.output_dir,args.config_path)
    
