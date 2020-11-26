#!/usr/bin/env python
from __future__ import print_function
import os
import argparse

import numpy as np
import rosbag

from ply_utils import PlyData, PlyElement
import sensor_msgs.point_cloud2 as pc2_proc

from tqdm import tqdm
import json
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext
from pathlib import Path
import yaml

def main(args):

    print("Extract PLY files from %s on topic %s into dir %s" % (args.bag_file,
                                                                 args.pcd_topic,
                                                                 args.output_dir))
    bag = rosbag.Bag(args.bag_file, "r")
    count = 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    field_to_type = {'x': np.float32, 'y': np.float32, 'z': np.float32,
                     'r': np.uint8, 'g': np.uint8, 'b': np.uint8,
                     'intensity': np.float32,
                     't': np.uint32,
                     'reflectivity': np.uint16,
                     'ring': np.uint8,
                     'noise': np.uint16,
                     'range': np.uint32,
                     'label': np.uint32}


    point_fields = []
    field_names = []
    skip_nans = not args.write_nan
    for topic, msg, t in bag.read_messages(topics=[args.pcd_topic]):
        point_fields = msg.fields
        field_names = [pf.name for pf in point_fields]
        yaml_info = {}
        yaml_info['data_type'] = 'pointcloud'
        yaml_info['width'] = msg.width
        yaml_info['height'] = msg.height
        yaml_info['fields'] = field_names
        yaml_info['frame_id'] = msg.header.frame_id
        print("pcd_info:", yaml_info)
        fname_dir = Path(args.output_dir)
        with open(str(fname_dir.parent / Path(fname_dir.stem+'_info.yaml')), 'w') as f:
            f.write(yaml.dump(yaml_info))
            # json.dump(yaml_info, f)
        break

    for topic, msg, t in tqdm(bag.read_messages(topics=[args.pcd_topic]),
                              total=bag.get_message_count(args.pcd_topic)):
        pts = pc2_proc.read_points(msg, field_names=field_names, skip_nans=skip_nans)
        dtype = [(pf.name, field_to_type[pf.name]) for pf in point_fields]
        pts_np= np.array([p for p in pts], dtype=dtype)
        el = PlyElement.describe(pts_np, 'vertex')
        stamp = msg.header.stamp
        stamp_ns_str = str(stamp.nsecs).zfill(9)
        out_file = os.path.join(args.output_dir, "frame%06i-%s_%s.ply" % (count, str(stamp.secs),
                                                                          stamp_ns_str[:3]))
        PlyData([el], text=args.write_ascii).write(out_file)
        count += 1

    print("Wrote {} PLY clouds to {}".format(count, args.output_dir))
    bag.close()

    return

if __name__ == '__main__':
    """Extract a folder of PLY from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract PLY from a ROS bag.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='''Example:
      ./extract_ply.py --bag_file ~/data/rcta_box.bag --output_dir ~/data/rcta_objects/box/00001/ply/ --pcd_topic /points''')
    parser.add_argument("--bag_file", default=None, help="Input ROS bag.")
    parser.add_argument("--rosbag_folder", default=None, help="Folder of synchronized rosbags to extract")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--pcd_topic", help="Point cloud topic.", default=None)
    parser.add_argument("--write_ascii", help="Write PLY in ascii format", action="store_true")
    parser.add_argument("--write_nan", help="Write NaN points", action="store_true")

    args = parser.parse_args()

    if args.rosbag_folder is not None:
        base_out_dir = args.output_dir
        bag_names = [f for f in listdir(args.rosbag_folder) if isfile(join(args.rosbag_folder, f))]
        for bag_name in bag_names:
            if bag_name[-11:] != '_synced.bag':
                continue

            args.bag_file = join(args.rosbag_folder, bag_name)
            args.output_dir = join(base_out_dir, bag_name[:-11], 'ply')
            main(args)
    else:
        main(args)
