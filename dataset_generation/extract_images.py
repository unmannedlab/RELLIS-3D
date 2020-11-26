#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

from __future__ import print_function
import os
import argparse

import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tqdm import tqdm
import json
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext
from pathlib import Path
import yaml

def main(args):

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic,
                                                          args.output_dir))

    if args.image_info_topic == None:
        print("For image topic", args.image_topic)
        args.image_info_topic = args.image_topic[:args.image_topic.rfind('/')] + "/camera_info"
    print("Info topic", args.image_info_topic)

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    size = None
    intrinsics = None
    distortion = None
    R = None
    P = None
    for topic, msg, t in bag.read_messages(topics=[args.image_info_topic]):
        print("camera_info:", msg)
        yaml_info = {}
        yaml_info['data_type'] = 'image'
        yaml_info['frame_id'] = msg.header.frame_id
        yaml_info['width'] = msg.width
        yaml_info['height'] = msg.height

        fx = msg.K[0]
        cx = msg.K[2]
        fy = msg.K[4]
        cy = msg.K[5]
        # distortion = msg.D
        # distortion_model = msg.distortion_model
        # yaml_info['intrinsic_matrix'] = [fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0]
        yaml_info['K'] = list(msg.K)
        yaml_info['D'] = list(msg.D)
        yaml_info['R'] = list(msg.R)
        yaml_info['P'] = list(msg.P)
        # size = (msg.width, msg.height)
        # intrinsics = np.array(msg.K, dtype=np.float64,
        #                          copy=True).reshape((3, 3))
        # distortion = np.array(msg.D, dtype=np.float64,
        #                          copy=True).reshape((len(msg.D), 1))
        # R = np.array(msg.R, dtype=np.float64, copy=True).reshape((3, 3))
        # P = np.array(msg.P, dtype=np.float64, copy=True).reshape((3, 4))
        fname_dir = Path(args.output_dir)
        with open(str(fname_dir.parent / Path(fname_dir.stem+'_info.yaml')), 'w') as f:
            f.write(yaml.dump(yaml_info))
            # json.dump(yaml_info, f)

        with open(os.path.join(args.output_dir + "/..", 'camera_info.txt'), 'w') as f:
            f.write("{} {} {} {}\n".format(fx, fy, cx, cy))
        break

    issued_warning = False
    issued_undistort_msg = False
    for topic, msg, t in tqdm(bag.read_messages(topics=[args.image_topic]), total=bag.get_message_count(args.image_topic)):
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=args.image_enc)
        except CvBridgeError as e:
            if not issued_warning:
                print ('Exception:', e, 'trying to force image type conversion')
            cv_img_raw = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_img_np = np.array(cv_img_raw)
            if 'depth' in args.image_topic:
                if np.nanmax(cv_img_np) < 1000:
                    if not issued_warning:
                        print ('Converting assumed depth image from m to mm, output will be 16uc1')
                    cv_img_mm = 1000.0 * cv_img_np
                    cv_img = cv_img_mm.astype(np.uint16)
                else:
                    cv_img = cv_img_np.astype(np.uint16)
            issued_warning = True

        if intrinsics is not None and args.undistort:
            if not issued_undistort_msg:
                print ('Undistorting raw input image')
                issued_undistort_msg = True
            cv_img_raw = cv_img
            cv_img = cv2.undistort(cv_img_raw, intrinsics, distortion)
        if np.any(np.isnan(cv_img)):
            print("Predicted image has NaNs")
        stamp = msg.header.stamp
        stamp_ns_str = str(stamp.nsecs).zfill(9)
        cv2.imwrite(os.path.join(args.output_dir,
                                 "frame%06i-%s_%s.%s" % (count, str(stamp.secs),
                                                         stamp_ns_str[:3], args.file_format)), cv_img)
        # print ("Wrote image %i" % count)

        count += 1

    print("Wrote {} images to {}".format(count,args.output_dir))
    bag.close()

    return

if __name__ == '__main__':
    # Default from xtion RGB: ./extract_images.py --bag_file ~/data/rcta_box.bag --output_dir ~/data/rcta_objects/box/00001/rgb/ --image_topic /rgb/image --image_enc bgr8 --file_format jpg
    # Default from xtion Depth: ./extract_images.py --bag_file ~/data/rcta_box.bag --output_dir ~/data/rcta_objects/box/00001/depth/ --image_topic /depth/image --image_enc mono16

    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='''Example:
      RGB:     ./extract_images.py --bag_file ~/data/rcta_box.bag --output_dir ~/data/rcta_objects/box/00001/rgb/ --image_topic /rgb/image --image_enc bgr8 --file_format jpg
      Depth:   ./extract_images.py --bag_file ~/data/rcta_box.bag --output_dir ~/data/rcta_objects/box/00001/depth/ --image_topic /depth/image --image_enc mono16''')
    parser.add_argument("--bag_file", default=None, help="Input ROS bag.")
    parser.add_argument("--rosbag_folder", default=None, help="Folder of synchronized rosbags to extract")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--image_topic", help="Image topic.", default=None)
    parser.add_argument("--image_info_topic", help="Info topic.", nargs='?', default=None)
    camera_type_param = parser.add_mutually_exclusive_group()
    camera_type_param.add_argument('--xtion', dest='camera_type', action='store_const', const='xtion', help='Assume that the topic names are for the xtion')
    camera_type_param.add_argument('--d435', dest='camera_type', action='store_const', const='d435', help='Assume that the topics are for the d435')
    parser.add_argument("--file_format", default='png', help="Image file format. Default 'png'")
    parser.add_argument("--image_enc", default='passthrough',
                        help="Image encoding. See cv_bridge tutorial for encodings. Default 'passthrough'")
    parser.add_argument("--undistort", default=False,
                        help="Undistort input images with given camera info")

    args = parser.parse_args()

    if args.rosbag_folder is not None:
        base_out_dir = args.output_dir
        bag_names = [f for f in listdir(args.rosbag_folder) if isfile(join(args.rosbag_folder, f))]
        for bag_name in bag_names:
            if bag_name[-11:] != '_synced.bag':
                continue

            args.bag_file = join(args.rosbag_folder, bag_name)
            print("Extracting depth from:", bag_name)
            args.output_dir = join(base_out_dir, bag_name[:-11], 'depth')
            args.image_enc = 'mono16'
            if args.camera_type == 'xtion':
                args.image_topic = '/depth/image'
            else:
                args.image_topic = '/camera/aligned_depth_to_color/image_raw'
            args.file_format = 'png'
            main(args)

            print("Extracting color from:", bag_name)
            args.output_dir = join(base_out_dir, bag_name[:-11], 'rgb')
            args.image_enc = 'bgr8'
            if args.camera_type == 'xtion':
                args.image_topic = '/rgb/image'
            else:
                args.image_topic = '/camera/color/image_rect_color'
            args.file_format = 'jpg'
            main(args)



    else:
        main(args)
