#!/usr/bin/env python
from __future__ import print_function
import math
import os
import sys
import shutil
import argparse
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext
import rospy
from pydoc import locate
import rosbag
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import re
from tqdm import tqdm

def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [path + f for f in listdir(path)
                if isfile(join(path, f)) and splitext(f)[1] == extension]
    file_list = sorted_alphanum(file_list)
    return file_list


def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if exists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
    return path


def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    return path_color, path_depth


def get_rgbd_file_lists(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + \
            get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        shutil.rmtree(path_folder)
        makedirs(path_folder)


def check_folder_structure(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    assert exists(path_depth), \
            "Path %s is not exist!" % path_depth
    assert exists(path_color), \
            "Path %s is not exist!" % path_color

def sync_cb(*args):
    topics, sync_bag = args[-2:]
    assert(len(topics) == len(args[:-2]))
    for i,m in enumerate(args[:-2]):
        sync_bag.write(topics[i], m, m.header.stamp)

def synchronize_rosbag(args):
    rospy.init_node('sync_bag_messages')
    bag = rosbag.Bag(args.dataset)
    sync_bag = rosbag.Bag(os.path.splitext(args.dataset)[0]+'_synced.bag', 'w')
    topics = bag.get_type_and_topic_info()[1].keys()
    types = [tt[0] for tt in bag.get_type_and_topic_info()[1].values()]
    sync_filt_types = [(ttopic, ttype[0][ttype[0].rfind('/')+1:]) for
                       ttopic, ttype in bag.get_type_and_topic_info()[1].items()
                       if ttype in args.type_filter]
    sync_filt_topics = [(ttopic, ttype[0][ttype[0].rfind('/')+1:]) for
                        ttopic, ttype in bag.get_type_and_topic_info()[1].items()
                        if ttopic in args.topic_filter]
    sync_topics_types = set(sync_filt_types + sync_filt_topics)
    subscribers = [Subscriber(top, getattr(sys.modules[__name__], typ))
                   for top,typ in sync_topics_types]
    publishers = [rospy.Publisher(top, getattr(sys.modules[__name__], typ), queue_size=100)
                  for top,typ in sync_topics_types]
    if args.approx:
        sync = ApproximateTimeSynchronizer(subscribers, queue_size=100, slop=args.slop)
    else:
        sync = TimeSynchronizer(subscribers, queue_size=100)
    sync_topics = [top for top,typ in sync_topics_types]
    sync.registerCallback(sync_cb, sync_topics, sync_bag)
    rospy.sleep(0.1)
    for topic, msg, t in tqdm(bag.read_messages(), total=bag.get_message_count()):
        if rospy.is_shutdown():
            break
        if topic in sync_topics:
            idx = sync_topics.index(topic)
            publishers[idx].publish(msg)
            rospy.sleep(0.01)
        elif args.add_remaining_msgs:
            sync_bag.write(topic, msg, t)
    rospy.sleep(2.0)

    print('synced', sync_bag)

    for sub in subscribers:
        sub.unregister()
    bag.close()
    sync_bag.close()

def synchronize_redwood(args):
    folder_path = args.dataset
    color_files, depth_files = get_rgbd_file_lists(folder_path)
    if args.debug_mode:
        print(depth_files)
        print(color_files)

    # filename format is:
    # frame-timestamp.filetype
    timestamps = {'depth':[None] * len(depth_files),
            'color':[None] * len(color_files)}
    for i, name in enumerate(depth_files):
        depth_timestamp = int(os.path.basename(depth_files[i]).replace('-','.').split('.')[1])
        timestamps['depth'][i] = depth_timestamp
    for i, name in enumerate(color_files):
        color_timestamp = int(os.path.basename(color_files[i]).replace('-','.').split('.')[1])
        timestamps['color'][i] = color_timestamp

    # associations' index is the color frame, and the value at
    # that index is the best depth frame for the color frame
    associations = []
    depth_idx = 0
    for i in range(len(color_files)):
        best_dist = float('inf')
        while depth_idx <= len(depth_files)-1 and i <= len(color_files)-1:
            dist = math.fabs(timestamps['depth'][depth_idx] - \
                    timestamps['color'][i])
            if dist > best_dist:
                break
            best_dist = dist
            depth_idx += 1
            if depth_idx > timestamps['depth'][-1]:
                print("Ended at color frame %d, depth frame %d" % (i, depth_idx))
        associations.append(depth_idx-1)
        if args.debug_mode:
            print("%d %d %d %d" % (i, depth_idx-1,
                    timestamps['depth'][depth_idx-1], timestamps['color'][i]))

    os.rename(os.path.join(folder_path, "depth"),
            os.path.join(folder_path, "temp"))
    if not os.path.exists(os.path.join(folder_path, "depth")):
        os.makedirs(os.path.join(folder_path, "depth"))
    for i, assn in enumerate(associations):
        temp_name = os.path.join(folder_path, "temp",
                os.path.basename(depth_files[assn]))
        new_name = os.path.join(folder_path, "depth/%06d-%012d.png" % (i+1, timestamps['depth'][assn]))
        print("i %d, assn %d, name %s" % (i, assn, depth_files[assn]))
        if args.debug_mode:
            print(temp_name)
            print(new_name)
        if not exists(temp_name):
            assert(i+1 == len(color_files))
            os.remove(color_files[-1])
        else:
            os.rename(temp_name, new_name)
    shutil.rmtree(os.path.join(folder_path, "temp"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synchronize topics from rosbags, or color/depth from Redwood dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", help="path to the dataset (currently supported : Redwood dataset or rosbag)", default='redwood')
    parser.add_argument("--rosbag_folder", help="path to folder of rosbags to convert", default=None)
    parser.add_argument("--type_filter", help="sync messages of given types", nargs='+', default=[]) # 'sensor_msgs/CameraInfo','sensor_msgs/Image'
    parser.add_argument("--topic_filter", help="sync messages of given topics", nargs='+', default=[])
    parser.add_argument("--approx", help="approximate time sync (if false, use exact time sync)", action="store_true")
    parser.add_argument("--slop", help="approximate time sync slop, in seconds", type=float, default=0.05)
    parser.add_argument("--debug_mode", help="turn on debug mode", action="store_true")
    parser.add_argument("--add_remaining_msgs", help="preserve all messages not filtered by sync", action="store_true")
    args = parser.parse_args()

    if args.rosbag_folder is not None:
        bag_names = [f for f in listdir(args.rosbag_folder) if isfile(join(args.rosbag_folder, f))]
        for bag_name in bag_names:
            if bag_name[-4:] != ".bag" or bag_name[-11:] == '_synced.bag':
                continue
            args.dataset = join(args.rosbag_folder, bag_name)
            print("Synchronizing:", args.dataset)
            synchronize_rosbag(args)
    elif args.dataset.endswith('bag'):
        synchronize_rosbag(args)
    elif args.dataset == 'redwood':
        synchronize_redwood(args)
    else:
        print("Error: Unknown dataset type, ", args.dataset)
