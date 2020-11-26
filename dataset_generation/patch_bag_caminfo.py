import rospy
import rosbag
import numpy as np
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Transform, TransformStamped, Vector3, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo
import tf.transformations as tfx
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm

def main(args):
    tmp_bag = "/tmp/temp.bag"
    overwrite = args.out_bag == args.bag or not args.out_bag
    b_out = (rosbag.Bag(tmp_bag, 'w') if overwrite else rosbag.Bag(args.out_bag, 'w'))
    b_in = rosbag.Bag(args.bag)
    b_cam_info = rosbag.Bag(args.cam_info_bag)

    cam_ns_msg_map = {}
    for topic, msg, t in b_cam_info.read_messages():
        if 'CameraInfo' in str(type(msg)):
            if "nerian" in topic and "right" in topic:
                topic = "/nerian/right/camera_info"
            if "nerian" in topic and "left" in topic:
                topic = "/nerian/left/camera_info"

            topic_path = Path(topic)
            cam_ns_msg_map[str(topic_path.parent)] = msg
            print (topic)
    b_cam_info.close()

    # print 'cam_ns_map',cam_ns_msg_map
    topics = b_in.get_type_and_topic_info()[1].keys()
    for topic, msg, t, cnxn_hdr in tqdm(b_in.read_messages(return_connection_header=True)):
        if 'Image' in str(type(msg)):
            if "nerian" in topic and "right" in topic:
                topic = "/nerian/right/image_raw"
            if "nerian" in topic and "left" in topic:
                topic = "/nerian/left/image_raw"

            topic_path = Path(topic)
            if (str(topic_path.parent) in cam_ns_msg_map and
                str(topic_path.parent)+"/camera_info" not in topics):
                cam_ns_msg_map[str(topic_path.parent)].header.stamp = msg.header.stamp
                b_out.write(str(topic_path.parent)+"/camera_info",
                            cam_ns_msg_map[str(topic_path.parent)], t)
        if 'CameraInfo' in str(type(msg)):
            topic_path = Path(topic)
            if (str(topic_path.parent) in cam_ns_msg_map):
                cam_ns_msg_map[str(topic_path.parent)].header.stamp = msg.header.stamp
                b_out.write(topic, cam_ns_msg_map[str(topic_path.parent)], t)
            else:
                b_out.write(topic, msg, t)
        elif 'tf_static' in topic:
            b_out.write(topic, msg, t, connection_header=cnxn_hdr)
        else:
            b_out.write(topic, msg, t)
    b_in.close()
    b_out.close()

    if overwrite:
        shutil.move(tmp_bag, args.bag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bag", default="")
    parser.add_argument('-c', "--cam_info_bag", default="")
    parser.add_argument("-o", "--out_bag", default="",
                        help="Output ROS bag. (Default is to _overwrite_ input bag)")
    args = parser.parse_args()
    main(args)
