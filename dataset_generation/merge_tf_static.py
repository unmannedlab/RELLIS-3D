import rospy
import rosbag
import numpy as np
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Transform, TransformStamped, Vector3, Quaternion
from std_msgs.msg import Header
import tf.transformations as tfx
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil

def main(args):
    tmp_bag = "/tmp/temp.bag"
    overwrite = args.out_bag in args.bags or not args.out_bag
    b_out = rosbag.Bag(tmp_bag, 'w') if overwrite else rosbag.Bag(args.out_bag, 'w')

    tf_static_msg = None
    tf_static_t = None
    tf_static_connection_header = None

    time_min = 1e15
    bag_min = None
    if not args.ref_bag:
        for b in args.bags:
            b_in = rosbag.Bag(b)
            print 'Start time ',b_in.get_start_time()
            if b_in.get_start_time() < time_min:
                time_min = b_in.get_start_time()
                bag_min = b
            b_in.close()
        args.ref_bag = bag_min

    for b in args.bags:
        reference_bag = False
        b_in = rosbag.Bag(b)
        print b
        if args.ref_bag and args.ref_bag == b:
            reference_bag = True
            tf_static_t = b_in.get_start_time() + 1.0
        for topic, msg, t, cnxn_hdr in tqdm(b_in.read_messages(return_connection_header=True)):
            if topic == '/tf_static':
                if reference_bag:
                    tf_static_connection_header = cnxn_hdr
                if tf_static_msg:
                    tf_static_msg.transforms.extend(msg.transforms)
                else:
                    tf_static_msg = msg
            else:
                if reference_bag:
                    b_out.write(topic, msg, t)
        b_in.close()

    for i, tfm in enumerate(tf_static_msg.transforms):
        tf_static_msg.transforms[i].header.stamp = rospy.Time.from_sec(tf_static_t)
    b_out.write('/tf_static', tf_static_msg, rospy.Time.from_sec(tf_static_t),
                connection_header=tf_static_connection_header)
    b_out.close()

    if overwrite:
        out_bag = args.bags[-1] if not args.out_bag else args.out_bag
        shutil.move(tmp_bag, out_bag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bags", nargs='+', default=[], help="bags to merge")
    parser.add_argument("-r", "--ref_bag", default="", help="reference bag (timestamp is relative to this bag). Default is earliest time bag in the list")
    parser.add_argument("-o", "--out_bag", default="",
                        help="Output ROS bag. (Default is to _overwrite_ last input bag)")
    args = parser.parse_args()
    main(args)
