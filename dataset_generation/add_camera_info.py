import rospy
import rosbag
import numpy as np
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Transform, TransformStamped, Vector3, Quaternion
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
import tf.transformations as tfx
from pathlib import Path
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bag", default="")
    parser.add_argument("-o", "--out_bag", default="",
                        help="Output ROS bag. (Default is to _overwrite_ input bag)")
    parser.add_argument("-c", "--camera", default="pylon_camera")
    args = parser.parse_args()

    b_path = Path(args.bag)
    b_in = rosbag.Bag(str(b_path))
    tmp_bag = '/tmp/tmp.bag'
    b_out=rosbag.Bag(tmp_bag, 'w')
    # b_out = rosbag.Bag(str(Path(args.out_dir) / Path(b_path.stem))+'.bag', 'w')

    ci = CameraInfo()
    ci.distortion_model = 'plumb_bob'

    ci.D = 5*[0]
    ci.D[0] = -0.1343
    ci.D[1] = -0.0259
    ci.D[2] = 0.0021
    ci.D[3] = 0.0008

    ci.K = 9*[0]
    ci.K[0] = 2813.64
    ci.K[2] = 969.28
    ci.K[4] = 2808.33
    ci.K[5] = 624.05
    ci.K[8] = 1.0

    ci.R = 9*[0]
    ci.R[0] = 1.0
    ci.R[4] = 1.0
    ci.R[8] = 1.0

    ci.P = 12*[0]
    ci.P[0] = 2766.88
    ci.P[2] = 970.50
    ci.P[5] = 2790.28
    ci.P[6] = 625.22
    ci.P[10] = 1.0

    for topic, msg, t, cnxn_hdr in b_in.read_messages(return_connection_header=True):
        if args.camera in topic and 'Image' in str(type(msg)):
            ci.header = msg.header
            topic_ns = Path(topic).parent
            cam_info = topic_ns / Path('camera_info')
            b_out.write(str(cam_info), ci, t)
        b_out.write(topic, msg, t)
    b_in.close()
    b_out.close()

    out_bag = args.out_bag if args.out_bag else args.bag
    shutil.move(tmp_bag, out_bag)
