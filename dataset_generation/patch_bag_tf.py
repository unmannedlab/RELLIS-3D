import rospy
import rosbag
import numpy as np
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Transform, TransformStamped, Vector3, Quaternion
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
    parser.add_argument('-p', "--parent", default="ouster1/os1_lidar")
    parser.add_argument('-c', "--child", default="pylon_camera")
    args = parser.parse_args()

    b_path = Path(args.bag)
    b_in = rosbag.Bag(str(b_path))
    tmp_bag = '/tmp/tmp.bag'
    b_out=rosbag.Bag(tmp_bag, 'w')
    # b_out = rosbag.Bag(str(Path(args.out_dir) / Path(b_path.stem))+'.bag', 'w')

    updated_tf = False
    has_static = b_in.get_message_count(topic_filters=['/tf_static']) > 0
    R = np.array([[0.0299636,  0.999547, -0.00265867],
                  [-0.00909438,  -0.00238713,  -0.999956],
                  [-0.99951,  0.0299864,  0.00901874]])
    t = np.array([-0.0537767, -0.269452, -0.322621])
    T = tfx.identity_matrix()
    T[:3,:3] = R
    T[:3,3] = t
    t = tfx.translation_from_matrix(tfx.inverse_matrix(T))
    q = tfx.quaternion_from_matrix(tfx.inverse_matrix(T))
    transform_msg = Transform()
    transform_msg.translation = Vector3(x=t[0], y=t[1], z=t[2])
    transform_msg.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    for topic, msg, t, cnxn_hdr in b_in.read_messages(return_connection_header=True):
        if not has_static and not updated_tf:
            tf_msg = TFMessage()
            tf_stamped = TransformStamped()
            tf_stamped.header.frame_id = args.parent
            tf_stamped.header.stamp = t + rospy.Duration(0.5)
            tf_stamped.child_frame_id = args.child
            tf_stamped.transform = transform_msg
            tf_msg.transforms = [tf_stamped]
            b_out.write('/tf_static', tf_msg, t + rospy.Duration(0.5))
            updated_tf = True
        if has_static and topic == '/tf_static':
            tf_msg = msg
            tf_stamped = TransformStamped()
            tf_stamped.header.frame_id = args.parent
            tf_stamped.header.stamp = t + rospy.Duration(0.5)
            tf_stamped.child_frame_id = args.child
            tf_stamped.transform = transform_msg
            tf_msg.transforms.append(tf_stamped)
            msg = tf_msg
        b_out.write(topic, msg, t, connection_header=cnxn_hdr)
    b_in.close()
    b_out.close()

    out_bag = args.out_bag if args.out_bag else args.bag
    shutil.move(tmp_bag, out_bag)
