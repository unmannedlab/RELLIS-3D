from __future__ import print_function
import rospy
import rosbag
import argparse
from pathlib import Path
import numpy as np
import tf2_ros
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import yaml
from tqdm import tqdm

def main(args):
    root_path = Path(args.root_dir)
    collection_path = root_path / Path(args.collection_name)

    bag_seq_map = {}
    bag_seq_path = collection_path / Path('bag_sequence_map.yaml')
    with open(str(bag_seq_path)) as f:
        bag_seq_map = yaml.load(f.read())

    tf_buff = tf2_ros.Buffer()
    for bagname, seq in bag_seq_map.items():
        tf_buff.clear()
        bag_path = collection_path / Path('bags') / Path(bagname)
        info_files = [s for s in (collection_path / Path(str(seq).zfill(5))).iterdir() if 'info.yaml' in s.name]
        sensor_names = [f.stem[:f.stem.find('_info')] for f in info_files]
        print('sensor_names', sensor_names)
        frame_ids = []
        for info_file in info_files:
            with info_file.open('r') as f:
                frame_id = yaml.load(f.read())['frame_id']
                frame_ids.append(frame_id)
        print('frame_ids',frame_ids)
        b = rosbag.Bag(str(bag_path))
        for topic, msg, t in b.read_messages(topics=['/tf','/tf_static']):
            if topic == '/tf_static':
                for m in msg.transforms:
                    tf_buff.set_transform_static(m, "default_authority")

            elif topic == '/tf':
                for m in msg.transforms:
                    tf_buff.set_transform(m, "default_authority")

        seq_tfs = {}
        for i, frame in tqdm(enumerate(frame_ids)):
            for j in range(i+1,len(frame_ids)):
                if tf_buff.can_transform(frame_ids[i], frame_ids[j], rospy.Time(0)):
                    tform_msg = TransformStamped()
                    tform_msg = tf_buff.lookup_transform(frame_ids[i], frame_ids[j], rospy.Time(0))
                    t = tform_msg.transform.translation
                    q = tform_msg.transform.rotation
                    seq_tfs[sensor_names[i]+'-'+sensor_names[j]] = {'t':{'x':t.x,'y':t.y,'z':t.z},
                                                                    'q':{'x':q.x,'y':q.y,'z':q.z,'w':q.w}}
        tf_seq_path = collection_path / Path(str(seq).zfill(5)) / Path('transforms.yaml')

        with open(str(tf_seq_path), 'w') as f:
            f.write(yaml.dump(seq_tfs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", default="",
                        help="dataset root dir")
    parser.add_argument("-n", "--collection_name", default="rellis",
                        help="collection name")
    args = parser.parse_args()
    main(args)
