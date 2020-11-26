#!/usr/bin/env python

import rosbag
import argparse
import shutil
import tf.transformations as tfx

def new_tf(tfs, tform):
    return tform.child_frame_id not in tfs

def conflicting_tf(tfs, tform):
    assert(not new_tf(tfs, tform))
    return tform.header.frame_id not in tfs[tform.child_frame_id]

def update_tf_chain(tfs, msg):
    for tform in msg.transforms:
        if new_tf(tfs,tform):
            tfs[tform.child_frame_id] = [tform.header.frame_id]
        elif conflicting_tf(tfs, tform):
            tfs[tform.child_frame_id].append(tform.header.frame_id)

def find_conflicts(tfs, parent_fix_frames,
                   child_fix_frames, all_child_frames):
    for ctf, ptfs in tfs.iteritems():
        if ctf in all_child_frames:
            for ptf in ptfs:
                if ptf not in parent_fix_frames:
                    parent_fix_frames.append(ptf)
                    child_fix_frames.append(ctf)
        else:
            all_child_frames.add(ctf)

def find_invert_tfs(conflicts, static_tfs, dynamic_tfs):
    for c, p in conflicts:
        if p in static_tfs:
            for pf in static_tfs[p]:
                conflicts.append((p,pf))
        elif p in dynamic_tfs:
            for pf in dynamic_tfs[p]:
                conflicts.append((p,pf))
    return conflicts

def fix_conflicts(b, bo, static_tfs, dynamic_tfs):
    parent_fix_frames = []
    child_fix_frames = []
    all_child_frames = set()
    find_conflicts(dynamic_tfs, parent_fix_frames,
                   child_fix_frames, all_child_frames)
    find_conflicts(static_tfs, parent_fix_frames,
                   child_fix_frames, all_child_frames)
    conflicts = zip(child_fix_frames, parent_fix_frames)
    all_conflicts = find_invert_tfs(conflicts, static_tfs, dynamic_tfs)

    print(all_conflicts)
    for topic, msg, t, cnxn_hdr in b.read_messages(return_connection_header=True):
        if topic == "/tf_static" or topic == "/tf":
            for i, tform in enumerate(msg.transforms):
                if (tform.child_frame_id, tform.header.frame_id) in all_conflicts:
                    msg.transforms[i] = flip_tform(tform)
        bo.write(topic, msg, t, connection_header=cnxn_hdr)


def flip_tform(tform_msg):
    ri = tfx.euler_from_quaternion([tform_msg.transform.rotation.x,
                                    tform_msg.transform.rotation.y,
                                    tform_msg.transform.rotation.z,
                                    tform_msg.transform.rotation.w])
    ti = [tform_msg.transform.translation.x,
          tform_msg.transform.translation.y,
          tform_msg.transform.translation.z]
    tform = tfx.compose_matrix(angles=ri, translate=ti)
    ro = tfx.quaternion_from_matrix(tfx.inverse_matrix(tform))
    to = tfx.translation_from_matrix(tfx.inverse_matrix(tform))
    tform_msg_out = tform_msg
    child_frame = tform_msg.child_frame_id
    parent_frame = tform_msg.header.frame_id
    tform_msg_out.child_frame_id = parent_frame
    tform_msg_out.header.frame_id = child_frame
    tform_msg_out.transform.translation.x = to[0]
    tform_msg_out.transform.translation.y = to[1]
    tform_msg_out.transform.translation.z = to[2]
    tform_msg_out.transform.rotation.x = ro[0]
    tform_msg_out.transform.rotation.y = ro[1]
    tform_msg_out.transform.rotation.z = ro[2]
    tform_msg_out.transform.rotation.w = ro[3]
    return tform_msg_out


def viz_tree(fname, static_tfs, dynamic_tfs):
    all_nodes = set(static_tfs.keys()+dynamic_tfs.keys())
    for vs in static_tfs.itervalues():
        for v in vs:
            all_nodes.add(v)
    for vs in dynamic_tfs.itervalues():
        for v in vs:
            all_nodes.add(v)
    with open(fname, 'w') as f:
        f.write('digraph {\n')
        for n in all_nodes:
            f.write('    {} [label={}]\n'.format(n, n))
        for k,vs in static_tfs.iteritems():
            for v in vs:
                f.write('    {} -> {}\n'.format(v,k))
        for k,vs in dynamic_tfs.iteritems():
            for v in vs:
                f.write('    {} -> {}\n'.format(v,k))
        f.write('}')

def main(args):
    b=rosbag.Bag(args.bag_file)
    tmp_bag = '/tmp/tmp.bag'
    bo=rosbag.Bag(tmp_bag, 'w')
    static_tfs = {}
    dynamic_tfs = {}
    for topic, msg, t in b.read_messages():
        if topic == "/tf_static":
            update_tf_chain(static_tfs, msg)
        elif topic == "/tf":
            update_tf_chain(dynamic_tfs, msg)
    viz_tree('/tmp/tree_before.dot',static_tfs, dynamic_tfs)

    fix_conflicts(b, bo, static_tfs, dynamic_tfs)
    b.close()
    bo.close()

    static_tfs = {}
    dynamic_tfs = {}
    bo=rosbag.Bag(tmp_bag)
    for topic, msg, t in bo.read_messages():
        if topic == "/tf_static":
            update_tf_chain(static_tfs, msg)
        elif topic == "/tf":
            update_tf_chain(dynamic_tfs, msg)
    viz_tree('/tmp/tree_after.dot', static_tfs, dynamic_tfs)

    out_bag = args.out_bag if args.out_bag else args.bag_file
    shutil.move(tmp_bag, out_bag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix tf tree conflicts retroactively",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag_file", default=None, help="Input ROS bag.")
    parser.add_argument("--out_bag", default=None, help="Output ROS bag. (Default is to _overwrite_ input bag)")
    args = parser.parse_args()
    main(args)
