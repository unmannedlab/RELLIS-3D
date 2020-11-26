#!/usr/bin/env python
from __future__ import print_function
import os
import rospy
import argparse

import numpy as np
import rosbag
from pathlib import Path
from tqdm import tqdm
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext


def get_seqs(args, stamp_range=None):
    if stamp_range:
        pass
    else:
        pass

def stamp_from_filename(fname):
    stem = Path(fname).stem
    msec_idx = stem.rfind('_')
    msecs = float(stem[msec_idx+1:])
    print('msecs',msecs)
    sec_idx = stem[:msec_idx].rfind('-')
    secs = float(stem[sec_idx+1:msec_idx])
    print('secs',secs)
    stamp = secs + (msecs / 1000.0)
    print('stamp',stamp)
    return stamp

def seq_in_bag(seq, bag):
    bag_start = bag.get_start_time()
    bag_end = bag.get_end_time()
    return (seq[0] >= bag_start and seq[1] <= bag_end)

def main(args):
    collection_dir = Path(args.collection_dir)
    seq_stamp_map = {} # sequence id: (start time, end time)
    stamp_seq_map = {}
    seq_dirs = [d for d in collection_dir.iterdir() if d.is_dir()]
    for sd in seq_dirs:
        ddl = [d for d in sd.iterdir() if d.is_dir()]
        data_files = sorted([df for dd in ddl for df in dd.iterdir()
                             if df.suffix in args.data_extensions])
        timestamp_begin = stamp_from_filename(data_files[0])
        timestamp_end = stamp_from_filename(data_files[-1])
        print('ts_begin',timestamp_begin,'ts_end',timestamp_end)
        seq_stamp_map[str(sd)] = (timestamp_begin, timestamp_end)
        stamp_seq_map[timestamp_begin] = str(sd)
    sorted_seqs = [stamp_seq_map[k] for k in sorted(stamp_seq_map.keys())]
    sorted_stamps = [seq_stamp_map[sk] for sk in sorted_seqs]
    print('All sequences in order: ',sorted_seqs)
    print('All stamps in order', sorted_stamps)
    print('Stamp differences', [sorted_stamps[i][0] - sorted_stamps[i-1][1]
                                for i in range(len(sorted_stamps)) if i>0])
    # print('Combined',zip(sorted_seqs,sorted_stamps))
    if args.bags:
        bag_seqs={}
        for b in args.bags:
            bp = Path(b)
            bag = rosbag.Bag(b)
            for sid, stamps in seq_stamp_map.items():
                if seq_in_bag(stamps, bag):
                    bag_seqs.setdefault(bp.stem,[]).append([sid])
        print ('bag seq map', bag_seqs)

if __name__ == '__main__':
    """Sort filenames by timestamp. Filename naming convention is assumed to be
    of the form <somename>-secs_msec.<extension>
    """
    parser = argparse.ArgumentParser(description="Sort multiple sequences according to timestamps of their contained data")
    parser.add_argument("-d", "--collection_dir", default=None,
                        help="Parent dir of sequences.")
    # parser.add_argument("--reorder_sequences", action="store_true",
    #                     help="Renames sequences according to their timestamp order")
    parser.add_argument("-e", "--data_extensions", nargs='+',default=['.png','.jpg','.ply'],
                        help="set of supported extensions of raw data files")
    parser.add_argument("-b", "--bags", nargs='+', default=[],
                        help="output the set of sequences that correspond to the input list of rosbags")

    args = parser.parse_args()
    main(args)
