#!/usr/bin/env python
from __future__ import print_function
import os
import rospy
import argparse
import urllib.request
import numpy as np
import rosbag
from skimage import io
from pathlib import Path
from tqdm import tqdm
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext
import json
import csv
import cv2
import yaml
# import tf2_ros
# from tf2_msgs.msg import TFMessage

def hex_to_rgb(h):
     h = h.lstrip('#')
     hlen = len(h)
     return tuple(int(h[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def hex_to_bgr(h):
    return tuple(reversed(hex_to_rgb(h)))

def get_ontology(ontology_path):
    id_color_map = {}
    id_rgb_map = {}
    id_name_map = {}
    if ontology_path.exists():
        ontology_lines = open(str(ontology_path),'rU').readlines()
        for parts in csv.reader(ontology_lines[1:], quotechar='"', delimiter=',',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True):
            color_hex = parts[0]
            lab_id = parts[3]
            lab_name = parts[2]#.split(' ')[0]
            id_color_map[int(lab_id)] = hex_to_bgr(color_hex)
            id_rgb_map[int(lab_id)] = hex_to_rgb(color_hex)
            id_name_map[int(lab_id)] = lab_name

        num_ontology_blocks = int(np.sqrt(len(id_color_map.keys())))
        if num_ontology_blocks**2 != len(id_color_map.keys()):
            num_ontology_blocks += 1
        block_sz = 200
        ontology_img = np.zeros((block_sz*num_ontology_blocks,block_sz*num_ontology_blocks,3), dtype=np.uint8)
        ont_block_idx = 0
        block_txt_map = {}
        block_color_map = {}
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        for o,c in id_color_map.items():
            j_idx = ont_block_idx // num_ontology_blocks
            i_idx = (ont_block_idx - j_idx * num_ontology_blocks) % num_ontology_blocks
            ont_start_pt = np.array([block_sz*i_idx,block_sz*j_idx])
            ont_end_pt = ont_start_pt + np.array([block_sz,block_sz])
            ont_text_pt = (ont_start_pt + ont_end_pt - np.array([block_sz,0])) // 2
            ontology_img[ont_start_pt[1]:ont_end_pt[1],ont_start_pt[0]:ont_end_pt[0],:] = c
            block_txt_map[tuple(ont_text_pt)] = id_name_map[o]+'('+str(o)+')'
            block_color_map[tuple(ont_text_pt)] = c
            ont_block_idx += 1
        for p,t in block_txt_map.items():
            font_sz, _ = cv2.getTextSize(t, font_type, font_scale, font_thickness)
            fscale = min(font_scale, block_sz / font_sz[0])
            cv2.putText(ontology_img, t, p, font_type, fscale, (0,0,0), font_thickness)

        cv2.imwrite(str(ontology_path.with_suffix('.png')), ontology_img)

        with open(str(ontology_path.with_suffix('.yaml')), 'w') as f:
            f.write(yaml.dump([id_name_map, id_rgb_map]))

        # dump_jsonl([id_name_map, id_rgb_map], str(ontology_path.with_suffix('.jsonl')))

    return (id_color_map, id_name_map)

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def stamp_str_extract(fname):
    stem = Path(fname).stem
    stamp_idx = stem.rfind('-')
    return stem[stamp_idx+1:]

def stamp_from_filename(fname):
    stem = Path(fname).stem
    msec_idx = stem.rfind('_')
    msecs = float(stem[msec_idx+1:])
    sec_idx = stem[:msec_idx].rfind('-')
    secs = float(stem[sec_idx+1:msec_idx])
    stamp = secs + (msecs / 1000.0)
    return stamp

def old_stamp_from_filename(fname):
    stem = Path(fname).stem
    msec_idx = stem.rfind('_')
    msecs = float(stem[msec_idx+1:])
    sec_idx = stem[:msec_idx].rfind('_')
    secs = float(stem[sec_idx+1:msec_idx])
    stamp = secs + (msecs / 1000.0)
    return stamp

def seq_in_bag(seq, bag):
    bag_start = bag.get_start_time()
    bag_end = bag.get_end_time()
    print('bag',bag_start,bag_end,'seq',seq[0],seq[1])
    return (seq[0] >= bag_start and seq[1] <= bag_end)

def stamp_in_seq(stamp, seq):
    return (seq[0]<= stamp and seq[1] >= stamp)

def get_associated_fname(seq, fname, args):
    fname_stamp = stamp_str_extract(fname)
    datadirs = [d for d in Path(seq).iterdir() if d.is_dir()]
    data_files = sorted([df for dd in datadirs for df in dd.iterdir()
                         if df.suffix in args.data_extensions])
    for df in data_files:
        if fname_stamp in str(df):
            return (df.parent.stem, df.stem)
    return None, None

def main(args):
    root_path = Path(args.root_dir)
    collection_path = root_path / Path(args.collection_name)

    # Get sequence start/end timestamps
    seq_stamp_map = {} # sequence id: (start time, end time)
    stamp_seq_map = {}
    sdirs = [d for d in collection_path.iterdir() if d.is_dir()]
    all_fnames = []
    all_stamps = []
    for sd in sdirs:
        print('SD',sd)
        if args.seq_dirs and str(sd.stem) not in args.seq_dirs:
            continue
        print(sd)
        ddl = [d for d in Path(sd).iterdir() if d.is_dir() and 'label_id' in str(d.stem)]
        print ('DDL',ddl)
        data_files = sorted([df for dd in ddl for df in dd.iterdir()
                             if df.suffix in args.data_extensions])
        stamps_sorted = sorted([stamp_from_filename(df) for df in data_files])
        fnames = [df for df in data_files]
        stamps = [stamp_from_filename(df) for df in fnames]
        if not stamps_sorted:
            continue
        all_fnames.extend([str(f) for f in fnames])
        all_stamps.extend(stamps)
        timestamp_begin = stamps_sorted[0]
        timestamp_end = stamps_sorted[-1]
        print('ts_begin',timestamp_begin,'ts_end',timestamp_end)
        seq_stamp_map[str(sd)] = (timestamp_begin, timestamp_end)
        stamp_seq_map[timestamp_begin] = str(sd)
    sorted_seqs = [stamp_seq_map[k] for k in sorted(stamp_seq_map.keys())]
    sorted_stamps = [seq_stamp_map[sk] for sk in sorted_seqs]
    print('All sequences in order: ',sorted_seqs)
    print('All stamps in order', sorted_stamps)
    print('Stamp differences', [sorted_stamps[i][0] - sorted_stamps[i-1][1]
                                for i in range(len(sorted_stamps)) if i>0])
    print('all',len(all_fnames))

    sort_idxs = np.argsort(all_stamps)
    all_stamps = np.array(all_stamps)[sort_idxs].tolist()
    all_fnames = np.array(all_fnames)[sort_idxs].tolist()
    print('ALL FNAMES',all_fnames[:10000])


    # The worst
    old_fnames = []
    old_stamps = []
    tmppath = Path("/data/tmp/mod3")
    old_fnames = [fn for fn in tmppath.glob('**/*.png')]
    old_stamps = [old_stamp_from_filename(df) for df in old_fnames]
    sorted_old = np.argsort(old_stamps)
    old_stamps = np.array(old_stamps)[sorted_old].tolist()
    old_fnames = np.array(old_fnames)[sorted_old].tolist()

    # Process Appen jsonlist files
    seq_json_map = {}
    seq_img_map = {}
    seq_ann_map = {}
    img_ann_map = {}

    for jlf in args.jsonl:
        jlines = load_jsonl(jlf)
        image_urls = []
        ann_urls = []
        for jl in tqdm(jlines):
            for jmt in jl['results']['judgments']:
                if jmt['data']['broken_link'] == 'true':
                    continue
                image_urls.append(jmt['unit_data']['image_url'])
                stamp = stamp_from_filename(jmt['unit_data']['image_url'])
                ann_urls.append(json.loads(jmt['data']['annotation'])['url'])
                img_ann_map[stamp] = json.loads(jmt['data']['annotation'])['url']
        timestamp_jl = stamp_from_filename(image_urls[len(image_urls) // 2])
        for sid, stamps in seq_stamp_map.items():
            if stamp_in_seq(timestamp_jl, stamps):
                seq_json_map.setdefault(sid, []).append([jlf])
                seq_img_map.setdefault(sid, []).extend(image_urls)
                seq_ann_map.setdefault(sid, []).extend(ann_urls)
        assert(len(image_urls) == len(ann_urls))

    missing=[]
    urls = []
    for k,v in img_ann_map.items():
         if k not in all_stamps and k not in old_stamps:
              for i,s in enumerate(all_stamps):
                   if s > k:
                       if i == 0:
                           break
                       missing.append(k)
                       fmti = str(i).zfill(6)
                       ns_str = str(round(1000*(k - int(k))))
                       out_stem = 'frame'+fmti+'_'+str(int(k))+'_'+ns_str+'.png'
                       out_name = tmppath / Path(out_stem)
                       print('URL REQ',v + ' ' + out_stem)
                       # urllib.request.urlretrieve(v, str(out_name))
                       urls.append(v + ' ' + out_stem)
                       break

    with open(str(tmppath / 'urls.txt'), 'w') as f:
        f.writelines("%s\n" % u for u in urls)
    with open(str(tmppath / 'ann_map.yaml'), 'w') as f:
        f.write(yaml.dump(img_ann_map))
    print('len missing',len(missing))
    print('len all',len(img_ann_map.keys()))
    assert(False)

    # Process Appen ontology file
    ontology_path = collection_path / Path('ontology.csv')
    id_color_map, id_name_map = get_ontology(ontology_path)

    # # Extract images
    # for sid, fnames in seq_img_map.items():
    #     print('Seq:',sid)
    #     for i, fname in tqdm(enumerate(fnames)):
    #         ext = Path(fname).suffix
    #         ext_out = ext if ext != '.jpg' else '.png'
    #         sensor_name, fname_stem = get_associated_fname(sid, fname, args)
    #         if sensor_name:
    #             out_name = collection_path / Path(sid) / Path(str(sensor_name)+'_label_id/'+fname_stem+ext_out)
    #             if not out_name.parent.exists():
    #                 out_name.parent.mkdir()
    #             try:
    #                 urllib.request.urlretrieve(seq_ann_map[sid][i], str(out_name))
    #                 if id_color_map:
    #                     color_file = collection_path / Path(sid) / Path(str(sensor_name)+'_label_color/'+fname_stem+ext_out)
    #                     if not color_file.parent.exists():
    #                         color_file.parent.mkdir()
    #                     img = io.imread(str(out_name))
    #                     img_flat = np.reshape(img, (-1,img.shape[-1]))
    #                     img_bgr_flat = np.array([id_color_map.get(p[0],(0,0,0)) for p in img_flat], dtype=np.uint8)
    #                     img_bgr = img_bgr_flat.reshape(img.shape[0],img.shape[1],3)
    #                     cv2.imwrite(str(color_file), img_bgr)
    #             except Exception as e:
    #                 try:
    #                     urllib.request.urlretrieve(seq_ann_map[sid][i], str(out_name))
    #                     if id_color_map:
    #                         color_file = collection_path / Path(sid) / Path(str(sensor_name)+'_label_color/'+fname_stem+ext_out)
    #                         if not color_file.parent.exists():
    #                             color_file.parent.mkdir()
    #                         img = io.imread(str(out_name))
    #                         img_flat = np.reshape(img, (-1,img.shape[-1]))
    #                         img_bgr_flat = np.array([id_color_map.get(p[0],(0,0,0)) for p in img_flat], dtype=np.uint8)
    #                         img_bgr = img_bgr_flat.reshape(img.shape[0],img.shape[1],3)
    #                         cv2.imwrite(str(color_file), img_bgr)
    #                 except Exception as e2:
    #                     print (e2)
    #                     continue


    # Check for bags in <root_dir>/<collection_name>/bags. Associate with sequences
    bag_path = collection_path / Path('bags')
    bag_seq_map={}
    if bag_path.exists():
        print('bag path',bag_path)
        bag_names = [b for b in bag_path.iterdir() if b.suffix == '.bag' and '_synced.bag' not in str(b)]
        for b in bag_names:
            bag = rosbag.Bag(str(b))
            for sid, stamps in seq_stamp_map.items():
                if seq_in_bag(stamps, bag):
                    bag_seq_map[b.name] = sid # assumes 1-1 bag/seq mapping (any split bags have already been merged)
                    # bag_seq_map.setdefault(b.name,[]).append([sid])

    bag_seq_path = collection_path / Path('bag_sequence_map.yaml')
    bag_seq_stem_map = {k:Path(v).stem for k,v in bag_seq_map.items()}
    with open(str(bag_seq_path), 'w') as f:
        f.write(yaml.dump(bag_seq_stem_map))

    for b,s in bag_seq_map.items():
        seq_map_path = collection_path / Path(s) / Path('source_bag.txt')
        seq_map_path.write_text(b)

    # For each sequence, get transform between sensors, write to file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Associate jsonlines files to sequences")
    parser.add_argument("-r", "--root_dir", default="",
                        help="dataset root dir")
    parser.add_argument("-n", "--collection_name", default="rellis",
                        help="collection name")
    parser.add_argument("-s", "--seq_dirs", nargs='+',default=[],
                        help="sequence_dirs to process, if empty, process entire collection")
    parser.add_argument("-j", "--jsonl", nargs='+', default=[],
                        help="Appen formatted JSONlines files.")
    parser.add_argument("-e", "--data_extensions", nargs='+',default=['.png','.jpg','.ply'],
                        help="set of supported extensions of raw data files")
    # parser.add_argument("-o", "--ontology_file", default="",
    #                     help="Appen formatted ontology file")
    # parser.add_argument("-c", "--color", action="store_true",
    #                     help="Output label color images as well as label id images")

    args = parser.parse_args()
    main(args)
