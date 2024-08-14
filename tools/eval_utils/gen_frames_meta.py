from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from centerpoint_tracker import CenterpointTracker as Tracker
from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", type=str, default=".",
            help="the dir to save logs and tracking results")
    parser.add_argument("--root", type=str, default="../data/nuscenes")
    parser.add_argument("--version", type=str, default='v1.0-mini')

    args = parser.parse_args()

    return args


def save_first_frame():
    args = parse_args()
    root= os.path.join(args.root, args.version)
    nusc = NuScenes(version=args.version, dataroot=root, verbose=True)
    if args.version == 'v1.0-trainval':
        scenes = splits.val
    elif args.version == 'v1.0-test':
        scenes = splits.test 
    elif args.version == 'v1.0-mini':
        scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name'] 
        if scene_name not in scenes:
            continue 

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp 

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True 
        else:
            frame['first'] = False 
        frames.append(frame)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)

if __name__ == '__main__':
    save_first_frame()
