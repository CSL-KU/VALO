import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

model = None
deadline_sec = 9999.9
# Callback function to process the received point cloud message
def point_cloud_callback(msg):
    global model
    global deadline_sec
    # Convert ROS point cloud message to a NumPy array
    pc_data = pc2.read_points(msg, field_names=("x", "y", "z", "intensity", "ring", "time"), skip_nans=True)
    pc_array = np.array(list(pc_data)).astype(np.float32)

    pc_array = np.concatenate((
        np.zeros((pc_array.shape[0], 1), dtype=pc_array.dtype),
        pc_array[:, :4], # x y z i
        np.zeros((pc_array.shape[0], 1), dtype=pc_array.dtype),
    ), axis=1)
    # Print the first few points for demonstration
    #print("Point cloud:", pc_array.shape, pc_array.dtype)
    #print("First Few Points:\n", pc_array[:10, :])

    batch_dict = {
        'points': torch.from_numpy(pc_array).cuda(),  # b x y z i t
        'frame_id' : ['ignore'],
        'metadata' : [{'token': '???'}], # Projection won't work like that for anytime
        'batch_size' : 1,
        'deadline_sec': deadline_sec,
        'abs_deadline_sec': time.time() + deadline_sec,
    }
    batch_dict = model(batch_dict)
    torch.cuda.synchronize()
    pred_dict = batch_dict['final_box_dicts'][0]
    model.calc_elapsed_times()
    labels = pred_dict['pred_labels'].cpu()
    num_humans = torch.sum(labels == 9).item()
    if num_humans > 0:
        print('Human detected!', num_humans)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    #cfg.TAG = Path(args.cfg_file).stem
    #cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    #np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()

    log_file = ('./log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=0, logger=logger, training=False
    )

    global model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.eval()
        # Remove the hooks as we don't have a dataset
        model.pre_hook_handle.remove()
        model.post_hook_handle.remove()
        model.cuda()

    rospy.init_node("point_cloud_inference", anonymous=True)
    rospy.Subscriber("/velodyne_points", PointCloud2, point_cloud_callback)

    # Spin until Ctrl+C is pressed
    rospy.spin()


if __name__ == '__main__':
    main()
