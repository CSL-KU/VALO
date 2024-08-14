from .detector3d_template import Detector3DTemplate
from .anytime_calibrator import AnytimeCalibrator, get_stats
from .sched_helpers import *
import torch
from nuscenes.nuscenes import NuScenes
import time
import sys
import json
import numpy as np
import scipy
import gc
import copy
import os

from ...ops.cuda_projection import cuda_projection
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
from .. import load_data_to_gpu

from typing import List

@torch.jit.script
def do_inds_calc(vinds : List[torch.Tensor], vcount_area : torch.Tensor, \
        tcount : int, dividers: torch.Tensor):
    # how would be the overhead if I create a stream here?
    outp = []
    outp.append(vcount_area[:tcount])
    for i, vind in enumerate(vinds):
        voxel_tile_coords = torch.div(vind[:, -1], dividers[i+1], \
                rounding_mode='trunc').int()
        outp.append(torch.bincount(voxel_tile_coords, \
                minlength=tcount)[:tcount].unsqueeze(0))
    return torch.cat(outp, dim=0).cpu() # num sparse layer groups x 18


# The fork call must be done in torch scripted function for it to be async
@torch.jit.script
def do_inds_calc_wrapper(vinds : List[torch.Tensor], \
        vcount_area : torch.Tensor, \
        tcount : int, dividers: torch.Tensor):
    fut = torch.jit.fork(do_inds_calc, vinds, vcount_area, tcount, dividers)
    return fut

class AnytimeTemplateV2(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.sched_disabled = (self.model_cfg.METHOD == SchedAlgo.RoundRobin_NoSchedNoProj)

        self.keep_projection_disabled = (self.model_cfg.METHOD == SchedAlgo.RoundRobin_NoProj or \
                self.model_cfg.METHOD == SchedAlgo.RoundRobin_NoSchedNoProj)

        self.use_voxelnext = (self.model_cfg.METHOD == SchedAlgo.RoundRobin_VN or \
                self.model_cfg.METHOD == SchedAlgo.RoundRobin_VN_BLTP)

        self.use_baseline_bb3d_predictor = (self.model_cfg.METHOD == SchedAlgo.RoundRobin_BLTP or \
                self.model_cfg.METHOD == SchedAlgo.RoundRobin_VN_BLTP)
        if self.use_baseline_bb3d_predictor:
            print('***** Using baseline time predictor! *****')

        self.enable_tile_drop = (self.model_cfg.METHOD != SchedAlgo.RoundRobin_NoTileDrop) and \
            not self.use_voxelnext

        self.sched_algo = SchedAlgo.ProjectionOnly if self.model_cfg.METHOD == \
                SchedAlgo.ProjectionOnly else SchedAlgo.RoundRobin

        self.projection_coeff = float(self.model_cfg.PROJECTION_COEFF)
        print('Projection coefficient is', self.projection_coeff)

        if 'BACKBONE_2D' in self.model_cfg:
            self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
            self.model_cfg.BACKBONE_2D.METHOD = self.sched_algo
        if 'DENSE_HEAD' in self.model_cfg:
            self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
            self.model_cfg.DENSE_HEAD.METHOD = self.sched_algo
        torch.backends.cudnn.benchmark = True
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()
#        torch.use_deterministic_algorithms(True)

        ################################################################################
        self.tcount = self.model_cfg.TILE_COUNT
        self.tcount_cuda = torch.tensor(self.model_cfg.TILE_COUNT).long().cuda()
        #self.total_num_tiles = self.tcount

        # divide the tiles in X axis only
        self.tile_size_voxels = torch.tensor(\
                self.dataset.grid_size[1] / self.tcount).float().item()
        self.backbone_3d.tile_size_voxels = self.tile_size_voxels

        ####Projection###
        self.enable_projection = False

        self.clear_add_dict()

        self.init_tile_coord = 0
        if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
            self.init_tile_coord = -1
        elif self.sched_algo == SchedAlgo.MirrorRR:
            m2 = self.tcount//2
            m1 = m2 - 1
            self.mtiles = np.array([m1, m2], dtype=np.int32)
        self.last_tile_coord = self.init_tile_coord

        self.past_detections = {}  #{'num_dets': []}
        if self.sched_algo == SchedAlgo.ProjectionOnly:
            self.past_poses = []
            self.past_ts = []
        else:
            # Poses include [cst(3) csr(4) ept(3) epr(4)]
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)
            self.num_dets_per_tile = torch.zeros([self.tcount], dtype=torch.long)

        # Needs to be calibrated
        self.score_thresh = self.model_cfg.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH

        total_num_classes = sum([m.size(0) for m in self.dense_head.class_id_mapping_each_head])
        self.cls_id_to_det_head_idx_map = torch.zeros((total_num_classes,), dtype=torch.int)
        self.num_det_heads = len(self.dense_head.class_id_mapping_each_head)
        for i, cls_ids in enumerate(self.dense_head.class_id_mapping_each_head):
            for cls_id in cls_ids:
                self.cls_id_to_det_head_idx_map[cls_id] = i
        #self.cls_id_to_det_head_idx_map = self.cls_id_to_det_head_idx_map.cuda()

        if self.sched_algo == SchedAlgo.AdaptiveRR:
            self.processing_time_limit_sec = 0.650 # Every x ms, reset
            self.sched_reset()

        self.pc_range = self.vfe.point_cloud_range.cpu()
        self.projection_stream = torch.cuda.Stream()

        ##########################################
        ng = self.backbone_3d.num_layer_groups
        ng += 1 if self.use_voxelnext else 0

        self.dividers = self.backbone_3d.get_inds_dividers(self.tile_size_voxels)
        if self.use_voxelnext:
            self.dividers.append(float(self.tile_size_voxels / 8))
        self.dividers.insert(0, 1.) # this will be ignored
        print('dividers', self.dividers, 'ng', ng)
        self.move_indscalc_to_init = True
        ##########################################

    def clear_add_dict(self):
        self.add_dict['bb3d_layer_times'] = []
        self.add_dict['bb3d_preds'] = []
        self.add_dict['bb3d_preds_layerwise'] = []
        self.add_dict['bb3d_voxel_nums'] = []
        self.add_dict['bb3d_voxel_preds'] = []
        self.add_dict['nonempty_tiles'] = []
        self.add_dict['chosen_tiles_1'] = []
        self.add_dict['chosen_tiles_2'] = []

    def initialize(self, latest_token : str) -> (float, bool):
        deadline_sec_override, reset = super().initialize(latest_token)
        if reset and self.enable_projection:
            self.projection_reset()

        if self.sched_disabled:
            return deadline_sec_override, reset
        elif reset:
            self.sched_reset()

        if self.move_indscalc_to_init and self.latest_batch_dict is not None and \
                'bb3d_intermediary_vinds' in self.latest_batch_dict:
            self.fut = do_inds_calc_wrapper(
                    self.latest_batch_dict['bb3d_intermediary_vinds'],
                    self.latest_batch_dict['vcount_area'],
                    self.tcount, torch.tensor(self.dividers))

        return deadline_sec_override, reset

    def get_nonempty_tiles(self, voxel_coords, training=False):
        # Calculate where each voxel resides in which tile
        voxel_tile_coords = torch.div(voxel_coords[:, -1], self.tile_size_voxels, \
                rounding_mode='trunc').long()

        if training:
            nonempty_tile_coords = torch.unique(voxel_tile_coords, sorted=True)
            return nonempty_tile_coords
        else:
            nonempty_tile_coords, voxel_counts = torch.unique(voxel_tile_coords, \
                    sorted=True, return_counts=True)

            netc = nonempty_tile_coords.cpu().numpy()
            voxel_counts = voxel_counts.cpu().numpy()

            return voxel_tile_coords, netc, voxel_counts

    def schedule1(self, batch_dict):
        voxel_coords = batch_dict['voxel_coords']
        if self.training or self.sched_disabled:
            batch_dict['chosen_tile_coords'] = self.get_nonempty_tiles(voxel_coords, True)
            return batch_dict

        voxel_tile_coords, netc, netc_vcounts = self.get_nonempty_tiles(voxel_coords)
        vcount_area = np.zeros((self.tcount,), dtype=netc_vcounts.dtype)
        vcount_area[netc] = netc_vcounts
        vcount_area = np.expand_dims(vcount_area, 0)
        batch_dict['vcount_area'] = torch.from_numpy(vcount_area).int().cuda()

        if self.sched_algo == SchedAlgo.MirrorRR:
            netc, netc_vcounts= fill_tile_gaps(netc, netc_vcounts)
        elif self.sched_algo == SchedAlgo.AdaptiveRR:
            latest_token = batch_dict['metadata'][0]['token']
            cur_ts = self.token_to_ts[latest_token]

            # upper limit 25%
            if self.num_blacklisted_tiles > netc.shape[0] - int(self.tcount//4):
                self.num_blacklisted_tiles = netc.shape[0] - int(self.tcount//4)

            if self.reset_ts is not None:
                elapsed_time_sec = (cur_ts - self.reset_ts) / 1000000.0
                if elapsed_time_sec > self.processing_time_limit_sec:
                    # Reset
                    if self.processed_area_perc < 1.0:
                        # We need to blacklist tiles
                        self.num_blacklisted_tiles += int(np.ceil(\
                                (netc.shape[0] - self.num_blacklisted_tiles) * \
                                (1.0 - self.processed_area_perc)))
                        self.num_blacklisted_tiles = min(self.num_blacklisted_tiles, \
                                netc.shape[0] - int(self.tcount//4))
                    elif self.processed_area_perc > 1.0:
                        self.num_blacklisted_tiles -= int(np.ceil(\
                                (netc.shape[0] - self.num_blacklisted_tiles) * \
                                (self.processed_area_perc - 1.0)))
                        self.num_blacklisted_tiles = max(self.num_blacklisted_tiles, 0)

                    self.reset_ts = cur_ts
                    self.processed_area_perc = 0.

            else:
                self.reset_ts = cur_ts

            if self.num_blacklisted_tiles > 0:
                lptr, rptr = 0, netc_vcounts.shape[0]-1
                bt = self.num_blacklisted_tiles
                while bt > 0 and rptr - lptr > 0:
                    if netc_vcounts[lptr] < netc_vcounts[rptr]:
                        lptr += 1
                    else:
                        rptr -= 1
                    bt -= 1
                netc = netc[lptr:rptr+1]
                netc_vcounts = netc_vcounts[lptr:rptr+1]

            self.cur_netc_num_tiles = netc.shape[0]

        batch_dict['nonempty_tile_coords'] = netc

        if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
            num_tiles, tiles_queue = round_robin_sched_helper(
                    netc, self.last_tile_coord, self.tcount)

        if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.MirrorRR or \
                self.sched_algo == SchedAlgo.AdaptiveRR:
            batch_dict['tiles_queue'] = tiles_queue
            self.add_dict['nonempty_tiles'].append(netc.tolist())

            if self.move_indscalc_to_init and self.latest_batch_dict is not None and \
                    'bb3d_intermediary_vinds' in self.latest_batch_dict:
                voxel_dists = torch.jit.wait(self.fut)
                self.calibrator.commit_bb3d_updates(self.latest_batch_dict['chosen_tile_coords'], \
                    voxel_dists.numpy())

            bb3d_times_layerwise, post_bb3d_times, num_voxel_preds = self.calibrator.pred_req_times_ms(\
                    vcount_area, tiles_queue, num_tiles)
            if not self.use_baseline_bb3d_predictor:
                bb3d_times = np.sum(bb3d_times_layerwise, axis=-1)
            else:
                bb3d_times = bb3d_times_layerwise
            batch_dict['post_bb3d_times'] = post_bb3d_times
            tpreds = bb3d_times + post_bb3d_times
            psched_start_time = time.time()
            rem_time_ms = (batch_dict['abs_deadline_sec'] - psched_start_time) * 1000

            # Choose configuration that can meet the deadline, that's it
            diffs = tpreds < rem_time_ms

            ##### MANUAL OVERRIDE
            #tiles_to_run = 4
            #for idx, nt in enumerate(num_tiles):
            #    if nt >= tiles_to_run:
            #        tiles_idx = idx + 1
            #        break
            #####

            if (not self.is_calibrating() and diffs[-1]) or (self.is_calibrating() and \
                    len(netc) <= self.calibrator.get_chosen_tile_num()):
                # choose all
                chosen_tile_coords = netc
                if self.sched_algo == SchedAlgo.MirrorRR:
                    self.last_tile_coord = self.init_tile_coord
                tiles_idx=0
                #print('\nPREdicted voxels:')
                #print(num_voxel_preds[-1].astype(np.int).flatten())
                predicted_bb3d_time = float(bb3d_times[-1])
                predicted_bb3d_time_layerwise = bb3d_times_layerwise[-1].tolist()
                predicted_voxels = num_voxel_preds[-1].astype(np.int).flatten()
            else:
                if self.is_calibrating():
                    tiles_idx = self.calibrator.get_chosen_tile_num()
                    if tiles_idx >= len(diffs):
                        tiles_idx = len(diffs)
                else:
                    tiles_idx=1
                    while tiles_idx < diffs.shape[0] and diffs[tiles_idx]:
                        tiles_idx += 1

                #print('\npredicted voxels:')
                #print(num_voxel_preds[tiles_idx-1].astype(np.int).flatten())
                predicted_bb3d_time = float(bb3d_times[tiles_idx-1])
                predicted_bb3d_time_layerwise = bb3d_times_layerwise[tiles_idx-1].tolist()
                predicted_voxels = num_voxel_preds[tiles_idx-1].astype(np.int).flatten()

                # Voxel filtering is needed
                if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
                    chosen_tile_coords = tiles_queue[:tiles_idx]
                    self.last_tile_coord = chosen_tile_coords[-1].item()
                else:
                    chosen_tile_coords = np.concatenate((self.mtiles, tiles_queue[:tiles_idx-1]))

                tile_filter = cuda_point_tile_mask.point_tile_mask(voxel_tile_coords, \
                        torch.from_numpy(chosen_tile_coords).cuda())

                if 'voxel_features' in batch_dict:
                    batch_dict['voxel_features'] = \
                            batch_dict['voxel_features'][tile_filter].contiguous()
                batch_dict['voxel_coords'] = voxel_coords[tile_filter].contiguous()

            self.add_dict['bb3d_preds'].append(predicted_bb3d_time)
            self.add_dict['bb3d_preds_layerwise'].append(predicted_bb3d_time_layerwise)
            self.add_dict['bb3d_voxel_preds'].append(predicted_voxels.tolist())
            batch_dict['chosen_tile_coords'] = chosen_tile_coords
            self.add_dict['chosen_tiles_1'].append(chosen_tile_coords.tolist())
        elif self.sched_algo == SchedAlgo.ProjectionOnly:
            batch_dict['chosen_tile_coords'] = netc

        batch_dict['record_int_vcoords'] = not self.sched_disabled and \
                not self.use_baseline_bb3d_predictor and \
                not self.move_indscalc_to_init
        batch_dict['record_int_indices'] = not self.sched_disabled and \
                not self.use_baseline_bb3d_predictor and \
                self.move_indscalc_to_init
        batch_dict['record_time'] = True
        batch_dict['tile_size_voxels'] = self.tile_size_voxels
        batch_dict['num_tiles'] = self.tcount

        return batch_dict

    # Recalculate chosen tiles based on the time spent on bb3d
    def schedule2(self, batch_dict):
        if self.sched_disabled:
            return batch_dict

        # bb3d time predictor commit
        if not self.use_baseline_bb3d_predictor:
            if not self.move_indscalc_to_init:
                vcoords = batch_dict['bb3d_intermediary_vcoords']
                if self.use_voxelnext:
                    out = batch_dict['encoded_spconv_tensor']
                    voxel_tile_coords = torch.div(out.indices[:, -1], self.tile_size_voxels / 8, \
                            rounding_mode='trunc').int()
                    voxel_dist = torch.bincount(voxel_tile_coords, minlength=self.tcount)[:self.tcount]
                    # just for the sake of timing
                    vcoords.append(voxel_dist.unsqueeze(0))

                vcoords.insert(0, batch_dict['vcount_area'])
                voxel_dists = torch.cat(vcoords, dim=0).cpu().numpy() # num sparse layer groups x 18
                self.calibrator.commit_bb3d_updates(batch_dict['chosen_tile_coords'], voxel_dists)
            else:
                if self.use_voxelnext:
                    out = batch_dict['encoded_spconv_tensor']
                    batch_dict['bb3d_intermediary_vinds'].append(out.indices)
            #print('\nactual voxels:')
            num_voxels_actual = np.array([batch_dict['voxel_coords'].size(0)] + \
                    [inds.size(0) for inds in batch_dict['bb3d_intermediary_vinds']], dtype=np.int)
            #print(num_voxels_actual)
            self.add_dict['bb3d_voxel_nums'].append(num_voxels_actual.tolist())
        else:
            self.add_dict['bb3d_voxel_nums'].append([batch_dict['voxel_coords'].size(0)])

        # Tile dropping
        if self.enable_tile_drop:
            torch.cuda.synchronize()
            post_bb3d_times = batch_dict['post_bb3d_times']
            rem_time_ms = (batch_dict['abs_deadline_sec'] - time.time()) * 1000
            diffs = post_bb3d_times < rem_time_ms

            m = int(self.sched_algo == SchedAlgo.MirrorRR)
            if not diffs[batch_dict['chosen_tile_coords'].shape[0]-1-m]:
                tiles_idx=1
                while tiles_idx < diffs.shape[0] and diffs[tiles_idx]:
                    tiles_idx += 1

                ctc = batch_dict['tiles_queue'][:tiles_idx-m]
                if self.sched_algo == SchedAlgo.MirrorRR:
                    batch_dict['chosen_tile_coords'] = np.concatenate((self.mtiles, ctc))
                elif self.sched_algo == SchedAlgo.RoundRobin or \
                        self.sched_algo == SchedAlgo.AdaptiveRR:
                    batch_dict['chosen_tile_coords'] = ctc
                    self.last_tile_coord = ctc[-1].item()

            if self.sched_algo == SchedAlgo.MirrorRR and \
                    batch_dict['chosen_tile_coords'].shape[0] > self.mtiles.shape[0]:
                self.last_tile_coord = batch_dict['chosen_tile_coords'][-1].item()
        ctc = batch_dict['chosen_tile_coords'].tolist()
        self.add_dict['chosen_tiles_2'].append(ctc)

        if self.sched_algo == SchedAlgo.AdaptiveRR:
            self.processed_area_perc += len(ctc) / self.cur_netc_num_tiles

        return batch_dict

    # This method adds the latests detections to the queue
    def add_past_proj_to_queue(self, proj_all=False):
        batch_dict = self.latest_batch_dict
        if batch_dict is None or self.cur_pose is None:
            return batch_dict

        pred_dict = batch_dict['final_box_dicts'][0]

        # Before appending the dets, extract the projected ones
        proj_mask = pred_dict['pred_scores'] >= self.score_thresh
        for k in ('pred_boxes', 'pred_labels', 'pred_scores'):
            pred_dict[k] = pred_dict[k][proj_mask]

        new_dets_dict = {}
        score_inds = torch.argsort(pred_dict['pred_scores'])
        for k in ('pred_boxes', 'pred_labels'):
            new_dets_dict[k] = pred_dict[k][score_inds]

        # update num dets per tile
        W, W_start = self.pc_range[3] - self.pc_range[0], self.pc_range[0]
        div = W / self.tcount
        tile_inds = torch.div((new_dets_dict['pred_boxes'][:, 0] - W_start), div, \
                rounding_mode='trunc').short()
        tile_bins = torch.bincount(tile_inds, minlength=self.tcount)
        ctc = torch.from_numpy(batch_dict['chosen_tile_coords']).long()
        self.num_dets_per_tile[ctc] = tile_bins[ctc]

        # NOTE The cur_pose and cur_ts here actually belongs to previous sample
        self.past_poses = torch.cat((self.past_poses, self.cur_pose.unsqueeze(0)))
        self.past_ts = torch.cat((self.past_ts, self.target_ts if proj_all else self.cur_ts))
        # Append the pose idx for the detection that will be added
        num_dets = new_dets_dict['pred_boxes'].size(0)
        past_poi = self.past_detections['pose_idx']
        poi = torch.full((num_dets,), self.past_poses.size(0)-1, dtype=past_poi.dtype)
        self.past_detections['pose_idx'] = torch.cat((past_poi, poi))

        for k in ('pred_boxes', 'pred_labels'):
            self.past_detections[k] = torch.cat((self.past_detections[k], new_dets_dict[k]))

        return batch_dict

    # Projection
    def schedule3(self, batch_dict):
        # Add the detection results of previous sample
        proj_all = ('final_box_dicts' in batch_dict)
        self.add_past_proj_to_queue(proj_all)
        latest_token = batch_dict['metadata'][0]['token']
        scene_token = self.token_to_scene[latest_token]
        self.cur_pose = self.token_to_pose[latest_token]
        self.cur_ts = self.token_to_ts[latest_token]

        # Remove detections which are no more needed
        active_num_dets = torch.sum(self.num_dets_per_tile)
        max_num_proj = int(active_num_dets * self.projection_coeff)
        if self.past_detections['pred_boxes'].size(0) > max_num_proj:
            # Remove oldest dets
            for k in ['pose_idx', 'pred_boxes', 'pred_labels']:
                self.past_detections[k] = self.past_detections[k][-max_num_proj:]

        # Weed out using the pose_idx of first det
        if self.past_detections['pose_idx'].size(0) > 0:
            pose_idx_0 = self.past_detections['pose_idx'][0]
            self.past_poses = self.past_poses[pose_idx_0:]
            self.past_ts = self.past_ts[pose_idx_0:]
            self.past_detections['pose_idx'] = self.past_detections['pose_idx'] - pose_idx_0

        proj_dict = {}
        if proj_all:
            # NMS will happen right after projection
            # project the available det results as well for streaming eval
            det_pred_dict = batch_dict['final_box_dicts'][0]
            num_boxes = det_pred_dict['pred_boxes'].size(0)

            if self.past_detections['pred_boxes'].size(0) == 0:
                proj_dict['pred_scores'] = det_pred_dict['pred_scores'].cpu()
                proj_dict['pred_labels'] = det_pred_dict['pred_labels'].cpu() - 1
                boxes_to_forecast = det_pred_dict['pred_boxes']
                proj_dict['pose_idx'] = torch.tensor([0] * num_boxes,
                        dtype=self.past_detections['pose_idx'].dtype, device='cuda')
                proj_dict['past_poses'] = self.cur_pose.unsqueeze(0).cuda()
                proj_dict['past_ts'] = self.cur_ts.cuda()
            else:
                # MOVE TO CPU
                proj_pred_scores = self.score_thresh - \
                        (self.score_thresh / (self.past_detections['pose_idx'] + 2))
                proj_dict['pred_scores'] = torch.cat((proj_pred_scores,
                    det_pred_dict['pred_scores'].cpu()), dim=0)

                # MOVE TO CPU
                proj_pred_labels = (self.past_detections['pred_labels'] - 1)
                proj_dict['pred_labels'] = torch.cat((proj_pred_labels,
                    det_pred_dict['pred_labels'].cpu() - 1), dim=0)

                # MOVE TO GPU
                boxes_to_forecast = torch.cat((self.past_detections['pred_boxes'].cuda(),
                    det_pred_dict['pred_boxes']), dim=0)

                # MOVE TO GPU
                det_pose_idx = (self.past_detections['pose_idx'][-1] + 1).repeat(num_boxes)
                proj_dict['pose_idx'] = torch.cat((self.past_detections['pose_idx'],
                    det_pose_idx), dim=0).cuda()

                # MOVE TO GPU
                proj_dict['past_poses'] = torch.cat((self.past_poses,
                    self.cur_pose.unsqueeze(0)), dim=0).cuda()

                # MOVE TO GPU
                proj_dict['past_ts'] = torch.cat((self.past_ts, self.cur_ts), dim=0).cuda()

            # Calculate the exact current timestamp, for now, ignore the timing of
            # projection and nms
            torch.cuda.synchronize()
            elapsed_musec = (time.time() - batch_dict['start_time_sec']) * 1000000
            self.target_ts = self.cur_ts
            while self.target_ts < self.cur_ts + elapsed_musec:
                self.target_ts += 50000 # keep adding 50 ms

            proj_dict['pred_boxes']= cuda_projection.project_past_detections(
                    boxes_to_forecast,
                    proj_dict['pose_idx'],
                    proj_dict['past_poses'],
                    self.cur_pose.cuda(),
                    proj_dict['past_ts'],
                    self.target_ts.item()).cpu()

        else:
            # Do projection in the GPU
            if self.past_detections['pred_boxes'].size(0) > 0:
                proj_dict['pred_scores'] = self.score_thresh - \
                        (self.score_thresh / (self.past_detections['pose_idx'] + 2))
                proj_dict['pred_labels'] = (self.past_detections['pred_labels'] - 1)
                with torch.cuda.stream(self.projection_stream):
                    proj_dict['pred_boxes'] = cuda_projection.project_past_detections(
                            self.past_detections['pred_boxes'].cuda(),
                            self.past_detections['pose_idx'].cuda(),
                            self.past_poses.cuda(),
                            self.cur_pose.cuda(),
                            self.past_ts.cuda(),
                            self.cur_ts.item())
                    self.projection_stream.synchronize() # maybe not needed?
                    proj_dict['pred_boxes'] = proj_dict['pred_boxes'].cpu()

        if 'pred_boxes' in proj_dict:
            # clear up detections which fall under the chosen tiles and also the overtimed ones
            projb = proj_dict['pred_boxes']
            box_x, box_y = projb[:,0], projb[:,1]
            range_mask = box_x >= self.pc_range[0]
            range_mask = torch.logical_and(range_mask, box_x <= self.pc_range[3])
            range_mask = torch.logical_and(range_mask, box_y >= self.pc_range[1])
            range_mask = torch.logical_and(range_mask, box_y <= self.pc_range[4])

            # This op can make nms faster
            with torch.cuda.stream(self.projection_stream):

                target_key = 'projections_nms'
                batch_dict[target_key] = cuda_projection.split_projections(
                        proj_dict['pred_boxes'][range_mask],
                        proj_dict['pred_scores'][range_mask],
                        proj_dict['pred_labels'][range_mask],
                        self.cls_id_to_det_head_idx_map,
                        self.num_det_heads,
                        True) # moves results to gpu if true

            # hmm, might be necessary
            if proj_all:
                range_mask = range_mask[:-num_boxes]
            for k in ('pred_boxes', 'pred_labels', 'pose_idx'):
                self.past_detections[k] = self.past_detections[k][range_mask]

            self.projection_stream.synchronize()

        return batch_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing_pre(self, batch_dict):
        return (batch_dict,)

    def post_processing_post(self, pp_args):
        batch_dict = pp_args[0]
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes.cuda(),
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

    def sched_reset(self):
        self.processed_area_perc = 0.
        self.num_blacklisted_tiles = 0
        self.reset_ts = None

    def projection_reset(self):
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.cur_pose, self.cur_ts = None, None
        if self.sched_algo == SchedAlgo.ProjectionOnly:
            for k in ('pred_boxes', 'pred_labels', 'pose_idx'):
                self.past_detections[k] = []
            self.past_poses, self.past_ts = [], []
        else:
            self.past_detections = self.get_empty_det_dict()
            self.past_detections['pose_idx'] = torch.zeros([0], dtype=torch.long)
#                device=self.past_detections["pred_labels"].device)
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)
            self.num_dets_per_tile = torch.zeros([self.tcount], dtype=torch.long)
        self.last_tile_coord = self.init_tile_coord

    def calibrate(self, batch_size=1):
        self.calibrator = AnytimeCalibrator(self)

        self.collect_calib_data = False
        self.calib_fname = f"calib_data_m{self.model_cfg.METHOD}_c{self.tcount}.json"
        try:
            self.calibrator.read_calib_data(self.calib_fname)
        except OSError:
            self.collect_calib_data = True

        score_threshold = self.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH
        # this temporary threshold will allow us to do calibrate cudnn benchmarking
        # of all detection heads, preventing to skip any of them
        self.calibration_on()
        self.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH = 0.0001
        super().calibrate(1)
        self.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH = score_threshold

        self.enable_projection = (not self.keep_projection_disabled)
        self.projection_reset()
        self.sched_reset()
        if self.training:
            return None

        if self.collect_calib_data:
            self.calibrator.collect_data_v2(self.sched_algo, self.calib_fname)
            # After this, the calibration data should be processed with dynamic deadline
        self.clear_stats()
        self.clear_add_dict()
        self.calibration_off()

        return None

    def post_eval(self):
        if self.collect_calib_data:
            # We need to put bb3d time prediction data in the calibration file
            with open(self.calib_fname, 'r') as handle:
                calib_dict = json.load(handle)
                calib_dict['bb3d_preds'] = self.add_dict['bb3d_preds']
                calib_dict['exec_times'] = self.get_time_dict()
            with open(self.calib_fname, 'w') as handle:
                json.dump(calib_dict, handle, indent=4)

        self.add_dict['tcount'] = self.tcount
        self.add_dict['bb3d_pred_shift_ms'] = self.calibrator.expected_bb3d_err
        print(f"\nDeadlines missed: {self._eval_dict['deadlines_missed']}\n")

        self.plot_post_eval_data()

    def plot_post_eval_data(self):
        import matplotlib.pyplot as plt
        from datetime import datetime
        #from anytime_calibrator import get_stats

        root_path = '/root/shared_data/latest_exp_plots/'
        timedata = datetime.now().strftime("%m_%d_%H_%M")

        # plot 3d backbone time pred error
        time_dict = self.get_time_dict()
        if len(time_dict['Backbone3D']) == len(self.add_dict['bb3d_preds']):
            bb3d_pred_err = np.array(time_dict['Backbone3D']) - np.array(self.add_dict['bb3d_preds'])
            if 'VoxelHead-conv-hm' in time_dict:
                bb3d_pred_err += np.array(time_dict['VoxelHead-conv-hm'])

            # plot the data
            min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(bb3d_pred_err)
            hist, bin_edges = np.histogram(bb3d_pred_err, bins=100, density=True)
            cdf = np.cumsum(hist * np.diff(bin_edges))
            plt.plot(bin_edges[1:], cdf, linestyle='-')
            plt.grid(True)
            plt.xlim([perc1_, perc99_])
            plt.ylim([0, 1])
            plt.xlabel('Actual - Predicted bb3d execution time (msec)')
            plt.ylabel('CDF')
            plt.savefig(f'{root_path}/m{self.model_cfg.METHOD}_bb3d_time_pred_err_{timedata}.pdf')
            plt.clf()

        if not self.use_baseline_bb3d_predictor and not self.sched_disabled:
            # plot 3d backbone time pred error layerwise
            layer_times_actual = np.array(self.add_dict['bb3d_layer_times'])
            layer_times_pred = np.array(self.add_dict['bb3d_preds_layerwise'])
            layer_time_err = layer_times_actual - layer_times_pred
            for i in range(layer_time_err.shape[1]):
                #min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(bb3d_vpred_err[:, i])
                #perc1_min = min(perc1_min, perc1_)
                #perc99_max = max(perc99_max, perc99_)
                hist, bin_edges = np.histogram(layer_time_err[:, i], bins=100, density=True)
                cdf = np.cumsum(hist * np.diff(bin_edges))
                plt.plot(bin_edges[1:], cdf, linestyle='-', label=f"layer {i}")

            #plt.xlim([perc1_min, perc99_max])
            plt.grid(True)
            plt.legend()
            plt.xlabel('Actual - Predicted bb3d layer times')
            plt.ylabel('CDF')
            plt.savefig(f'{root_path}/m{self.model_cfg.METHOD}_bb3d_layer_time_err_{timedata}.pdf')
            plt.clf()


            # plot 3d backbone voxel pred error layerwise
            vactual = np.array(self.add_dict['bb3d_voxel_nums'])
            vpreds = np.array(self.add_dict['bb3d_voxel_preds'])
            bb3d_vpred_err = vactual - vpreds

            perc1_min, perc99_max = 50000, -50000
            for i in range(bb3d_vpred_err.shape[1]):
                min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(bb3d_vpred_err[:, i])
                perc1_min = min(perc1_min, perc1_)
                perc99_max = max(perc99_max, perc99_)
                hist, bin_edges = np.histogram(bb3d_vpred_err[:, i], bins=100, density=True)
                cdf = np.cumsum(hist * np.diff(bin_edges))
                plt.plot(bin_edges[1:], cdf, linestyle='-', label=f"layer {i}")

            #plt.xlim([perc1_min, perc99_max])
            plt.grid(True)
            plt.legend()
            plt.xlabel('Actual - Predicted bb3d num voxels')
            plt.ylabel('CDF')
            plt.savefig(f'{root_path}/m{self.model_cfg.METHOD}_bb3d_num_voxel_pred_err_{timedata}.pdf')
            plt.clf()
            print('Num voxel to exec time plot saved.')

            # plot 3d backbone fitted equations
            coeffs_calib, intercepts_calib = self.calibrator.time_reg_coeffs, \
                    self.calibrator.time_reg_intercepts
            coeffs_new, intercepts_new = self.calibrator.fit_voxel_time_data(vactual, \
                    layer_times_actual)
            calib_voxels, calib_times = self.calibrator.get_calib_data_arranged()
            fig, axes = plt.subplots(len(coeffs_calib)//2+1, 2, \
                    figsize=(6, 2*(len(coeffs_calib)//2+1)), 
                    sharex=True,
                    constrained_layout=True)
            axes = np.ravel(axes)
            for i in range(len(coeffs_calib)):
                #vlayer = vactual[:, i]
                vlayer = calib_voxels[:, i]
                xlims = [min(vlayer), max(vlayer)]
                x = np.arange(xlims[0], xlims[1], (xlims[1]-xlims[0])//100)

                num_voxels_ = np.expand_dims(x, -1) / self.calibrator.num_voxels_normalizer
                num_voxels_ = np.concatenate((num_voxels_, np.square(num_voxels_)), axis=-1)
                bb3d_time_calib = np.sum(num_voxels_ * coeffs_calib[i], axis=-1) + \
                        intercepts_calib[i]
                bb3d_time_new  = np.sum(num_voxels_ * coeffs_new[i], axis=-1) + \
                        intercepts_new[i]

                layer_times_ = calib_times[:, i]
                layer_voxels_ = calib_voxels[:, i]
                sort_indexes = np.argsort(layer_times_)
                layer_times_ = layer_times_[sort_indexes] #[0::5]
                layer_voxels_ = layer_voxels_[sort_indexes] #[0::5]
                #ax.scatter(layer_voxels_, layer_times_, label="calib")
                #ax.scatter(vlayer,layer_times_actual[:, i] , label="new")
                #ax.plot(x, bb3d_time_calib, label="calib")
                #ax.plot(x, bb3d_time_new, label="new")

                #ax.grid('True', ls='--')
                ax = axes[i]
                ax.scatter(layer_voxels_, layer_times_, label=f"Data")
                ax.plot(x, bb3d_time_calib, label="Model", color='orange')
                ax.set_ylim([0, max(layer_times_)*1.1])
                ax.set_title(f"Block {i+1}", fontsize='medium', loc='left')
                ax.legend(fontsize='medium')
                #ax.set_ylabel(f'Layer block {i}\nexecution\ntime (msec)', fontsize='x-large')
            fig.supxlabel('Number of input voxels', fontsize='x-large')
            fig.supylabel('Block execution time (msec)', fontsize='x-large')
            #ax.set_ylabel(f'Layer block {i}\nexecution\ntime (msec)', fontsize='x-large')
            #plt.subplots_adjust(wspace=0, hspace=0)

            plt.savefig(f'{root_path}/m{self.model_cfg.METHOD}_bb3d_fitted_data_all_{timedata}.pdf')
            #ax.set_xlim([0, 70000])

    def projection_for_test(self, batch_dict):
        pred_dicts = batch_dict['final_box_dicts']

        if self.enable_projection:
            # only keeps the previous detection
            projected_boxes=None
            pb = self.past_detections['pred_boxes']
            if len(pb) >= self.projLastNth and pb[-self.projLastNth].size(0) > 0:

                projected_boxes = cuda_projection.project_past_detections(
                        self.past_detections['pred_boxes'][-self.projLastNth],
                        self.past_detections['pose_idx'][-self.projLastNth],
                        self.past_poses[-self.projLastNth].cuda(),
                        self.cur_pose.cuda(),
                        self.past_ts[-self.projLastNth].cuda(),
                        self.cur_ts.item())

                projected_labels = self.past_detections['pred_labels'][-self.projLastNth]
                projected_scores = self.past_detections['pred_scores'][-self.projLastNth]

            ####USE DETECTION DATA#### START
#            # Second, append new detections
#            num_dets = pred_dicts[0]['pred_labels'].size(0)
#            self.past_detections['num_dets'] = num_dets
#            # Append the current pose
#            self.past_poses = self.cur_pose.unsqueeze(0)
#            self.past_ts = self.cur_ts #.unsqueeze(0)
#            # Append the pose idx for the detection that will be added
#            self.past_detections['pose_idx'] = \
#                    torch.full((num_dets,), 0, dtype=torch.long, device='cuda')
#
#            for k in ('pred_boxes', 'pred_scores', 'pred_labels'):
#                self.past_detections[k] = pred_dicts[0][k]
#
#            # append the projected detections
#            if projected_boxes is not None:
#                pred_dicts[0]['pred_boxes'] = projected_boxes
#                pred_dicts[0]['pred_scores'] = projected_scores
#                pred_dicts[0]['pred_labels'] = projected_labels
            ####USE DETECTION DATA#### END

            ####USE GROUND TRUTH#### START
            self.past_detections['pred_boxes'].append(batch_dict['gt_boxes'][0][..., :9])
            self.past_detections['pred_labels'].append(batch_dict['gt_boxes'][0][...,-1].int())
            self.past_detections['pred_scores'].append(torch.ones_like(\
                    self.past_detections['pred_labels'][-1]))

            num_dets = self.past_detections['pred_scores'][-1].size(0)
            self.past_poses.append(self.cur_pose.unsqueeze(0))
            self.past_ts.append(self.cur_ts)
            self.past_detections['pose_idx'].append( \
                    torch.zeros((num_dets,), dtype=torch.long)) #, device='cuda'))
            ####USE GROUND TRUTH#### END

            while len(self.past_poses) > self.projLastNth:
                for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'pose_idx'):
                    self.past_detections[k].pop(0)
                self.past_poses.pop(0)
                self.past_ts.pop(0)

            # append the projected detections
            if projected_boxes is not None:
                pred_dicts[0]['pred_boxes']  = projected_boxes
                pred_dicts[0]['pred_labels'] = projected_labels
                pred_dicts[0]['pred_scores'] = projected_scores
            else:
                # use groud truth if projection was not possible
                pred_dicts[0]['pred_boxes']  = self.past_detections['pred_boxes'][-1]
                pred_dicts[0]['pred_labels'] = self.past_detections['pred_labels'][-1]
                pred_dicts[0]['pred_scores'] = self.past_detections['pred_scores'][-1]

        return batch_dict


