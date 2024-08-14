from .detector3d_template import Detector3DTemplate
import torch
from nuscenes.nuscenes import NuScenes
import time
import sys
import json
import numpy as np
import scipy
import gc
import copy

from ..model_utils import model_nms_utils
from ...ops.cuda_projection import cuda_projection
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
from .. import load_data_to_gpu

#os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

class AnytimeTemplateV1(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if 'BACKBONE_2D' in self.model_cfg:
            self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
        if 'DENSE_HEAD' in self.model_cfg:
            self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
        self.module_list = self.build_networks()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
#        torch.use_deterministic_algorithms(True)

        ################################################################################
        self.tcount = self.model_cfg.TILE_COUNT
        self.tcount_cuda = torch.tensor(self.model_cfg.TILE_COUNT).long().cuda()
        self.total_num_tiles = self.tcount[0] * self.tcount[1]

        # This number will be determined by the scheduling algorithm initially for each input
        self.last_tile_coord = -1
        #self.reduce_mask_stream = torch.cuda.Stream()
        self.tile_size_voxels = torch.from_numpy(\
                self.dataset.grid_size[:2] / self.tcount).cuda().long()

        ####Projection###
        self.enable_projection = False
        self.token_to_scene = {}
        self.token_to_ts = {}
        with open('token_to_pos.json', 'r') as handle:
            self.token_to_pose = json.load(handle)

        for k, v in self.token_to_pose.items():
            cst, csr, ept, epr = v['cs_translation'],  v['cs_rotation'], \
                    v['ep_translation'], v['ep_rotation']
            # convert time stamps to seconds
            # 3 4 3 4
            self.token_to_pose[k] = torch.tensor((*cst, *csr, *ept, *epr), dtype=torch.float)
            self.token_to_ts[k] = torch.tensor((v['timestamp'],), dtype=torch.long)
            self.token_to_scene[k] = v['scene']
        ################################################################################

        self.calibrating_now = False
        self.add_dict = self._eval_dict['additional']
        for k in ('voxel_counts', 'num_tiles', 'PostSched'):
            self.add_dict[k] = []

        self.time_pred_coeffs_1 = torch.zeros(6, device='cuda')
        self.time_pred_coeffs_ALL = torch.zeros(6)
        self.pred_net_time_stats_1 = {'99perc': 0.0,}
        self.pred_net_time_stats_ALL = {'avrg': 0.0,}

        self.calib_num_tiles = -1
        self.skip_projection=False

        self.tile_ages = torch.ones(self.total_num_tiles, dtype=torch.long, device='cuda')
        self.RoundRobin = 1
        self.AgingWithDistance = 2
        self.ProjectionOnly, self.projLastNth = 3, 1

        self.sched_algo = self.model_cfg.METHOD

#        self.aft_prj_nms_conf = copy.deepcopy(\
#                self.model_cfg.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG)
#        self.aft_prj_nms_conf.NMS_PRE_MAXSIZE = 9999
#        self.aft_prj_nms_conf.NMS_POST_MAXSIZE = 300

        self.past_detections = {'num_dets': []}
        self.det_timeout_limit = int(0.6 * 1000000) # in microseconds
        self.prev_scene_token = ''
        if self.sched_algo == self.ProjectionOnly:
            self.past_poses = []
            self.past_ts = []
        else:
            # Poses include [cst(3) csr(4) ept(3) epr(4)]
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)

    def projection_init(self, batch_dict):
        if self.enable_projection:
            latest_token = batch_dict['metadata'][0]['token']
            scene_token = self.token_to_scene[latest_token]
            if scene_token != self.prev_scene_token:
                self.projection_reset()
                self.prev_scene_token = scene_token

            self.cur_pose = self.token_to_pose[latest_token]
            self.cur_ts = self.token_to_ts[latest_token]

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
                    torch.zeros((num_dets,), dtype=torch.long, device='cuda'))
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


    #TODO det_timeout_limit needs to be calibrated
    def projection(self, batch_dict):
        if self.sched_algo == self.ProjectionOnly:
            return self.projection_for_test(batch_dict)

        pred_dicts = batch_dict['final_box_dicts']
        if self.enable_projection:
            if self.skip_projection:
                self.projection_reset()

            # The limit is 500 objest per sample
            num_dets = pred_dicts[0]['pred_labels'].size(0)
            num_past_dets, dets_to_rm = self.past_detections['num_dets'], []
            while sum(num_past_dets) + num_dets > 500:
                dets_to_rm.append(num_past_dets.pop(0))

            if dets_to_rm:
                self.past_poses = self.past_poses[len(dets_to_rm):]
                self.past_ts = self.past_ts[len(dets_to_rm):]
                dets_to_rm_sum = sum(dets_to_rm)
                for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'pose_idx'):
                    self.past_detections[k] = self.past_detections[k][dets_to_rm_sum:]
                self.past_detections['pose_idx'] -= len(dets_to_rm)

            projected_boxes=None
            if self.past_detections['pred_boxes'].size(0) > 0 and not self.skip_projection:
                projected_boxes = cuda_projection.project_past_detections(
                        self.past_detections['pred_boxes'],
                        self.past_detections['pose_idx'],
                        self.past_poses.cuda(),
                        self.cur_pose.cuda(),
                        self.past_ts.cuda(),
                        self.cur_ts.item())

                projected_scores = self.past_detections['pred_scores']
                projected_labels = self.past_detections['pred_labels']

            # Second, append new detections
            self.past_detections['num_dets'].append(num_dets)
            # Append the current pose
            self.past_poses = torch.cat((self.past_poses, self.cur_pose.unsqueeze(0)))
            self.past_ts = torch.cat((self.past_ts, self.cur_ts))
            # Append the pose idx for the detection that will be added
            past_poi = self.past_detections['pose_idx']
            poi = torch.full((num_dets,), self.past_poses.size(0)-1,
                dtype=past_poi.dtype, device=past_poi.device)
            self.past_detections['pose_idx'] = torch.cat((past_poi, poi))
            for k in ('pred_boxes', 'pred_scores', 'pred_labels'):
                self.past_detections[k] = torch.cat((self.past_detections[k], pred_dicts[0][k]))

            # append the projected detections
            if projected_boxes is not None:
                pred_dicts[0]['pred_boxes'] = torch.cat((pred_dicts[0]['pred_boxes'],
                    projected_boxes))
                pred_dicts[0]['pred_scores'] = torch.cat((pred_dicts[0]['pred_scores'],
                    projected_scores))
                pred_dicts[0]['pred_labels'] = torch.cat((pred_dicts[0]['pred_labels'],
                    projected_labels))

                batch_dict['final_box_dicts'] = pred_dicts

#        if not self.skip_projection:
#            # Run the final NMS
#            selected, selected_scores = model_nms_utils.class_agnostic_nms(
#                box_scores=pred_dicts[0]['pred_scores'], box_preds=pred_dicts[0]['pred_boxes'],
#                nms_config=self.aft_prj_nms_conf, score_thresh=None
#            )
#
#            pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'][selected]
#            pred_dicts[0]['pred_scores'] = selected_scores
#            pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'][selected]
#
#            batch_dict['final_box_dicts'] = pred_dicts

        self.skip_projection=False

        return batch_dict

    def get_nonempty_tiles(self, voxel_coords):
        # Calculate where each voxel resides in which tile
        tile_coords = torch.div(voxel_coords[:, -2:], self.tile_size_voxels, \
                rounding_mode='trunc').long()

        voxel_tile_coords = tile_coords[:, 1] * self.tcount[1] + tile_coords[:, 0]
        nonempty_tile_coords, voxel_counts = torch.unique(voxel_tile_coords, \
                sorted=True, return_counts=True)
        return voxel_tile_coords, nonempty_tile_coords, voxel_counts

    def schedule(self, batch_dict):
        self.measure_time_start('Sched')
        voxel_coords = batch_dict['voxel_coords']
        voxel_tile_coords, netc, netc_vcounts= self.get_nonempty_tiles(voxel_coords)
        batch_dict['mask'] = None
        if self.training:
            batch_dict['chosen_tile_coords'] = netc
            return
        torch.cuda.synchronize()
        self.psched_start_time = time.time()
        rem_time = batch_dict['abs_deadline_sec'] - self.psched_start_time

        num_nonempty_tiles = netc.size(0)
        x1, y1 = batch_dict['voxel_coords'].size(0), num_nonempty_tiles
        XY = torch.tensor((1., x1, y1, x1*y1, x1**2, y1**2))
        tpred_all = torch.dot(XY, self.time_pred_coeffs_ALL)
        rt = rem_time - self.pred_net_time_stats_ALL['avrg'] # aggresive

        calibrating_all_tiles = (self.calibrating_now and \
                self.calib_num_tiles == num_nonempty_tiles)
        if calibrating_all_tiles or (not self.calibrating_now and tpred_all < rt):
            # If we can simply process all tiles, no need for scheduling
            idx = num_nonempty_tiles
            chosen_tile_coords = netc
            self.last_tile_coord = -1
            self.skip_projection=True
        elif self.sched_algo == self.RoundRobin:
            # Here I need to run the scheduling algorithm
            # If it is not going to process all tiles, then process at most
            # half of the tiles

            tile_begin_idx = \
                    (netc > self.last_tile_coord).type(torch.uint8).argmax()
#            tl_end = tile_begin_idx + num_nonempty_tiles
#            ntc = netc.expand(2, num_nonempty_tiles).flatten()
#            netc = ntc[tile_begin_idx:tl_end].contiguous()
            tl_end = min(tile_begin_idx + num_nonempty_tiles//2+1, num_nonempty_tiles)
            netc = netc[tile_begin_idx:tl_end]

#            num_tiles = torch.arange(1, netc.size(0)+1, device=netc_vcounts.device).float()
            num_tiles = torch.arange(1, tl_end-tile_begin_idx+1, \
                    device=netc_vcounts.device).float()
#            cnts = netc_vcounts.expand(2, netc_vcounts.size(0)).flatten()
#            cnts = cnts[tile_begin_idx:tl_end].contiguous()
            cnts = netc_vcounts[tile_begin_idx:tl_end]
            cnts_cumsum = torch.cumsum(cnts, dim=0).float()

        elif self.sched_algo == self.AgingWithDistance:

            netc_y = torch.div(netc, self.tcount[1], rounding_mode='trunc') - (self.tcount[1]//2)
            netc_x = (netc % self.tcount[1]) - (self.tcount[0]//2)
            s = torch.pow(netc_y, 2) + torch.pow(netc_x, 2)
            s = s.float() / ((self.tcount[0]//2)**2 + (self.tcount[1]//2)**2) # normalize
            dists = 1.0 - s # prioritize closer tiles

            ages = self.tile_ages[netc]
            prios = ages * dists # NOTE, this is an ad-hoc formula

            inds = torch.argsort(prios, descending=True)
            netc = netc[inds]
            netc_vcounts = netc_vcounts[inds]

            num_tiles = torch.arange(1, netc.size(0)+1, device=netc_vcounts.device).float()
            cnts_cumsum = torch.cumsum(netc_vcounts, dim=0).float()

#<<<<<<< Updated upstream
        elif self.sched_algo == self.ProjectionOnly:
            batch_dict['chosen_tile_coords'] = netc
            self.measure_time_end('Sched')
            return batch_dict

#        # Get execution time predictions
#        if self.time_pred_coeffs:
#            coeffs = self.time_pred_coeffs
#            pred_time_stats = self.pred_net_time_stats
#        else:
#            # do it 2 for worst case
#            coeffs = [torch.zeros(6, device='cuda')] * 2
#            pred_time_stats = [{'95perc':0.0, '99perc':0.0, 'max': 0.0}] * 2
#
#        tpreds = []
#        for C in coeffs:
#            #tpreds = C[0]*cnts_cumsum + C[1]*num_tiles + C[2]
#=======
        if not self.skip_projection:
            # We will do partial execution, get execution time predictions
#            tpreds = []
            C = self.time_pred_coeffs_1
#>>>>>>> Stashed changes
            x1, y1 = cnts_cumsum, num_tiles
            XY = torch.stack((torch.ones(x1.size(0), device='cuda'), x1, y1, \
                    x1*y1, x1**2, y1**2), dim=1)
            tpreds = torch.matmul(XY, C)
            rem_time -= self.pred_net_time_stats_1['99perc']

            diffs = (tpreds < rem_time).cpu()
            idx = torch.sum(diffs).item()
            if idx == 0: # Try to meet the deadline as fast as possible
                idx = int(num_tiles[0])

            if self. calibrating_now:
                if self.sched_algo == self.RoundRobin:
                    idx = int(min(num_tiles[-1], self.calib_num_tiles))
                else:
                    idx = self.calib_num_tiles

            # Voxel filtering is needed
            chosen_tile_coords = netc[:idx]
            self.last_tile_coord = chosen_tile_coords[-1]
            tile_filter = cuda_point_tile_mask.point_tile_mask(voxel_tile_coords, \
                    chosen_tile_coords)
            batch_dict['mask'] = tile_filter
            if 'voxel_features' in batch_dict:
                batch_dict['voxel_features'] = \
                        batch_dict['voxel_features'][tile_filter].contiguous()
            batch_dict['voxel_coords'] = voxel_coords[tile_filter].contiguous()

            if self.sched_algo == self.AgingWithDistance:
                self.tile_ages += 1

        if self.sched_algo == self.AgingWithDistance:
            self.tile_ages[chosen_tile_coords] = 1

        batch_dict['chosen_tile_coords'] = chosen_tile_coords
        self.add_dict['voxel_counts'].append(\
                batch_dict['voxel_coords'].size(0))
        self.add_dict['num_tiles'].append(batch_dict['chosen_tile_coords'].size(0))


        self.measure_time_end('Sched')

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
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

    def projection_reset(self):
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        if self.sched_algo == self.ProjectionOnly:
            for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'pose_idx', 'num_dets'):
                self.past_detections[k] = []
            self.past_poses, self.past_ts = [], []
        else:
            self.past_detections = self.get_empty_det_dict()
            self.past_detections['num_dets'] = []
            self.past_detections['pose_idx'] = torch.zeros([0], dtype=torch.long,
                device=self.past_detections["pred_labels"].device)
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)
        self.last_tile_coord = -1
        self.tile_ages = torch.ones(self.total_num_tiles, dtype=torch.long, device='cuda')

    def calc_time_pred_coeffs(self, num_voxels, num_tiles, psched_time):
        X = np.array(num_voxels, dtype=np.float32)
        Y = np.array(num_tiles, dtype=np.float32)
        Z = np.array(psched_time, dtype=np.float32)
        x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()

        # linear
        #A = np.c_[x1, y1, np.ones(x1.shape[0])]
        #C,_,_,_ = scipy.linalg.lstsq(A, z1)    # coefficients
        #plane_z = C[0]*x1 + C[1]*y1 + C[2]

        # quadratic
        xy = np.stack([x1, y1], axis=1)
        A = np.c_[np.ones(x1.shape[0]), xy, np.prod(xy, axis=1), xy**2]
        C,_,_,_ = scipy.linalg.lstsq(A, z1)
        plane_z = np.dot(np.c_[np.ones(x1.shape), x1, y1, x1*y1, x1**2, y1**2], C)

        time_pred_coeffs = torch.from_numpy(C).float().cuda()
        print('time_pred_coeffs', time_pred_coeffs)

        diff = z1 - plane_z
        perc95 = np.percentile(diff, 95, method='lower')
        perc99 = np.percentile(diff, 99, method='lower')
        pred_net_time_stats = {
            'min': float(min(diff)),
            'avrg': float(sum(diff)/len(diff)),
            '95perc': float(perc95),
            '99perc': float(perc99),
            'max': float(max(diff))}
        print('time_pred_stats', pred_net_time_stats)

        return time_pred_coeffs, pred_net_time_stats

    def calibrate(self, fname='calib_raw_data.json'):
        super().calibrate(1)
        self.enable_projection = True
        self.projection_reset()

        for l in self.add_dict.values():
            l.clear()

        if self.sched_algo == self.ProjectionOnly:
            print('Projection test is running.')
            return None

        # check if the wcet pred file is there
        try:
            with open(fname, 'r') as handle:
                calib_dict = json.load(handle)

            num_voxels = calib_dict["voxel_counts"]

            if 'num_tiles' in calib_dict and calib_dict['num_tiles']:
                num_tiles = calib_dict['num_tiles']
            else:
                tile_coords = calib_dict["chosen_tile_coords"]
                num_tiles = [len(tc) for tc in tile_coords]

            num_ALL_samples = calib_dict['calib_dataset_len']
            num_voxels_ALL = num_voxels[-num_ALL_samples:]
            num_tiles_ALL = num_tiles[-num_ALL_samples:]

            num_voxels = num_voxels[:-num_ALL_samples]
            num_tiles = num_tiles[:-num_ALL_samples]

            psched_time = calib_dict["PostSched"]
            psched_time_ALL = psched_time[-num_ALL_samples:]
            psched_time = psched_time[:-num_ALL_samples]

            self.time_pred_coeffs_1, self.pred_net_time_stats_1 = \
                    self.calc_time_pred_coeffs(num_voxels, num_tiles, psched_time)
            self.time_pred_coeffs_ALL, self.pred_net_time_stats_ALL = \
                    self.calc_time_pred_coeffs(num_voxels_ALL, num_tiles_ALL, psched_time_ALL)
            self.time_pred_coeffs_ALL = self.time_pred_coeffs_ALL.cpu()

        except FileNotFoundError:
            print(f'Calibration file {fname} not found, running calibration')
            self.calibrating_now = True # time calibration!
            self.calibration_procedure(fname)
            sys.exit()

        return None

    def calibration_procedure(self, fname="calib_raw_data.json"):
        gc.disable()
        all_max_num_tiles = []

        pc_range = torch.from_numpy(self.dataset.point_cloud_range).cuda()[[0, 1]]
        voxel_size = torch.tensor(self.dataset.voxel_size).cuda()[[0, 1]]
        grid_size = torch.from_numpy(self.dataset.grid_size).cuda()[[0, 1]]
        for i in range(len(self.dataset)):
            data_dict = self.dataset.getitem_pre(i)
            data_dict = self.dataset.getitem_post(data_dict)
            data_dict = self.dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
    
            if 'voxel_coords' not in data_dict:
                points = data_dict['points']
                #print(pc_range, voxel_size, grid_size)
                #print(points[-10:])
                points_coords = torch.floor( \
                    (points[:, [1, 2]] - pc_range) / voxel_size).int()
                mask = ((points_coords >= 0) & (points_coords < grid_size)).all(dim=1)
                #print(points_coords[-10:])
                data_dict['voxel_coords'] = points_coords[mask][:, [1,0]]

            _, nonempty_tile_coords, _ = self.get_nonempty_tiles(data_dict['voxel_coords'])
            max_num_tiles = nonempty_tile_coords.size(0)
            all_max_num_tiles.append(max_num_tiles)

        mit, mat = min(all_max_num_tiles), max(all_max_num_tiles)
        print(f'Min num tiles: {mit}, Max num tiles: {mat}')
        torch.cuda.empty_cache()
        gc.collect()

        # 10 different num of tiles should be enough
        for num_tiles in range(1, mat, mat//10):
            print('Num tiles:', num_tiles)
            for i in range(len(self.dataset)):
                if num_tiles < all_max_num_tiles[i]:
                    self.calib_num_tiles = num_tiles
                    with torch.no_grad():
                        pred_dicts, ret_dict = self([i])
                    gc.collect()

        print('Num tiles: ALL')
        for i in range(len(self.dataset)):
            self.calib_num_tiles = all_max_num_tiles[i]
            with torch.no_grad():
                pred_dicts, ret_dict = self([i])
            gc.collect()

        gc.enable()
        self.add_dict['tcount'] = self.tcount
        self.add_dict['method'] = self.sched_algo
        self.add_dict['exec_times'] = self.get_time_dict()
        self.add_dict['exec_time_stats'] = self.get_time_dict_stats()
        self.add_dict['calib_dataset_len'] = len(self.dataset)
        print('Time calibration Complete')
        with open(fname, 'w') as handle:
            json.dump(self.add_dict, handle, indent=4)

    def post_eval(self):
        self.add_dict['tcount'] = self.tcount
        print(f"\nDeadlines missed: {self._eval_dict['deadlines_missed']}\n")
