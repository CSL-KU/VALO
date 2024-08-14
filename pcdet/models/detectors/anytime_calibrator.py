import torch
import time
import json
import numpy as np
import numba
import gc
import os
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression

if __name__ != "__main__":
    from .sched_helpers import SchedAlgo, get_num_tiles
    from ...ops.cuda_point_tile_mask import cuda_point_tile_mask

def calc_grid_size(pc_range, voxel_size):
    return np.array([ int((pc_range[i+3]-pc_range[i]) / vs)
            for i, vs in enumerate(voxel_size)])

@numba.njit()
def tile_coords_to_id(tile_coords):
    tid = 0
    for tc in tile_coords:
        tid += 2 ** tc
    return int(tid)

def get_stats(np_arr):
    min_, max_, mean_ = np.min(np_arr), np.max(np_arr), np.mean(np_arr)
    perc1_ = np.percentile(np_arr, 1, method='lower')
    perc5_ = np.percentile(np_arr, 5, method='lower')
    perc95_ = np.percentile(np_arr, 95, method='lower')
    perc99_ = np.percentile(np_arr, 99, method='lower')
    print("Min\t1Perc\t5Perc\tMean\t95Perc\t99Perc\tMax")
    print(f'{min_:.2f}\t{perc1_:.2f}\t{perc5_:.2f}\t{mean_:.2f}\t{perc95_:.2f}\t{perc99_:.2f}\t{max_:.2f}')
    return (min_, mean_, perc1_, perc5_, perc95_, perc99_, max_)


class AnytimeCalibrator():
    def __init__(self, model):
        self.model = model
        self.calib_data_dict = None
        if model is None:
            self.dataset = None
            #NOTE modify the following params depending on the config file
            self.num_det_heads = 8
            self.num_tiles = 18
        else:
            self.dataset = model.dataset
            self.num_det_heads = len(model.dense_head.class_names_each_head)
            self.num_tiles = model.model_cfg.TILE_COUNT

        self.time_reg_degree = 2
        self.bb3d_num_l_groups  =self.model.backbone_3d.num_layer_groups
        if self.model.use_voxelnext:
            # count the convolutions of the detection head to be
            # a part of 3D backbone
            self.bb3d_num_l_groups += 1 # detection head convolutions

        self.use_baseline_bb3d_predictor = self.model.use_baseline_bb3d_predictor
        self.move_indscalc_to_init = self.model.move_indscalc_to_init
        if self.use_baseline_bb3d_predictor:
            self.time_reg_coeffs = np.ones((self.time_reg_degree,), dtype=float)
            self.time_reg_intercepts = np.ones((1,), dtype=float)
        else:
            self.time_reg_coeffs = np.ones((self.bb3d_num_l_groups, self.time_reg_degree), dtype=float)
            self.time_reg_intercepts = np.ones((self.bb3d_num_l_groups,), dtype=float)

            self.scale_num_voxels = False # False appears to be better!
            self.voxel_coeffs_over_layers = np.array([[1.] * self.num_tiles \
                    for _ in range(self.bb3d_num_l_groups)])

        # backbone2d and detection head heatmap convolutions
        # first elem unused
        self.det_head_post_wcet_ms = .0
        if not self.model.use_voxelnext:
            self.bb2d_times_ms = np.zeros((self.num_tiles+1,), dtype=float)

        self.expected_bb3d_err = 0.
        self.num_voxels_normalizer = 100000.
        self.chosen_tiles_calib = 18


    # voxel dists should be [self.bb3d_num_l_groups, num_tiles]
    def commit_bb3d_updates(self, ctc, voxel_dists):
        voxel_dists = voxel_dists[:, ctc]
        if self.scale_num_voxels:
            self.voxel_coeffs_over_layers[:, ctc] = voxel_dists / voxel_dists[0]
        else:
            self.voxel_coeffs_over_layers[:, ctc] = voxel_dists

    # overhead on jetson-agx: 1 ms
    def pred_req_times_ms(self, vcount_area, tiles_queue, num_tiles): # [num_nonempty_tiles, num_max_tiles]
        if self.use_baseline_bb3d_predictor:
            assert self.time_reg_degree == 2

            vcounts = vcount_area.flatten()
            num_voxels = np.empty((tiles_queue.shape[0]),dtype=float)
            for i in range(len(tiles_queue)):
                num_voxels[i] = np.sum(vcounts[tiles_queue[:i+1]])

            num_voxels_n_ = np.expand_dims(num_voxels, -1) / self.num_voxels_normalizer
            num_voxels_n_ = np.concatenate((num_voxels_n_, np.square(num_voxels_n_)), axis=-1)
            bb3d_time_preds = np.sum(num_voxels_n_ * self.time_reg_coeffs.flatten(), \
                    axis=-1) +  self.time_reg_intercepts + self.expected_bb3d_err
        else:
            if self.scale_num_voxels:
                vcounts = vcount_area * self.voxel_coeffs_over_layers 
            else:
                vcounts = self.voxel_coeffs_over_layers
                vcounts[0] = vcount_area.flatten()
            num_voxels = np.empty((tiles_queue.shape[0], vcounts.shape[0]), dtype=float)
            for i in range(len(tiles_queue)):
                num_voxels[i] = np.sum(vcounts[:, tiles_queue[:i+1]], axis=1)
            if self.time_reg_degree == 1:
                bb3d_time_preds = num_voxels / self.num_voxels_normalizer * \
                        self.time_reg_coeffs.flatten() + \
                        self.time_reg_intercepts
            elif self.time_reg_degree == 2:
                num_voxels_n_ = np.expand_dims(num_voxels, -1) / self.num_voxels_normalizer
                num_voxels_n_ = np.concatenate((num_voxels_n_, np.square(num_voxels_n_)), axis=-1)
                bb3d_time_preds = np.sum(num_voxels_n_ * self.time_reg_coeffs, axis=-1) + \
                        self.time_reg_intercepts
                # need to divide this cuz adding it to each layer individually
                bb3d_time_preds[:, 0] += self.expected_bb3d_err

        if self.model.use_voxelnext:
            return bb3d_time_preds, self.det_head_post_wcet_ms, num_voxels
        else:
            return bb3d_time_preds, self.bb2d_times_ms[num_tiles] + \
                    self.det_head_post_wcet_ms, num_voxels

    def pred_final_req_time_ms(self, dethead_indexes):
        return self.det_head_post_wcet_ms

    def fit_voxel_time_data(self, voxel_data, times_data):
        coeffs, intercepts = [], []
        for i in range(self.bb3d_num_l_groups): # should be 4, num bb3d conv blocks
            voxels = voxel_data[:, i:i+1] / self.num_voxels_normalizer
            times = times_data[:, i:i+1]

            if self.time_reg_degree == 2:
                voxels = np.concatenate((voxels, np.square(voxels)), axis=-1)
            reg = LinearRegression().fit(voxels, times)

            coeffs.append(reg.coef_.flatten())
            intercepts.append(reg.intercept_[0])
        return np.array(coeffs), np.array(intercepts)

    def get_calib_data_arranged(self):
        if self.calib_data_dict['version'] == 1:
            bb3d_voxels_samples = self.calib_data_dict['bb3d_voxels']
            exec_times_ms_samples = self.calib_data_dict['bb3d_time_ms']

            all_times, all_voxels = [], []
            for bb3d_voxels_s, exec_times_ms_s in zip(bb3d_voxels_samples, exec_times_ms_samples):
                all_voxels.extend(bb3d_voxels_s)
                all_times.extend(exec_times_ms_s)
        elif self.calib_data_dict['version'] == 2:
            all_voxels = self.calib_data_dict['bb3d_voxels']
            all_times = self.calib_data_dict['bb3d_time_ms']

        all_times=np.array(all_times, dtype=float)
        all_voxels=np.array(all_voxels, dtype=float)
        return all_voxels, all_times

    def read_calib_data(self, fname='calib_data.json'):
        f = open(fname)
        self.calib_data_dict = json.load(f)
        f.close()

        # Fit the linear model for bb3
        all_voxels, all_times = self.get_calib_data_arranged()
        # As a baseline predictor, do linear regression using
        # the number of voxels

        # plot voxel to time graph
        bb3d_times  = np.sum(all_times, axis=-1, keepdims=True)
        bb3d_voxels = all_voxels[:, :1]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
        #ax.grid(True)
        ax.scatter(bb3d_voxels, bb3d_times) #, label='data')
        ax.set_xlim([0, 70000])
        ax.set_ylim([0, 150])
        ax.set_xlabel('Number of input voxels', fontsize='x-large')
        ax.set_ylabel('3D backbone\nexecution time (msec)', fontsize='x-large')

#            bb3d_voxels_n = bb3d_voxels / self.num_voxels_normalizer
#            if self.time_reg_degree == 2:
#                bb3d_voxels_n = np.concatenate((bb3d_voxels_n,
#                    np.square(bb3d_voxels_n)), axis=-1)
#            reg = LinearRegression().fit(bb3d_voxels_n, bb3d_times)
#
#            self.time_reg_coeffs = reg.coef_
#            self.time_reg_intercepts = reg.intercept_
#
#            pred_times = np.sum(bb3d_voxels_n * self.time_reg_coeffs.flatten(), \
#                    axis=-1) +  self.time_reg_intercepts
#            plt.scatter(bb3d_voxels.flatten(), pred_times.flatten(), label='pred')
#            plt.legend()
        plt.savefig(f'/root/shared_data/latest_exp_plots/voxels_to_bb3dtime.pdf')
        plt.clf()
        if not self.use_baseline_bb3d_predictor:
            self.time_reg_coeffs, self.time_reg_intercepts = self.fit_voxel_time_data(all_voxels, all_times)

            # the input is voxels: [NUM_CHOSEN_TILES, self.bb3d_num_l_groups],
            # the output is times: [NUM_CHOSEN_TILEs, self.bb3d_num_l_groups]
            all_voxels_n = np.expand_dims(all_voxels, -1) / self.num_voxels_normalizer
            all_voxels_n = np.concatenate((all_voxels_n, np.square(all_voxels_n)), axis=-1)
            all_preds = np.sum(all_voxels_n * self.time_reg_coeffs, axis=-1)
            all_preds += self.time_reg_intercepts
            diffs = all_times - all_preds
            print('Excepted time prediction error for each 3D backbone layer\n' \
                    ' assuming the number of voxels are predicted perfectly:')
            for i in range(self.bb3d_num_l_groups):
                get_stats(diffs[:,i])

        dh_post_time_data = self.calib_data_dict['det_head_post_time_ms']
        self.det_head_post_wcet_ms = np.percentile(dh_post_time_data, \
                99, method='lower')
        print('det_head_post_wcet_ms', self.det_head_post_wcet_ms)

        if not self.model.use_voxelnext:
            bb2d_time_data = self.calib_data_dict['bb2d_time_ms']
            self.bb2d_times_ms = np.array([np.percentile(arr if arr else [0], 99, method='lower') \
                    for arr in bb2d_time_data])
            print('bb2d_times_ms')
            print(self.bb2d_times_ms)

        if 'exec_times' in self.calib_data_dict:
            # calculate the 3dbb err cdf
            time_dict = self.calib_data_dict['exec_times']
            bb3d_pred_err = np.array(time_dict['Backbone3D']) - \
                    np.array(self.calib_data_dict['bb3d_preds'])
            if 'VoxelHead-conv-hm' in time_dict:
                bb3d_pred_err += np.array(time_dict['VoxelHead-conv-hm'])

            print('Overall 3D Backbone time prediction error stats:')
            min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(bb3d_pred_err)
            self.expected_bb3d_err = int(os.getenv('PRED_ERR_MS', 0))
            print('Expected bb3d error ms:', self.expected_bb3d_err)

    def get_chosen_tile_num(self):
        return self.chosen_tiles_calib

    def collect_data_v2(self, sched_algo, fname="calib_data.json"):
        print('Calibration starting...')
        print('NUM_POINT_FEATURES:', self.model.vfe.num_point_features)
        print('POINT_CLOUD_RANGE:', self.model.vfe.point_cloud_range)
        print('VOXEL_SIZE:', self.model.vfe.voxel_size)
        print('GRID SIZE:', self.model.vfe.grid_size)

        num_samples = len(self.dataset)
        print('Number of samples:', num_samples)

        if not self.model.use_voxelnext:
            bb2d_time_data =  [list() for _ in range(self.bb2d_times_ms.shape[0])]
        dh_post_time_data = []

        gc.disable()
        sample_idx, tile_num = 0, 1
        while sample_idx < num_samples:
            time_begin = time.time()
            print(f'Processing sample {sample_idx}-{sample_idx+10}', end='', flush=True)

            # Enforce a number of tile
            for i in range(10):
                if sample_idx < num_samples:
                    self.chosen_tiles_calib = self.num_tiles if i == 0 else tile_num
                    self.model([sample_idx])
                    lbd = self.model.latest_batch_dict
                    if not self.model.use_voxelnext:
                        e1, e2 = lbd['bb2d_time_events']
                        bb2d_time = e1.elapsed_time(e2)
                        nt = get_num_tiles(lbd['chosen_tile_coords'])
                        bb2d_time_data[nt].append(bb2d_time)
                    e1, e2 = lbd['detheadpost_time_events']
                    dh_post_time_data.append(e1.elapsed_time(e2))
                    sample_idx += 1
                    gc.collect()
            tile_num = (tile_num % self.num_tiles) + 1

            time_end = time.time()
            #print(torch.cuda.memory_allocated() // 1024**2, "MB is being used by tensors.")
            print(f' took {round(time_end-time_begin, 2)} seconds.')
        gc.enable()

        self.calib_data_dict = {
                "version": 2,
                "bb3d_time_ms": self.model.add_dict['bb3d_layer_times'][1:],
                "bb3d_voxels": self.model.add_dict['bb3d_voxel_nums'][1:],
                "chosen_tile_coords": self.model.add_dict['chosen_tiles_1'][1:],
                "det_head_post_time_ms": dh_post_time_data,
        }

        if not self.model.use_voxelnext:
            self.calib_data_dict["bb2d_time_ms"] = bb2d_time_data

        with open(fname, "w") as outfile:
            json.dump(self.calib_data_dict, outfile, indent=4)

        # Read and parse calib data after dumping
        self.read_calib_data(fname)

