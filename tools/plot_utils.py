import os, re
import glob
import sys
import copy
import json
import math
import gc
import threading
import concurrent.futures
from multiprocessing import Process
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

proto_AP_types_dict = None
proto_AP_dict = None
proto_mAP_dict = None
proto_eval_dict = None

def init_dicts(dataset):
    global proto_AP_types_dict
    global proto_AP_dict
    global proto_mAP_dict
    global proto_eval_dict

    if dataset == 'KittiDataset':
        proto_AP_types_dict = {
            "aos": [],
            "3d": [],
            "bev": [],
            "image": [],
        }

        # Rows will be aos image bev 3d, cols will be easy medium hard
        proto_AP_dict = {
            'Car': copy.deepcopy(proto_AP_types_dict),
            'Pedestrian': copy.deepcopy(proto_AP_types_dict),
            'Cyclist': copy.deepcopy(proto_AP_types_dict),
        }

        proto_mAP_dict = {
            "aos":0.0,
            '3d': 0.0,
            'bev': 0.0,
            'image': 0.0,
        }

    elif dataset == 'NuScenesDataset':
        proto_AP_types_dict = {
            "AP": [], # 0.5, 1.0, 2.0, 4.0
        }

        proto_AP_dict = {
            'car': copy.deepcopy(proto_AP_types_dict),
            'pedestrian': copy.deepcopy(proto_AP_types_dict),
            'traffic_cone': copy.deepcopy(proto_AP_types_dict),
            'motorcycle': copy.deepcopy(proto_AP_types_dict),
            'bicycle': copy.deepcopy(proto_AP_types_dict),
            'bus': copy.deepcopy(proto_AP_types_dict),
            'trailer': copy.deepcopy(proto_AP_types_dict),
            'truck': copy.deepcopy(proto_AP_types_dict),
            'construction_vehicle': copy.deepcopy(proto_AP_types_dict),
            'barrier': copy.deepcopy(proto_AP_types_dict),
        }

        proto_mAP_dict = {
            'NDS': 0.0,
            'mAP': 0.0,
        }

    proto_exec_time_dict = {
        'End-to-end': [],
    }

    proto_eval_dict = {
        'method': 1,  # VAL
        'rpn_stg_exec_seqs': [],
        'gt_counts': [],  # 2D_LIST
        'deadline_sec': 0.1,  # VAL
        'deadline_msec': 100.,  # VAL
        'deadlines_missed': 0,  # VAL
        'deadline_diffs': [],  # 1D_LIST
        'exec_times': proto_exec_time_dict,  # DICT
        'exec_time_stats': proto_exec_time_dict,  # DICT
        "AP": proto_AP_dict,
        "mAP": proto_mAP_dict,
        'eval_results_dict': {},
        "dataset": 'NuScenesDataset',
        "time_err": {},
        'avrg_recognize_time': 0.,
        'avrg_instances_detected': 0.,
        'color': 'r',
        'lnstyle': '-',
    }

def merge_eval_dicts(eval_dicts):
    merged_ed = copy.deepcopy(proto_eval_dict)

    proto_keys_set = set(proto_eval_dict.keys())
    for ed in eval_dicts:
        missing_keys_set = set(ed.keys())
        diff_keys = proto_keys_set.difference(missing_keys_set)
        for k in diff_keys:
            ed[k] = proto_eval_dict[k]
            
    # use np.concatenate on a 2D list if you want to merge multiple 2D arrays
    for k, v in merged_ed.items():
        if not isinstance(v, dict):
            merged_ed[k] = [e[k] for e in eval_dicts]

    for k1 in ['exec_times', 'exec_time_stats',  'mAP']:
        for k2 in proto_eval_dict[k1].keys():
            merged_ed[k1][k2] = [e[k1][k2] \
                    for e in eval_dicts]
#                    for e in eval_dicts if e[k1].__contains__(k2)]

    for cls in merged_ed['AP'].keys():
        for eval_type in proto_AP_dict[cls].keys():
            merged_ed['AP'][cls][eval_type] = \
                [e['AP'][cls][eval_type] for e in eval_dicts]

    return merged_ed

# each experiment has multiple eval dicts
def load_eval_dict(path):
    print('Loading', path)
    with open(path, 'r') as handle:
        eval_d = json.load(handle)
    eval_d['deadline_msec'] = int(eval_d['deadline_sec'] * 1000)

    dataset = eval_d.get('dataset','NuScenesDataset')
    # Copy AP dict with removing threshold info, like @0.70
    AP_dict_json = eval_d["eval_results_dict"]
    AP_dict = copy.deepcopy(proto_AP_dict)
    if dataset == 'KittiDataset':
        for cls_metric, AP in AP_dict_json.items():
            cls_metric, difficulty = cls_metric.split('/')
            if cls_metric == 'recall':
                continue
            cls, metric = cls_metric.split('_')
            AP_dict[cls][metric].append(AP)
        for v in AP_dict.values():
            for v2 in v.values():
                v2.sort() # sort according to difficulty

        eval_d["AP"] = AP_dict

        # Calculate mAP values
        eval_d["mAP"] = copy.deepcopy(proto_mAP_dict)
        for metric in eval_d["mAP"].keys():
            mAP, cnt = 0.0, 0
            for v in eval_d["AP"].values():
                mAP += sum(v[metric])  # hard medium easy
                cnt += len(v[metric])  # 3
            if cnt > 0:
                eval_d["mAP"][metric] = mAP / cnt
    elif dataset == 'NuScenesDataset':
        results = AP_dict_json['result_str'].split('\n')
        for i, r in enumerate(results):
            for cls in proto_AP_dict.keys():
                if cls in r:
                    AP_scores = results[i+1].split('|')[1].split(',')
                    AP_scores = [float(a.strip()) for a in AP_scores]
                    AP_dict[cls]['AP'] = AP_scores

        eval_d["AP"] = AP_dict
        # Get mAP values
        eval_d["mAP"] = copy.deepcopy(proto_mAP_dict)
        eval_d['mAP']['NDS'] = AP_dict_json['NDS']
        eval_d['mAP']['mAP'] = AP_dict_json['mAP']

        if 'time_err' in eval_d:
            eval_d['avrg_recognize_time'] = 0.
            eval_d['avrg_instances_detected'] = 0
            for cls, timings_per_thr in  eval_d['time_err'].items():
                avrg_timing, avrg_instances = .0, .0
                for timings in timings_per_thr:
                    l = len(timings)
                    if l > 0:
                        avrg_timing += sum(list(timings.values())) / l
                        avrg_instances += l
                eval_d['avrg_recognize_time'] += avrg_timing / len(timings_per_thr)
                eval_d['avrg_instances_detected'] += avrg_instances/ len(timings_per_thr)
            eval_d['avrg_recognize_time'] /= len(eval_d['time_err'])
            eval_d['avrg_instances_detected'] /= len(eval_d['time_err'])

    if 'additional' in eval_d:
        for k,v in eval_d['additional'].items():
            eval_d[k] = v
        del eval_d['additional']

    return eval_d

# compare deadlines misses
def plot_func_dm(out_path, exps_dict):
    i=0
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    for exp_name, evals in exps_dict.items():
        for e in evals:
            x = [e['deadline_msec'] for e in evals]
            y = [e['deadlines_missed']/len(e['deadline_diffs'])*100. for e in evals]
        l2d = ax.plot(x, y, label=exp_name, 
            marker='.', markersize=10, markeredgewidth=0.7,
            c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
        i+=1
        #ax.scatter(x, y, color=l2d[0].get_c())
    ax.set_xticks(x)
    ax.invert_xaxis()
    ax.set_ylim(-5.0, 110.)
    ax.legend(fontsize='medium')
    ax.set_ylabel('Deadline miss ratio (%)', fontsize='x-large')
    ax.set_xlabel('Deadline (msec)', fontsize='x-large')
    ax.grid('True', ls='--')
    #fig.suptitle("Ratio of missed deadlines over a range of deadlines", fontsize=16)
    plt.savefig(out_path + "/deadlines_missed.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    labels = list(exps_dict.keys())
    #x_values = np.arange(len(labels))
    x_values = [k for k in exps_dict.keys()]
    y_values = [ round(evals[0]['deadlines_missed'] / \
            len(evals[0]['deadline_diffs']) * 100.) \
            for exp_name, evals in exps_dict.items()]

    rects = ax.bar(x_values, y_values, color=[e[0]['color'] for e in exps_dict.values()])
#    ax.tick_params(
#        axis='x',          # changes apply to the x-axis
#        which='both',      # both major and minor ticks are affected
#        bottom=False,      # ticks along the bottom edge are off
#        top=False,         # ticks along the top edge are off
#        labelbottom=False) # labels along the bottom edge are off
    autolabel(rects, ax)
    for r, l in zip(rects, labels):
        r.set_label(l)
    #ax.legend(fontsize='medium', ncol=3)
    ax.set_ylabel('Deadline miss ratio (%)', fontsize='x-large')
    #ax.set_xlabel(')', fontsize='x-large')
    #ax.grid('True', ls='--')
    ax.set_yticks(np.arange(0, 100.1, 20))
    ax.set_ylim(-5.0, 110.)

    plt.savefig(out_path + "/dl_miss_ratio_bar.pdf")


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')




def plot_func_rem_time_on_finish(out_path, exps_dict):
    # compare execution times end to end
    for exp_name, evals in exps_dict.items(): # This loop runs according to num of methods
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
        labels = [str(e['deadline_msec']) for e in evals]
        time_rem = [np.expand_dims(-np.array(e['deadline_diffs'])*1000., -1)  for e in evals]

        for i, (label, data) in enumerate(zip(labels, time_rem)):
            ax.boxplot(data, labels=[label], positions=[i], sym=".")
#        i+=1
        ax.invert_xaxis()
        ax.set_ylabel('Remaining time to deadline\non finish (msec)', fontsize='x-large')
        ax.set_xlabel('Deadline (msec)', fontsize='x-large')
#        #fig.suptitle("BB3D time pred. err.", fontsize='x-large')
        plt.savefig(out_path + f"/{exp_name}_rem_time_to_dl.pdf")




def plot_func_eted(out_path, exps_dict):
    i=0
    # compare execution times end to end
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    for exp_name, evals in exps_dict.items():
        x = [e['deadline_msec'] for e in evals]
        y = [e['exec_time_stats']['End-to-end'][1] for e in evals]
        l2d = ax.plot(x, y, label=exp_name,
            marker='.', markersize=10, markeredgewidth=0.7,
            c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
        i+=1
        ax.scatter(x, y, color=l2d[0].get_c())
    ax.invert_xaxis()
    ax.set_ylim(-1.0, 400)
    ax.legend(fontsize='medium', ncol=2)
    ax.set_ylabel('End-to-end time (msec)', fontsize='x-large')
    ax.set_xlabel('Deadline (msec)', fontsize='x-large')
    ax.grid('True', ls='--')
    #fig.suptitle("Average end-to-end time over different deadlines", fontsize=16)
    plt.savefig(out_path + "/end-to-end_deadlines.pdf")


# box plan can be cdf as well
def plot_func_component_time(out_path, exps_dict, plot_type='boxplot'):
    # compare three different cases
    e1, e2 = 'CenterPoint75', 'VALO-CP75'
    if e1 not in exps_dict or e2 not in exps_dict:
        return

    bl_eval_dict = exps_dict[e1][-1]
    alv2_eval_dict_1 = exps_dict[e2][-1]
    alv2_eval_dict_2 = exps_dict[e2][0] # is this the 100ms case?
    eval_data = [bl_eval_dict, alv2_eval_dict_1, alv2_eval_dict_2]

#    # Create CenterHead-PostAll
#    for data in eval_data:
#        et = data['exec_times']
#        et['CenterHead-PostAll'] = np.array(et['CenterHead-Topk']) + \
#                np.array(et['CenterHead-Post']) + np.array(et['CenterHead-GenBox'])

    components = ['Backbone3D', 'Backbone2D', 'CenterHead']

    fig, axes = plt.subplots(1, 1, figsize=(4, 6), constrained_layout=True)
    #axes = axes.ravel()

    #labels = [str(e['deadline_msec']) + 'msec' for e in eval_data]
    labels = [ \
        'CenterPoint75\nNo Deadline', \
        'VALO-CP75\nNo Deadline', \
        'VALO-CP75\n90 ms Deadline']
    #labels = [ 'CenterPoint75', 'VALO-CP75']

    for comp in components:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
        time_data = [np.array(ed['exec_times'][comp]) for ed in  eval_data]
        #bb3d_pred_err = [ np.expand_dims(arr, -1) for arr in bb3d_pred_err]

        for i, (label, data) in enumerate(zip(labels, time_data)):
            if plot_type == 'boxplot':
                ax.boxplot(data, labels=[label], positions=[i], sym=".")
            elif plot_type == 'cdf':
                data.sort()
                perc99 = np.percentile(data, 99)
                data = data[data < perc99]

                # Calculate the histogram
                hist, bin_edges = np.histogram(data, bins=100, density=True)
                # Calculate the CDF from the histogram
                cdf = np.cumsum(hist * np.diff(bin_edges))
                ax.plot(bin_edges[1:], cdf, marker='.', linestyle='-', markersize=2, label=label)

        #ax.invert_xaxis()
        #ax.set_title(f'{comp}', fontsize='x-large')
        if plot_type == 'boxplot':
            ax.set_ylabel('Execution time (msec)', fontsize='x-large')
        elif plot_type == 'cdf':
            ax.set_xlim(0.)
            ax.grid('True', ls='--')
            ax.set_ylabel('CDF', fontsize='x-large')
            ax.set_xlabel('Execution time (msec)', fontsize='x-large')
            ax.legend(fontsize='medium')
        plt.savefig(out_path + f"/{comp}_time_{plot_type}.pdf")

    #fig.suptitle("BB3D time pred. err.", fontsize='x-large')

def plot_func_bb3d_time_diff(out_path, exps_dict):
    # compare execution times end to end
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    for exp_name, evals in exps_dict.items(): # This loop runs according to num of methods
        if 'bb3d_preds' not in evals[0] or not evals[0]['bb3d_preds']:
            continue
        #evals = evals[:4] # use periods 100 150 200 250
        evals = [evals[1]] + [evals[3]]

        labels = []
        for e in evals:
            m = e['method']
            dl = e["deadline_msec"]
            if m == 4 or m == 6:
                labels.append(f'History (dl={dl} ms)')
            elif m == 5 or m == 7:
                labels.append(f'Quadratic (dl={dl} ms)')
            else:
                continue
                labels.append(f"VALO {str(e['deadline_msec'])}"
                        " ms deadline")

        #labels = [f"VALO {str(e['deadline_msec'])} ms period" for e in evals]
        bb3d_pred_err = [np.array(e['exec_times']['Backbone3D']) - np.array(e['bb3d_preds']) \
                for e in evals]
        if 'VoxelHead-conv-hm' in evals[0]['exec_times']:
            bb3d_pred_err2 = [np.array(e['exec_times']['VoxelHead-conv-hm']) for e in evals]
            bb3d_pred_err = [e1 + e2 for e1, e2 in zip(bb3d_pred_err, bb3d_pred_err2)]
        #bb3d_pred_err = [ np.expand_dims(arr, -1) for arr in bb3d_pred_err]

        for i, (label, data) in enumerate(zip(labels, bb3d_pred_err)):
            data_abs = np.abs(data)
            #data.sort()
            perc99 = np.percentile(data_abs, 99)
            data99 = data[data_abs < perc99]

            # Calculate the histogram
            hist, bin_edges = np.histogram(data99, bins=100, density=True)
            # Calculate the CDF from the histogram
            cdf = np.cumsum(hist * np.diff(bin_edges))
            ax.plot(bin_edges[1:], cdf, linestyle='-', label=label)

    ax.set_ylabel('CDF', fontsize='x-large')
    ax.set_xlabel('(Actual - Predicted) 3D Backbone time (msec)', fontsize='x-large')
    ax.legend()
    ax.grid('True', ls='--')
    ax.set_ylim(-0.05, 1.10)
    #ax.set_ylabel('Backbone 3D time\nActual - Predicted (msec)', fontsize='x-large')
    #ax.set_xlabel('Deadline (msec)', fontsize='x-large')
    #fig.suptitle("Backbone3D time prediction error", fontsize='x-large')
    plt.savefig(out_path + f"/{exp_name}_bb3d_pred_err.pdf")


def plot_func_area_processed(out_path, exps_dict):
    # compare execution times end to end
    for exp_name, evals in exps_dict.items(): # This loop runs according to num of methods
        if 'nonempty_tiles' not in evals[0] or not evals[0]['nonempty_tiles']:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
        labels = [str(e['deadline_msec']) for e in evals]
        nonempty_tiles = [e['nonempty_tiles']  for e in evals]
        num_nonempty_tiles = [np.array([len(t) for t in net]) for net in nonempty_tiles]
        processed_tiles = [e['chosen_tiles_2']  for e in evals]
        num_processed_tiles = [np.array([len(t) for t in net]) for net in processed_tiles]
        
        processed_area_perc = [np.expand_dims((p/n)*100., -1) \
                for p, n in zip(num_processed_tiles, num_nonempty_tiles)]

#        bb3d_pred_err = [ np.expand_dims(arr, -1) for arr in bb3d_pred_err]
        for i, (label, data) in enumerate(zip(labels, processed_area_perc)):
            ax.boxplot(data, labels=[label], positions=[i], sym=".")
#        i+=1
        ax.invert_xaxis()
        ax.set_ylabel('Processed area (%)', fontsize='x-large')
        ax.set_xlabel('Deadline (msec)', fontsize='x-large')
#        #fig.suptitle("BB3D time pred. err.", fontsize='x-large')
        plt.savefig(out_path + f"/{exp_name}_processed_area.pdf")


def plot_func_tile_drop_rate(out_path, exps_dict):
    # compare execution times end to end
    for exp_name, evals in exps_dict.items(): # This loop runs according to num of methods
        if exp_name != 'VALO' or 'chosen_tiles_1' not in evals[0] or not evals[0]['chosen_tiles_1']:
            continue

        evals = evals[:4] # use periods 100 150 200 250

        fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
        labels = [f"VALO {str(e['deadline_msec'])} ms period" for e in evals]
        chosen_tiles = [e['chosen_tiles_1']  for e in evals]
        num_chosen_tiles = [np.array([len(t) for t in net]) for net in chosen_tiles]
        processed_tiles = [e['chosen_tiles_2']  for e in evals]
        num_processed_tiles = [np.array([len(t) for t in net]) for net in processed_tiles]

        num_dropped_tiles = [c - p for p, c in zip(num_processed_tiles, num_chosen_tiles)]
        #dropped_area_perc = [np.expand_dims((1. - p/c)*100., -1) \
        #        for p, c in zip(num_processed_tiles, num_chosen_tiles)]

#        bb3d_pred_err = [ np.expand_dims(arr, -1) for arr in bb3d_pred_err]
        for i, (label, data) in enumerate(zip(labels, num_dropped_tiles)):
            #perc99 = np.percentile(data, 99)
            #data99 = data[data_abs < perc99]
            unq, cnts = np.unique(data, return_counts=True)

            print(label, ', num droped tiles:', unq, ', perc', \
                    np.round((cnts/data.shape[0])*1000)/10)

            # Calculate the histogram
            hist, bin_edges = np.histogram(data, bins=100, density=True)
            # Calculate the CDF from the histogram
            cdf = np.cumsum(hist * np.diff(bin_edges))
            ax.plot(bin_edges[1:], cdf, linestyle='-', label=label)

        ax.set_ylabel('CDF', fontsize='x-large')
        ax.set_xlabel('Number of dropped tiles after Backbone3D', fontsize='x-large')
        ax.legend()
        ax.set_ylim(0., 1.)
        ax.grid('True', ls='--')


#            ax.boxplot(data, labels=[label], positions=[i], sym=".")
#        i+=1
#        ax.invert_xaxis()
#        ax.set_ylabel('Dropped tiles after\nBackbone 3D (%)', fontsize='x-large')
#        ax.set_xlabel('Deadline (msec)', fontsize='x-large')
#        #fig.suptitle("BB3D time pred. err.", fontsize='x-large')
        plt.savefig(out_path + f"/{exp_name}_dropped_area.pdf")



# compare averaged AP of all classes seperately changing deadlines
def plot_avg_AP(out_path, merged_exps_dict):
    cls_per_file = 5
    cls_names = list(proto_AP_dict.keys())
    num_classes = len(cls_names)
    num_files = num_classes // cls_per_file + (num_classes % cls_per_file != 0)
    for filenum in range(num_files):
        if filenum == num_files-1:
            plot_num_classes = num_classes - (num_files-1) * cls_per_file
        else:
            plot_num_classes = cls_per_file
        fig, axs = plt.subplots(plot_num_classes, 1, \
                figsize=(12, 3*plot_num_classes), constrained_layout=True)
        cur_cls_names = cls_names[filenum*cls_per_file:filenum*cls_per_file+plot_num_classes]
        for ax, cls in zip(axs, cur_cls_names):
            for exp_name, evals in merged_exps_dict.items():
                x = evals['deadline_msec']
                y = evals['AP'][cls]['AP'] # for now, use AP as the only eval type as in nuscenes data
                y = [sum(e) / len(e) if len(e) > 0 else .0 for e in y ]
                l2d = ax.plot(x, y, label=exp_name)
                ax.scatter(x, y, color=l2d[0].get_c())
            ax.invert_xaxis()
            ax.legend(fontsize='medium')
            ax.set_ylabel(cls + ' AP', fontsize='large')
            ax.set_xlabel('Deadline (msec)', fontsize='large')
            ax.grid('True', ls='--')
        cur_cls_names_str = ""
        for s in cur_cls_names:
            cur_cls_names_str += s + ' '
        fig.suptitle(cur_cls_names_str + " classes, average precision over different deadlines", fontsize=16)
        plt.savefig(out_path + f"/AP_deadlines_{filenum}.pdf")

def plot_stage_and_head_usage(out_path, merged_exps_dict):
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
#    # axes[0] for num stages, axes[1] for heads, bar graph all, x axis deadlines
#    #calculate misses due to skipping heads
    i = 0
    for exp_name, evals in merged_exps_dict.items():
        x = evals['deadline_msec']
        y1, y2 = [], []
        for er in evals['rpn_stg_exec_seqs']:
            arr = [len(r[0]) for r in er]
            y1.append(sum(arr) / len(arr))
            arr = [len(r[1]) for r in er]
            y2.append(sum(arr) / len(arr))
        axes[0].plot(x, y1, label=exp_name,
                marker='.', markersize=10, markeredgewidth=0.7,
                c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
        axes[1].plot(x, y2, label=exp_name,
                marker='.', markersize=10, markeredgewidth=0.7,
                c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
        i+=1

    ylim = 3.5
    for ax, ylbl in zip(axes, ('Avrg. RPN stages', 'Avrg. det heads')):
        ax.invert_xaxis()
        ax.legend(fontsize='medium')
        ax.set_ylabel(ylbl, fontsize='x-large')
        ax.set_xlabel('Deadline (msec)', fontsize='x-large')
        ax.grid('True', ls='--')
        ax.set_ylim(0.0, ylim)
        ylim += 3.0

    plt.savefig(out_path + "/rpn_and_heads_stats.pdf")

def plot_instance_data(out_path, merged_exps_dict):
    i=0
    # compare execution times end to end
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
    for ax, k in zip(axs, ['avrg_instances_detected', 'avrg_recognize_time']):
        for exp_name, evals in exps_dict.items():
            x = [e['deadline_msec'] for e in evals]
            y = [e[k] for e in evals]
            l2d = ax.plot(x, y, label=exp_name,
                marker='.', markersize=10, markeredgewidth=0.7,
                c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
            i+=1
            ax.scatter(x, y, color=l2d[0].get_c())
        ax.invert_xaxis()
        ax.set_ylim(.0, 140)
        ax.legend(fontsize='medium')
        ax.set_ylabel(k, fontsize='x-large')
        ax.set_xlabel('Deadline (msec)', fontsize='x-large')
        ax.grid('True', ls='--')
    #fig.suptitle("Average end-to-end time over different deadlines", fontsize=16)
    plt.savefig(out_path + "/instance_data.pdf")


def plot_func_normalized_NDS(out_path, exps_dict, merged_exps_dict):
    max_NDS = 0.
    for exp_name, evals in exps_dict.items():
        NDS_arr = [e['mAP']['NDS'] for e in evals]
        max_NDS = max(max(NDS_arr), max_NDS)
    
    #max_NDS = 0.67
    #print('USING HARDCODED MAX NDS!')
    for exp_name, evals in merged_exps_dict.items():
        evals['mAP']['normalized_NDS'] = np.array(evals['mAP']['NDS']) / max_NDS * 100.
 
    #Add normalized accuracy
    i = 0
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    for exp_name, evals in merged_exps_dict.items():
        x = evals['deadline_msec']
        y = evals['mAP']['normalized_NDS']
        l2d = ax.plot(x, y, label=exp_name,
                marker='.', markersize=10, markeredgewidth=0.7,
                c=evals['color'][0], linestyle=evals['lnstyle'][0])
        i+=1
    ax.set_xticks(x)
    ax.invert_xaxis()
    ax.legend(fontsize='medium')
    ax.set_ylabel('Normalized accuracy (%)', fontsize='x-large')
    ax.set_xlabel('Deadline (msec)', fontsize='x-large')
    ax.grid('True', ls='--')
    ax.set_ylim(-5.0, 110.)

    plt.savefig(out_path + "/normalized_NDS_deadlines.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    labels = list(merged_exps_dict.keys())
    #x_values = np.arange(len(labels))
    x_values = labels
    y_values = [round(sum(evals['mAP']['normalized_NDS'])/ \
            len(evals['mAP']['normalized_NDS']),1) \
            for evals in merged_exps_dict.values()]

    rects = ax.bar(x_values, y_values, color=[v['color'][0] for v in merged_exps_dict.values()])
    #ax.tick_params(
    #    axis='x',          # changes apply to the x-axis
    #    which='both',      # both major and minor ticks are affected
    #    bottom=False,      # ticks along the bottom edge are off
    #    top=False,         # ticks along the top edge are off
    #    labelbottom=False) # labels along the bottom edge are off
    autolabel(rects, ax)
    for r, l in zip(rects, labels):
        r.set_label(l)
    #ax.legend(fontsize='medium', ncol=3)
    ax.set_ylabel('Normalized accuracy (%)', fontsize='x-large')
    #ax.set_xlabel(')', fontsize='x-large')
    #ax.grid('True', ls='--')
    ax.set_yticks(np.arange(0, 100.1, 20))
    ax.set_ylim(-1.0, 110.)

    plt.savefig(out_path + "/normalized_NDS_bar.pdf")

