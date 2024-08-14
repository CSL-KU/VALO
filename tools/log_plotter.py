import os
import glob
import sys
import json
import threading
import concurrent.futures
from multiprocessing import Process
from plot_utils import *

#linestyles = ['-', '--', '-.', ':'] * 4
linestyles = ['--', ':'] * 10
method_colors= [
    'tab:purple', 
    'tab:red', 
    'tab:olive', 
    'tab:orange', 
    'tab:green',
    'tab:brown',
    'tab:blue', 
    'tab:pink',  #'xkcd:coral', 
    'xkcd:coral', 
    'tab:pink',
    'tab:orange',
    'tab:red',
    'tab:blue',
    'tab:purple', 
]

method_num_to_str = [
        '0CenterPoint75',
        '1CenterPoint100',
        '2CenterPoint200',
        '3VoxelNext75',
        '4VALO-CP75',
        '5VALO-BLTP-CP75',
        '6VALO-VN75',
        '7VALO-BLTP-VN75',
        '8VALO-NF-CP75',
        '9VALO-NSNF-CP75',
        '9AnytimeLidar-CP75',
        '9VALO-CP100',
        '9VALO-NRD-CP75', # no tile drop
]

method_remap = {i:i for i in range(len(method_num_to_str))}

if __name__ == '__main__':
    inp_dir = sys.argv[1]
   
    any_path = glob.glob(inp_dir + "/eval_dict_*json")[0]
    with open(any_path, 'r') as handle:
        eval_d = json.load(handle)
        dataset = eval_d.get('dataset','NuScenesDataset')
        init_dicts(dataset)

    # load eval dicts
    exps_dict = {}
    m_to_c_ls = [(method_colors[i], linestyles[i]) for i in range(len(method_colors))]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futs = []
        paths = sorted(glob.glob(inp_dir + "/eval_dict_*json"))
        for path in paths:
            futs.append(executor.submit(load_eval_dict, path))
        for f in concurrent.futures.as_completed(futs):
            ed = f.result()
            method = method_remap[ed['method']]
            k = method_num_to_str[method]
            if k not in exps_dict:
                exps_dict[k] = []
            ed['color'] = m_to_c_ls[method][0]
            ed['lnstyle'] = m_to_c_ls[method][1]
            exps_dict[k].append(ed)

    #Sort exps
    exp_names = sorted(exps_dict.keys())
    exps_dict= {nm[1:]:exps_dict[nm] for nm in exp_names}

    plot_sets=[]


    plot_sets.append({ nm:exps_dict[nm] for nm in [ \
            'CenterPoint75',
            'AnytimeLidar-CP75',
            'VALO-CP75',
    ]})

    plot_sets.append({ nm:exps_dict[nm] for nm in [ \
            'CenterPoint75',
            'CenterPoint100',
            'CenterPoint200',
            'VALO-CP75',
    ]})

    plot_sets.append({ nm:exps_dict[nm] for nm in [ \
            'CenterPoint75',
            'VALO-NSNF-CP75',
            'VALO-NF-CP75',
            #'VALO-NRD-CP75',
            'VALO-CP75',
    ]})

    plot_sets.append({ nm:exps_dict[nm] for nm in [ \
            'VoxelNext75',
            'VALO-BLTP-VN75',
            'VALO-VN75',
    ]})

    #plot_sets.append({ nm:exps_dict[nm] for nm in [ \
    #        'CenterPoint75',
    #        'VALO-CP75',
    #        #'VALO-NRD-CP75',
    #]})


    plot_set_choice = int(sys.argv[2])
    exps_dict=plot_sets[plot_set_choice]
    out_path="/root/shared_data/exp_plots/set" + sys.argv[2]
    for p in ["/root/shared_data/exp_plots", out_path]:
        try:
            os.mkdir(p)
        except FileExistsError:
            pass

    plot_head_selection = False

    for exp, evals in exps_dict.items():
        # Sort according to deadlines
        evals.sort(key=lambda e: e['deadline_sec'])
        evals.sort(key=lambda e: e['deadline_msec'])
        print('Experiment:',exp)
        for e in evals:
            if dataset == 'KittiDataset':
                mAP_image, mAP_bev, mAP_3d = e["mAP"]['image'], e["mAP"]['bev'], e["mAP"]['3d']
                print('\tdeadline:', e['deadline_sec'], "\tmissed:", e['deadlines_missed'],
                      f"\tmAP (image, bev, 3d):\t{mAP_image:.2f},\t{mAP_bev:.2f},\t{mAP_3d:.2f}")
            elif dataset == 'NuScenesDataset':
                mAP, NDS = e["mAP"]['mAP'], e["mAP"]['NDS']
                print('\tdeadline:', e['deadline_sec'], "\tmissed:", e['deadlines_missed'],
                      f"\tmAP, NDS:\t{mAP:.4f},\t{NDS:.4f}")
    merged_exps_dict = {}
    for k, v in exps_dict.items():
        merged_exps_dict[k] = merge_eval_dicts(v)

    # for plotting
    procs = []

    procs.append(Process(target=plot_func_dm, args=(out_path, exps_dict,)))
    procs[-1].start()

    procs.append(Process(target=plot_func_eted, args=(out_path, exps_dict,)))
    procs[-1].start()

    procs.append(Process(target=plot_func_normalized_NDS, args=(out_path, exps_dict,merged_exps_dict)))
    procs[-1].start()

    procs.append(Process(target=plot_func_bb3d_time_diff, args=(out_path, exps_dict)))
    procs[-1].start()
   
    procs.append(Process(target=plot_func_component_time, args=(out_path, exps_dict, 'boxplot')))
    procs[-1].start()

    procs.append(Process(target=plot_func_area_processed, args=(out_path, exps_dict)))
    procs[-1].start()
    
    procs.append(Process(target=plot_func_tile_drop_rate, args=(out_path, exps_dict)))
    procs[-1].start()

    procs.append(Process(target=plot_func_rem_time_on_finish, args=(out_path, exps_dict)))
    procs[-1].start()

    for p in procs:
        p.join()

    sys.exit(0)
