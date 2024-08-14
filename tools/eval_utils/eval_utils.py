import pickle
import json
import time
import gc
import os
import copy

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

from eval_utils.centerpoint_tracker import CenterpointTracker as Tracker

speed_test = False
visualize = False
if visualize:
    import open3d
    from visual_utils import open3d_vis_utils as V

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    # Forward once for initialization and calibration
    batch_size = dataloader.batch_size
    if 'calibrate' in dir(model):
        torch.cuda.cudart().cudaProfilerStop()
        model.calibrate()
        torch.cuda.cudart().cudaProfilerStart()

    global speed_test
    num_samples = 20 if speed_test and len(dataset) >= 10 else len(dataset)
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    gc.disable()

    if visualize:
        V.initialize_visualizer()

    det_elapsed_musec = []
    if cfg.MODEL.STREAMING_EVAL:
        with open('token_to_pos.json', 'r') as handle:
            token_to_pose = json.load(handle)

        def get_ts(data_dict):
            return token_to_pose[data_dict['metadata']['token']]['timestamp']

        def get_scene_token(data_dict):
            return token_to_pose[data_dict['metadata']['token']]['scene']

        all_data_dicts = [dataset.get_metadata_dict(i) for i in range(len(dataloader))]
        print('Loaded data dicts')
        all_scene_tokens = [get_scene_token(dd) for dd in all_data_dicts]
        tokens_and_num_samples, idx = [[all_scene_tokens[0], 0]], 0
        for tkn in all_scene_tokens:
            if tkn != tokens_and_num_samples[idx][0]:
                tokens_and_num_samples.append([tkn, 0])
                idx += 1
            tokens_and_num_samples[idx][1] += 1

        print('Scenes and number samples in them:')
        for tns in tokens_and_num_samples:
            print(tns)

        # process each scene seperately
        # NOTE For nonblocking, set E2E_REL_DEADLINE_S to 0
        do_dyn_sched = bool(int(os.getenv('DO_DYN_SCHED', '0')))
        e2e_dl_musec = int(float(os.getenv('E2E_REL_DEADLINE_S', 0.1)) * 1000000)
        print('Dynamic Scheduling:', 'ON' if do_dyn_sched else 'OFF')
        print('End to end deadline (microseconds):', 'IGNORED' if do_dyn_sched else e2e_dl_musec)
        det_idx = 0
        all_sample_tokens = []
        for scene_token, num_samples in tokens_and_num_samples:
            # initialize buffer
            buffered_pred_dicts = [model.get_dummy_det_dict()]
            #buf_det_idx = -1 #DEBUG

            scene_end_idx = det_idx + num_samples
            samples_added = 0

            det_ts = get_ts(all_data_dicts[det_idx])
            #init_ts = det_ts # DEBUG

            while det_idx < scene_end_idx:
                with torch.no_grad():
                    ts = get_ts(all_data_dicts[det_idx])
                    inf_idx = det_idx if det_ts == ts else det_idx - 1
                    #print(f'Init detection {inf_idx}')
                    pred_dicts, ret_dict = model([inf_idx])
                #save_det_idx = det_idx #DEBUG
                disp_dict = {}
                statistics_info(cfg, ret_dict, metric, disp_dict)

                etm = model.last_elapsed_time_musec
                det_end_ts = det_ts + etm
                if do_dyn_sched:
                    # Below three lines are not the real dyn sched
                    #etm_cut = (etm - (etm % 50000))
                    #etm = etm_cut if etm % 50000 < 25000 else etm_cut + 50000
                    #e2e_end_ts = det_ts + etm

                    # Compute the cur tail and approx next tail
                    while inf_idx+1 < scene_end_idx and \
                            get_ts(all_data_dicts[inf_idx+1]) < det_end_ts:
                        inf_idx += 1

                    wait_idx = inf_idx + 1
                    cur_tail = det_end_ts - get_ts(all_data_dicts[inf_idx])

                    # Assume the next execution time will be same (etm)
                    next_det_end_ts = det_end_ts + etm
                    while inf_idx+1 < scene_end_idx and \
                            get_ts(all_data_dicts[inf_idx+1]) < next_det_end_ts:
                        inf_idx += 1
                    next_tail = next_det_end_ts - get_ts(all_data_dicts[inf_idx])

                    # 0 will force nonblocking
                    #print(f'cur_tail: {cur_tail}, next_tail: {next_tail}')
                    if wait_idx < scene_end_idx and next_tail < cur_tail:
                        e2e_end_ts = get_ts(all_data_dicts[wait_idx])
                    else:
                        e2e_end_ts = 0

                elif e2e_dl_musec > 0:
                    e2e_end_ts = det_ts + e2e_dl_musec

                while det_ts < det_end_ts and det_idx < scene_end_idx:
                    dd = all_data_dicts[det_idx]
                    bd = {'metadata': [{'token':dd['metadata']['token']}],
                            'frame_id': [dd['frame_id']]}
                    all_sample_tokens.append(dd['metadata']['token'])
                    annos = dataset.generate_prediction_dicts(
                        bd, copy.deepcopy(buffered_pred_dicts), class_names,
                        output_path=final_output_dir if args.save_to_file else None
                    )
                    det_annos += annos

                    if visualize:
                        # Can infer which detections are projection from the scores
                        # -x -y -z +x +y +z
                        batch_dict = dataset.getitem_pre(det_idx)
                        pd = buffered_pred_dicts[0]
                        #print(batch_dict['gt_boxes'][:5, :])
                        lbd = model.latest_batch_dict
                        V.draw_scenes(
                            points=batch_dict['points'],
                            ref_boxes=pd['pred_boxes'],
                            gt_boxes=batch_dict['gt_boxes'],
                            ref_scores=pd['pred_scores'],
                            ref_labels=pd['pred_labels'],
                            max_num_tiles=(model.tcount if hasattr(model, 'tcount') else None),
                            pc_range=model.vfe.point_cloud_range.cpu().numpy(),
                            nonempty_tile_coords=lbd.get('nonempty_tile_coords', None),
                            tile_coords=lbd.get('chosen_tile_coords', None),
                            clusters=lbd.get('clusters', None))

                    #print(f'Using det {buf_det_idx} for {det_idx}') # DEBUG
                    det_idx += 1
                    samples_added += 1
                    progress_bar.set_postfix(disp_dict)
                    progress_bar.update()
                    if det_idx < scene_end_idx:
                        det_ts = get_ts(all_data_dicts[det_idx])

                #buf_det_idx = save_det_idx # DEBUG
                #print(f'Detected {save_det_idx}') # DEBUG
                buffered_pred_dicts = copy.deepcopy(pred_dicts)

                late = (e2e_end_ts < det_end_ts)
                if late:
                    det_ts = det_end_ts
                else:
                    # since e2e_end_ts is not exact, we break when we are close,
                    # assuming the time between sapmles are 50000 microseconds

                    while round((e2e_end_ts - det_ts) / 50000.) > 0 and det_idx < scene_end_idx:
                        dd = all_data_dicts[det_idx]
                        bd = {'metadata': [{'token':dd['metadata']['token']}],
                                'frame_id': [dd['frame_id']]}
                        all_sample_tokens.append(dd['metadata']['token'])
                        annos = dataset.generate_prediction_dicts(
                            bd, copy.deepcopy(buffered_pred_dicts), class_names,
                            output_path=final_output_dir if args.save_to_file else None
                        )
                        det_annos += annos
                        #print(f'Using det {buf_det_idx} for {det_idx}') # DEBUG

                        if visualize:
                            # Can infer which detections are projection from the scores
                            # -x -y -z +x +y +z
                            batch_dict = dataset.getitem_pre(det_idx)
                            pd = buffered_pred_dicts[0]
                            lbd = model.latest_batch_dict
                            V.draw_scenes(
                                points=batch_dict['points'], ref_boxes=pd['pred_boxes'],
                                gt_boxes=batch_dict['gt_boxes'],# ???
                                ref_scores=pd['pred_scores'], ref_labels=pd['pred_labels'],
                                max_num_tiles=(model.tcount if hasattr(model, 'tcount') else None),
                                pc_range=model.vfe.point_cloud_range.cpu().numpy(),
                                nonempty_tile_coords=lbd.get('nonempty_tile_coords', None),
                                tile_coords=lbd.get('chosen_tile_coords', None),
                                clusters=lbd.get('clusters', None))

                        det_idx += 1
                        samples_added += 1
                        progress_bar.set_postfix(disp_dict)
                        progress_bar.update()
                        if det_idx < scene_end_idx:
                            det_ts = get_ts(all_data_dicts[det_idx])

                gc.collect()

            assert samples_added == num_samples

    else:
        for i in range(len(dataloader)):
            if speed_test and i == num_samples:
                break
            if getattr(args, 'infer_time', False):
                start_time = time.time()
            data_indexes = [i*batch_size+j for j in range(batch_size) \
                    if i*batch_size+j < len(dataset)]
            with torch.no_grad():
                pred_dicts, ret_dict = model(data_indexes)
            batch_dict = model.latest_batch_dict
            det_elapsed_musec.append(model.last_elapsed_time_musec)
            disp_dict = {}

            if getattr(args, 'infer_time', False):
                inference_time = time.time() - start_time
                infer_time_meter.update(inference_time * 1000)
                # use ms to measure inference time
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

            if visualize:
                # Can infer which detections are projection from the scores
                # -x -y -z +x +y +z
                pd = batch_dict['final_box_dicts'][0]
                V.draw_scenes(
                    points=batch_dict['points'][:, 1:], ref_boxes=pd['pred_boxes'],
                    gt_boxes=batch_dict['gt_boxes'].cpu().flatten(0,1).numpy(),
                    ref_scores=pd['pred_scores'], ref_labels=pd['pred_labels'],
                    max_num_tiles=(model.tcount if hasattr(model, 'tcount') else None),
                    pc_range=model.vfe.point_cloud_range.cpu().numpy(),
                    nonempty_tile_coords=batch_dict.get('nonempty_tile_coords', None),
                    tile_coords=batch_dict.get('chosen_tile_coords', None),
                    clusters=batch_dict.get('clusters', None))

            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if args.save_to_file else None
            )
            det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

            gc.collect()
    gc.enable()

    if visualize:
        V.destroy_visualizer()

    if 'post_eval' in dir(model):
        model.post_eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    model.print_time_stats()

    if cfg.LOCAL_RANK != 0:
        return {}

    if speed_test:
        model.dump_eval_dict(ret_dict)
        model.clear_stats()
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    do_eval = (int(os.getenv('DO_EVAL', 1)) == 1)

    do_tracking=False
    if do_eval:
        if dataset.dataset_cfg.DATASET != 'NuScenesDataset':
            result_str, result_dict = dataset.evaluation(
                det_annos, class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path=final_output_dir,
            )
        else:
            nusc_annos = {}
            result_str, result_dict = dataset.evaluation(
                det_annos, class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path=final_output_dir,
                nusc_annos_outp=nusc_annos,
                #det_elapsed_musec=det_elapsed_musec,
            )

            if do_tracking:
                ## NUSC TRACKING START
                tracker = Tracker(max_age=6, hungarian=False)
                predictions = nusc_annos['results']
                with open('frames_meta.json', 'rb') as f:
                    frames=json.load(f)['frames']

                nusc_trk_annos = {
                    "results": {},
                    "meta": None,
                }
                size = len(frames)

                print("Begin Tracking\n")
                start = time.time()
                for i in range(size):
                    token = frames[i]['token']

                    # reset tracking after one video sequence
                    if frames[i]['first']:
                        # use this for sanity check to ensure your token order is correct
                        # print("reset ", i)
                        tracker.reset()
                        last_time_stamp = frames[i]['timestamp']

                    time_lag = (frames[i]['timestamp'] - last_time_stamp)
                    last_time_stamp = frames[i]['timestamp']

                    preds = predictions[token]

                    outputs = tracker.step_centertrack(preds, time_lag)
                    annos = []

                    for item in outputs:
                        if item['active'] == 0:
                            continue
                        nusc_anno = {
                            "sample_token": token,
                            "translation": item['translation'],
                            "size": item['size'],
                            "rotation": item['rotation'],
                            "velocity": item['velocity'],
                            "tracking_id": str(item['tracking_id']),
                            "tracking_name": item['detection_name'],
                            "tracking_score": item['detection_score'],
                        }
                        annos.append(nusc_anno)
                    nusc_trk_annos["results"].update({token: annos})
                end = time.time()
                second = (end-start)
                speed=size / second
                print("The speed is {} FPS".format(speed))
                nusc_trk_annos["meta"] = {
                    "use_camera": False,
                    "use_lidar": True,
                    "use_radar": False,
                    "use_map": False,
                    "use_external": False,
                }

                with open('tracking_result.json', "w") as f:
                    json.dump(nusc_trk_annos, f)

                #result is nusc_annos
                dataset.tracking_evaluation(
                    output_path=final_output_dir,
                    res_path='tracking_result.json'
                )

                ## NUSC TRACKING END

    if do_eval:
        logger.info(result_str)
        ret_dict.update(result_dict)
        ret_dict['result_str'] = result_str
    else:
        print('Skipping evaluation.')
        print('Dumping eval data')
        t0 = time.time()
        with open('eval.pkl', 'wb') as f:
            all_data = [dataset, det_annos, class_names,cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                final_output_dir, {}, det_elapsed_musec]
            pickle.dump(all_data, f)
        print(f'Dumping took {(time.time() - t0):.2f} seconds.')

    if cfg.MODEL.STREAMING_EVAL:
        ret_dict['e2e_dl_musec'] = e2e_dl_musec

#    logger.info('Result is saved to %s' % result_dir)
#    logger.info('****************Evaluation done.*****************')

    model.dump_eval_dict(ret_dict)
    model.clear_stats()

    return ret_dict


if __name__ == '__main__':
    pass
