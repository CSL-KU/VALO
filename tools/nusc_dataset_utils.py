#!/root/trainvalconda3/envs/pointpillars/bin/python
import sys
import os
import json
import math
import copy
import random
import uuid
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from nuscenes.utils.splits import train, val, mini_train, mini_val

all_scenes = set(train + val + mini_train + mini_val)
nusc = None

def read_nusc():
    global nusc
    nusc = NuScenes(version='v1.0-trainval', dataroot='../data/nuscenes/v1.0-trainval', verbose=True)
    #nusc.list_scenes()

# atan2 to quaternion:
# Quaternion(axis=[0, 0, 1], radians=atan2results)

def generate_pose_dict():
    print('generating pose dict')
    global nusc
    token_to_cs_and_pose = {}

    global all_scenes
    for scene in nusc.scene:
        if scene['name'] not in all_scenes:
            #print(f'Skipping {scene["name"]}')
            continue
        tkn = scene['first_sample_token']
        while tkn != "":
            #print('token:',tkn)
            sample = nusc.get('sample', tkn)
            #print('timestamp:', sample['timestamp'])
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = nusc.get('calibrated_sensor',
                    sample_data['calibrated_sensor_token'])
            #print('calibrated sensor translation:', cs['translation'])
            #print('calibrated sensor rotation:', cs['rotation'])
            pose = nusc.get('ego_pose',
                sample_data['ego_pose_token'])
            #print('ego pose translation:', pose['translation'])
            #print('ego pose rotation:', pose['rotation'])
            scene_name = nusc.get('scene', sample['scene_token'])['name']
            token_to_cs_and_pose[tkn] = {
                    'timestamp' : sample['timestamp'],
                    'scene' : sample['scene_token'],
                    'scene_name': scene_name,
                    'cs_translation' : cs['translation'],
                    'cs_rotation' : cs['rotation'],
                    'ep_translation' : pose['translation'],
                    'ep_rotation' : pose['rotation'],
            }
            tkn = sample['next']

    print('Dict size:', sys.getsizeof(token_to_cs_and_pose)/1024/1024, ' MB')

    with open('token_to_pos.json', 'w') as handle:
        json.dump(token_to_cs_and_pose, handle, indent=4)


def generate_anns_dict():
    print('generating annotations dict')
    global nusc

    map_name_from_general_to_detection = {
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.wheelchair': 'ignore',
        'human.pedestrian.stroller': 'ignore',
        'human.pedestrian.personal_mobility': 'ignore',
        'human.pedestrian.police_officer': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'animal': 'ignore',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.emergency.ambulance': 'ignore',
        'vehicle.emergency.police': 'ignore',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'barrier',
        'movable_object.trafficcone': 'traffic_cone',
        'movable_object.pushable_pullable': 'ignore',
        'movable_object.debris': 'ignore',
        'static_object.bicycle_rack': 'ignore',
    }

    classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    token_to_anns = {}

    global all_scenes
    for scene in nusc.scene:
        if scene['name'] not in all_scenes:
            #print(f'Skipping {scene["name"]}')
            continue
        tkn = scene['first_sample_token']
        #print(scene['name'])
        categories_in_scene = set()
        while tkn != "":
            #print('token:',tkn)
            sample = nusc.get('sample', tkn)
            #print('timestamp:', sample['timestamp'])
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = nusc.get('calibrated_sensor',
                    sample_data['calibrated_sensor_token'])
            #print('calibrated sensor translation:', cs['translation'])
            #print('calibrated sensor rotation:', cs['rotation'])
            pose = nusc.get('ego_pose',
                sample_data['ego_pose_token'])
            #print('ego pose translation:', pose['translation'])
            #print('ego pose rotation:', pose['rotation'])

            annos = np.zeros((len(sample['anns']),9))
            labels = []
            num_ignored = 0
            for i, anno_token in enumerate(sample['anns']):
                anno = nusc.get('sample_annotation', anno_token)
                cn = anno['category_name']
                name = map_name_from_general_to_detection[cn]
                if name == 'ignore':
                    num_ignored += 1
                    continue
                categories_in_scene.add(name)
                labels.append(classes.index(name)+1)
                #print(anno['category_name'])
                anno_vel = nusc.box_velocity(anno_token)
                box = Box(anno['translation'], anno['size'],
                    Quaternion(anno['rotation']), velocity=tuple(anno_vel))
                box.translate(-np.array(pose['translation']))
                box.rotate(Quaternion(pose['rotation']).inverse)
                box.translate(-np.array(cs['translation']))
                box.rotate(Quaternion(cs['rotation']).inverse)

                idx = i - num_ignored
                annos[idx, :3] = box.center
                annos[idx, 3] = box.wlh[1]
                annos[idx, 4] = box.wlh[0]
                annos[idx, 5] = box.wlh[2]
                r, x, y, z = box.orientation.elements
                annos[idx, 6] = 2. * math.atan2(math.sqrt(x*x+y*y+z*z),r)
                annos[idx, 7:] = box.velocity[:2] # this is actually global velocity
            annos = annos[:annos.shape[0]-num_ignored]

            labels = np.array(labels)
            indices = labels.argsort()
            labels.sort()
            annos = annos[indices]
            #print('Annos:\n', annos)
            token_to_anns[tkn] = {
                'pred_boxes': annos.tolist(),
                'pred_scores': [1.0] * annos.shape[0],
                'pred_labels': labels.tolist(),
            }
            tkn = sample['next']
        print(len(categories_in_scene), categories_in_scene)

    #print('Dict size:', sys.getsizeof(token_to_anns)/1024/1024, ' MB')

    with open('token_to_anns.json', 'w') as handle:
        json.dump(token_to_anns, handle, indent=4)

def gen_new_token(table_name):
    # Generate a unique anno token
    # each token is 32 chars
    global nusc
    
    while True:
        new_token = uuid.uuid4().hex
        if new_token not in nusc._token2ind[table_name]:
            nusc._token2ind[table_name][new_token] = -1 # enough for now
            break

    return new_token

# step defines the time between populated annotations in milliseconds
# step 50ms, 100ms, 150ms, ...
def populate_annos_v2(step):
    print('populating annotations')
    global nusc
    step = step//50
    scene_to_sd = {}
    scene_to_sd_cam = {}
    for i, sd_rec in enumerate(nusc.sample_data):
        for channel, dct in zip(['LIDAR_TOP', 'CAM_FRONT'], \
                [scene_to_sd, scene_to_sd_cam]):
            if sd_rec['channel'] == channel:
                scene_tkn = nusc.get('sample', sd_rec['sample_token'])['scene_token']
                if scene_tkn not in dct:
                    dct[scene_tkn] = []
                dct[scene_tkn].append(sd_rec)

    for dct in [scene_to_sd, scene_to_sd_cam]:
        for k, v in dct.items():
            dct[k] = sorted(v, key=lambda item: item['timestamp'])

    scene_to_kf_indexes = {}
    for k, v in scene_to_sd.items():
        # Filter based on time, also filter the ones which cannot
        # be interpolated
        is_kf_arr = [sd['is_key_frame'] for sd in v]
        kf_indexes = [i for i in range(len(is_kf_arr)) if is_kf_arr[i]]
        scene_to_kf_indexes[k] = kf_indexes

    all_new_sample_datas = []
    all_new_samples = []
    all_new_annos = []
    global all_scenes
    for scene in nusc.scene:
        if scene['name'] not in all_scenes:
            #print(f'Skipping {scene["name"]}')
            continue
        #print('Processing scene', scene['name'])
        sd_records = scene_to_sd[scene['token']]
        sd_records_cam = scene_to_sd_cam[scene['token']]
        kf_indexes = scene_to_kf_indexes[scene['token']]
        for idx in range(len(kf_indexes) - 1):
            # generate sample between these two
            begin_kf_idx = kf_indexes[idx]
            end_kf_idx = kf_indexes[idx+1]
            cur_sample = nusc.get('sample', sd_records[begin_kf_idx]['sample_token'])
            next_sample = nusc.get('sample', sd_records[end_kf_idx]['sample_token'])
            # if these two are equal, this is a problem for interpolation
            assert cur_sample['token'] != next_sample['token']
            sd_rec_indexes = np.arange(begin_kf_idx+step, end_kf_idx-step+1, step)

            new_samples = []
            new_sample_annos = []
            for sd_rec_idx in sd_rec_indexes:
                sd_rec = sd_records[sd_rec_idx]
                new_token = gen_new_token('sample')
                # find the sd_record_cam with closest timestamp
                lidar_ts = sd_rec['timestamp']
                cam_ts_arr = np.asarray([sd_rec_cam['timestamp'] \
                        for sd_rec_cam in sd_records_cam])
                cam_idx = (np.abs(cam_ts_arr - lidar_ts)).argmin()
                sd_rec_cam = sd_records_cam[cam_idx]
                new_samples.append({
                        'token': new_token,
                        'timestamp' : lidar_ts,
                        'prev': "",
                        'next': "",
                        'scene_token': scene['token'],
                        'data': {'LIDAR_TOP': sd_rec['token'],
                            'CAM_FRONT': sd_rec_cam['token']},
                        'anns': [],
                })

                # update sample data record
                sd_rec['sample_token'] = new_samples[-1]['token']
                sd_rec['is_key_frame'] = True # not sure this is right
                if not sd_rec_cam['is_key_frame']:
                    sd_rec_cam['sample_token'] = new_samples[-1]['token']
                    sd_rec_cam['is_key_frame'] = True # not sure this is right
                else:
                    # Fabricate an sd_rec_cam with a new token
                    # because we cannot override this one as it is a keyframe
                    new_sd_rec_cam = copy.deepcopy(sd_rec_cam)
                    new_token = gen_new_token('sample_data')
                    new_sd_rec_cam['token'] = new_token
                    new_sd_rec_cam['sample_token'] = new_samples[-1]['token'] 
                    # I am not sure whether this one should be befor or after
                    # sd_rec_cam, but I will assume it will be after
                    new_sd_rec_cam['prev'] = sd_rec_cam['token']
                    new_sd_rec_cam['next'] = sd_rec_cam['next']
                    if new_sd_rec_cam['next'] != "":
                        nusc.get('sample_data', new_sd_rec_cam['next'])['prev'] = \
                                new_token
                    sd_rec_cam['next'] = new_token

                    # Do I need to generate a corresponding ego_pose_rec? I hope not
                    all_new_sample_datas.append(new_sd_rec_cam)

            # link the samples
            if not new_samples:
                continue

            cur_sample['next'] = new_samples[0]['token']
            assert cur_sample['timestamp'] < new_samples[0]['timestamp']
            new_samples[0]['prev'] = cur_sample['token']
            for i in range(1, len(new_samples)):
                new_samples[i-1]['next'] = new_samples[i]['token']
                new_samples[i]['prev'] = new_samples[i-1]['token']
            new_samples[-1]['next'] = next_sample['token']
            next_sample['prev'] = new_samples[-1]['token']

            # Generate annotations
            # For each anno in the cur_sample, find its corresponding anno
            # in the next sample. The matching can be done via instance_token
            total_time_diff = next_sample['timestamp'] - cur_sample['timestamp']
            for cur_anno_tkn in cur_sample['anns']:
                cur_anno = nusc.get('sample_annotation', cur_anno_tkn)
                next_anno_tkn = cur_anno['next']
                if next_anno_tkn == "":
                    continue
                next_anno = nusc.get('sample_annotation', next_anno_tkn)

                new_annos = []
                # Interpolate this anno for all new samples
                for new_sample in new_samples:
                    new_token = gen_new_token('sample_annotation')
                    new_anno = copy.deepcopy(cur_anno)

                    new_anno['token'] = new_token
                    new_anno['sample_token'] = new_sample['token']
                    new_sample['anns'].append(new_token)

                    time_diff = new_sample['timestamp'] - cur_sample['timestamp']
                    rratio = time_diff / total_time_diff
                    new_anno['translation'] = (1.0 - rratio) * \
                            np.array(cur_anno['translation'], dtype=float) + \
                            rratio * np.array(next_anno['translation'], dtype=float)
                    new_anno['translation'] = new_anno['translation'].tolist()
                    new_anno['rotation'] = Quaternion.slerp(
                            q0=Quaternion(cur_anno['rotation']),
                            q1=Quaternion(next_anno['rotation']),
                            amount=rratio
                    ).elements.tolist()
                    new_anno['prev'] = ''
                    new_anno['next'] = ''
                    new_annos.append(new_anno)

                # link the annos
                cur_anno['next'] = new_annos[0]['token']
                new_annos[0]['prev'] = cur_anno_tkn
                for i in range(1, len(new_annos)):
                    new_annos[i-1]['next'] = new_annos[i]['token']
                    new_annos[i]['prev'] = new_annos[i-1]['token']
                new_annos[-1]['next'] = next_anno_tkn
                next_anno['prev'] = new_annos[-1]['token']

                all_new_annos.extend(new_annos)
                # increase the number of annos in the instance table
                nusc.get('instance', cur_anno['instance_token'])['nbr_annotations'] += \
                        len(new_annos)

            all_new_samples.extend(new_samples)

            scene['nbr_samples'] += len(new_samples)

    nusc.sample.extend(all_new_samples)
    nusc.sample_annotation.extend(all_new_annos)
    nusc.sample_data.extend(all_new_sample_datas)


def prune_annos(step):
    print('pruning annotations')
    global nusc
    #num_lidar_sample_data = sum([sd['channel'] == 'LIDAR_TOP' for sd in nusc.sample_data])
    # make sure the number of samples are equal to number of sample_datas
    #assert len(nusc.sample) == num_lidar_sample_data, \
    #        "{len(nusc.sample)}, {num_lidar_sample_data}"

    step = step//50
    print('step', step)
    num_skips=step-1
    new_nusc_samples = []
    discarded_nusc_samples = []
    global all_scenes
    for scene in nusc.scene:
        if scene['name'] not in all_scenes:
            #print(f'Skipping {scene["name"]}')
            continue
        # skip skip skip get, skip skip skip get...
        samples_to_del = [] # sample token : replacement sample
        samples_to_connect = []
        sample_tkn = scene['first_sample_token']
        for i in range(num_skips):
            if sample_tkn != '':
                sample = nusc.get('sample', sample_tkn)
                samples_to_del.append(sample)
                sample_tkn = sample['next']
        while sample_tkn != '':
            sample = nusc.get('sample', sample_tkn)
            samples_to_connect.append(sample)
            sample_tkn = sample['next']
            for i in range(num_skips):
                if sample_tkn != '':
                    sample = nusc.get('sample', sample_tkn)
                    samples_to_del.append(sample)
                    sample_tkn = sample['next']

        #print(f'samples to connect {len(samples_to_connect)}')
        #print(f'samples to del {len(samples_to_del)}')
        # Update the scene
        scene['first_sample_token'] = samples_to_connect[0]['token']
        scene['last_sample_token'] = samples_to_connect[-1]['token']

        #update samples
        samples_to_connect[0]['prev'] = ''
        for i in range(len(samples_to_connect)-1):
            s1, s2 = samples_to_connect[i], samples_to_connect[i+1]
            s1['next'] = s2['token']
            s2['prev'] = s1['token']
        samples_to_connect[-1]['next'] = ''

        #delete the samples
        new_nusc_samples.extend(samples_to_connect)
        discarded_nusc_samples.extend(samples_to_del)

    tokens_c = set([s['token'] for s in new_nusc_samples])
    ts_arr_c = np.array([s['timestamp'] for s in new_nusc_samples])
    tokens_d = set([s['token'] for s in discarded_nusc_samples])

    new_nusc_sample_datas = []
    for sd in nusc.sample_data:
        tkn = sd['sample_token']
        if tkn in tokens_c:
            sd['is_key_frame'] = True
            new_nusc_sample_datas.append(sd)
        elif tkn in tokens_d:
            sd['is_key_frame'] = False
            new_nusc_sample_datas.append(sd)
            # point to the sample with closest timestamp

            sd_ts = sd['timestamp']
            diffs = np.abs(ts_arr_c - sd_ts)
            min_idx = np.argmin(diffs)
            s = new_nusc_samples[min_idx]
            assert nusc.get('sample', tkn)['scene_token'] == s['scene_token']
            sd['sample_token'] = s['token']

    new_nusc_sample_annos=[]
    new_nusc_instances=[]
    # Go through all instances and prune deleted sample annotations
    num_removed_instances = 0
    for inst in nusc.instance:
        sa_tkn = inst['first_annotation_token']
        sa = nusc.get('sample_annotation', sa_tkn)
        while sa_tkn != '' and sa['sample_token'] not in tokens_c:
            sa_tkn = sa['next']
            if sa_tkn != '':
                sa = nusc.get('sample_annotation', sa_tkn)
        if sa_tkn == '':
            #whoops, need to remove this instance!
            num_removed_instances += 1
            continue

        inst['first_annotation_token'] = sa_tkn
        sa['prev'] = ''
        new_nusc_sample_annos.append(sa)
        cnt = 1

        # find next and connect
        sa_tkn = sa['next']
        while sa_tkn != '':
            sa = nusc.get('sample_annotation', sa_tkn)
            while sa_tkn != '' and sa['sample_token'] not in tokens_c:
                sa_tkn = sa['next']
                if sa_tkn != '':
                    sa = nusc.get('sample_annotation', sa_tkn)
            if sa_tkn != '':
                new_nusc_sample_annos[-1]['next'] = sa_tkn
                sa['prev'] = new_nusc_sample_annos[-1]['token']
                new_nusc_sample_annos.append(sa)
                cnt += 1
                sa_tkn = sa['next']

        new_nusc_sample_annos[-1]['next'] = ''
        inst['last_annotation_token'] = new_nusc_sample_annos[-1]['token']
        inst['nbr_annotations'] = cnt
        new_nusc_instances.append(inst)

    print('Num prev instances:', len(nusc.instance))
    print('Num new instances:', len(new_nusc_instances))
    print('Num removed instances:', num_removed_instances)
    nusc.sample_annotation = new_nusc_sample_annos
    nusc.sample = new_nusc_samples
    nusc.sample_data = new_nusc_sample_datas
    nusc.instances = new_nusc_instances


def calc_scene_velos():
    global nusc

    scene_to_velos={}
    sample_to_egovel={}
    for sample in nusc.sample:
        # Get egovel
        sd_tkn = sample['data']['LIDAR_TOP']
        sample_data = nusc.get('sample_data', sd_tkn)
        ep = nusc.get('ego_pose', sample_data['ego_pose_token'])
        # timestamps are in microseconds
        ts = sample_data['timestamp']
        if sample_data['prev'] == '':
            #No prev data, calc speed w.r.t next
            next_sample_data = nusc.get('sample_data', sample_data['next'])
            next_ep = nusc.get('ego_pose', next_sample_data['ego_pose_token'])
            next_ts = next_sample_data['timestamp']
            trnsl = np.array(ep['translation'])
            next_trnsl = np.array(next_ep['translation'])
            egovel = (next_trnsl - trnsl)[:2] / ((next_ts - ts) / 1000000.)
        else:
            prev_sample_data = nusc.get('sample_data', sample_data['prev'])
            prev_ep = nusc.get('ego_pose', prev_sample_data['ego_pose_token'])
            prev_ts = prev_sample_data['timestamp']
            trnsl = np.array(ep['translation'])
            prev_trnsl = np.array(prev_ep['translation'])
            egovel = (trnsl - prev_trnsl)[:2] / ((ts - prev_ts) / 1000000.)

        st = sample['scene_token']
        if st not in scene_to_velos:
            scene_to_velos[st] = []

        for sa_tkn in sample['anns']:
            velo = nusc.box_velocity(sa_tkn)[:2]
            if np.any(np.isnan(velo)):
                continue

            # Calculate the relative velocity
            relv = np.linalg.norm(velo - egovel)
            scene_to_velos[st].append(relv)

    scene_tuples = []
    for scene_tkn, velos in scene_to_velos.items():
        scene = nusc.get('scene', scene_tkn)
        scene_tuples.append((scene['name'], sum(velos), scene['description']))
        #print(scene['name'], sum(velos), scene['description'])
    scene_tuples = sorted(scene_tuples, key=lambda x: x[1])
    for t in scene_tuples:
        print('VAL' if t[0] in val else 'TRAIN' , t[0], t[1], t[2])

def prune_training_data_from_tables():
    global nusc

    val_scenes = set(['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073'])

    calib_scenes = set(['scene-0061', 'scene-0655', 'scene-0757', 'scene-1077', 'scene-1094', 'scene-1100'])

    val_scenes.update(calib_scenes)

    keep_indexes = {nm: set() for nm in nusc.table_names}

    for scene in nusc.scene:
        if scene['name'] in val_scenes:
            keep_indexes['scene'].add(nusc.getind('scene', scene['token']))
            #keep_indexes['log'].add(nusc.getind('log', scene['log_token'])

            sample_tkn = scene['first_sample_token']
            while sample_tkn != '':
                keep_indexes['sample'].add(nusc.getind('sample', sample_tkn))
                sample = nusc.get('sample', sample_tkn)
                sample_tkn = sample['next']

    for sample_data in nusc.sample_data:
        sample_tkn = sample_data['sample_token']
        if nusc.getind('sample', sample_tkn) in keep_indexes['sample']:
            keep_indexes['sample_data'].add(nusc.getind('sample_data', sample_data['token']))
            keep_indexes['ego_pose'].add(nusc.getind('ego_pose', sample_data['ego_pose_token']))
            #keep_indexes['calibrated_sensor'].add(nusc.getind('calibrated_sensor',
            #    sample_data['calibrated_sensor_token']))

    for sample_anno in nusc.sample_annotation:
        sample_tkn = sample_anno['sample_token']
        if nusc.getind('sample', sample_tkn) in keep_indexes['sample']:
            keep_indexes['sample_annotation'].add(nusc.getind('sample_annotation',
                sample_anno['token']))
            keep_indexes['instance'].add(nusc.getind('instance',
                sample_anno['instance_token']))
    
    for k, indexes in keep_indexes.items():
        table = getattr(nusc, k)
        setattr(nusc, k, [table[i] for i in indexes])


def dump_data(dumpdir='.'):
    global nusc

    indent_num=0
    print('Dumping the tables')
    if dumpdir != '.':
        os.makedirs(dumpdir, exist_ok=True)
        curdir = os.getcwd()
        os.chdir(dumpdir)

    with open('scene.json', 'w') as handle:
        json.dump(nusc.scene, handle, indent=indent_num)
    
    for sd in nusc.sample:
        del sd['anns']
        del sd['data']
    with open('sample.json', 'w') as handle:
        json.dump(nusc.sample, handle, indent=indent_num)

    for sd in nusc.sample_data:
        del sd['sensor_modality']
        del sd['channel']
    with open('sample_data.json', 'w') as handle:
        json.dump(nusc.sample_data, handle, indent=indent_num)

    with open('ego_pose.json', 'w') as handle:
        json.dump(nusc.ego_pose, handle, indent=indent_num)

    for sd in nusc.sample_annotation:
        del sd['category_name']
    with open('sample_annotation.json', 'w') as handle:
        json.dump(nusc.sample_annotation, handle, indent=indent_num)

    with open('instance.json', 'w') as handle:
        json.dump(nusc.instance, handle, indent=indent_num)

    if dumpdir != '.':
        os.chdir(curdir)


def main():
    read_nusc()
    if len(sys.argv) == 3 and sys.argv[1] == 'populate_annos_v2':
        step = int(sys.argv[2])
        populate_annos_v2(step)
        dump_data()
    elif len(sys.argv) == 3 and sys.argv[1] == 'prune_annos':
        step = int(sys.argv[2])
        prune_annos(step)
        dump_data()
    elif len(sys.argv) == 2 and sys.argv[1] == 'generate_dicts':
        generate_anns_dict()
        generate_pose_dict()
    elif len(sys.argv) == 2 and sys.argv[1] == 'calc_velos':
        calc_scene_velos()
    elif len(sys.argv) == 2 and sys.argv[1] == 'prune_training_data_from_tables':
        prune_training_data_from_tables()
        dump_data("./pruned_tables")
    else:
        print('Usage error, doing nothing.')

if __name__ == "__main__":
    main()
