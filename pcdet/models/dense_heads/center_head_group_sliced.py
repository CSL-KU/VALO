import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from ...ops.cuda_slicer import cuda_slicer
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
from ..detectors.sched_helpers import SchedAlgo

import time

# Divides input channels equally to convolutions
# Produces some useless output channels for the sake of efficiency
class AdaptiveGroupConv(nn.Module):
    def __init__(self, input_channels, output_channels_list, ksize,
            stride, padding, bias):
        super().__init__()

        self.outp_ch_l = output_channels_list
        max_ch = max(self.outp_ch_l)
        self.num_outp = len(self.outp_ch_l)
        self.inp_ch_per_conv = input_channels // self.num_outp
        self.conv = nn.Conv2d(input_channels, max_ch * self.num_outp,
                kernel_size=ksize,stride=stride,
                padding=padding, groups=self.num_outp, bias=bias)

    def fill_bias(self, bias):
        m = self.conv
        if hasattr(self.conv, "bias") and m.bias is not None:
            m.bias.data.fill_(bias)

    def init_kaiming(self):
        m = self.conv
        kaiming_normal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        outp = [None] * self.num_outp
        max_ch = max(self.outp_ch_l)

        y = self.conv(inp)

        for i, ch in enumerate(self.outp_ch_l):
            outp[i] = y[:,(i*max_ch):(i*max_ch+ch)]

        return outp

class CenterHeadGroupSliced(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.tcount=self.model_cfg.TILE_COUNT # There should be a better way but its fine for now
        self.sched_algo = self.model_cfg.METHOD
        post_process_cfg = self.model_cfg.POST_PROCESSING
        self.max_obj_per_sample = post_process_cfg.MAX_OBJ_PER_SAMPLE

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        ksize=3

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        num_heads = len(self.class_names_each_head)
        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), \
                f'class_names_each_head={self.class_names_each_head}'
        use_bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3,
                stride=1, padding=1, bias=use_bias
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        hm_list = []
        inp_channels = self.model_cfg.SHARED_CONV_CHANNEL
        outp_channels, groups = inp_channels * num_heads, 1
        for k in range(self.model_cfg.NUM_HM_CONV - 1):
            hm_list.append(nn.Conv2d(inp_channels, outp_channels, kernel_size=ksize,
                    stride=1, padding=1, groups=groups, bias=use_bias))
            if num_heads > 1:
                hm_list.append(nn.GroupNorm(num_heads, outp_channels))
            else:
                hm_list.append(nn.BatchNorm2d(outp_channels))
            hm_list.append(nn.ReLU())
            if k == 0:
                inp_channels = outp_channels
                groups = num_heads

        if self.model_cfg.NUM_HM_CONV <= 1:
            outp_channels = inp_channels

        hm_outp_channels = [len(cur_class_names) \
                for cur_class_names in self.class_names_each_head]

        hm_list.append(AdaptiveGroupConv(outp_channels, hm_outp_channels, \
                ksize, stride=1, padding=1, bias=True))

        hm_list[-1].fill_bias(-2.19)

        if len(hm_list) > 1:
            self.heatmap_convs = nn.Sequential(*hm_list)
        else:
            self.heatmap_convs = hm_list[0]

        # Now, we need seperate merged conv and group conv for each head because their inputs
        # will be different.
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        head_dict = self.separate_head_cfg.HEAD_DICT
        num_convs = [v['num_conv'] for v in head_dict.values()]
        assert all([num_convs[0] == nc for nc in num_convs])
        num_convs = num_convs[0]

        attr_outp_channels = [v['out_channels'] for v in head_dict.values()]

        self.det_heads=nn.ModuleList()
        self.attr_conv_names = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER
        self.slice_size = 1
        for head_idx in range(num_heads):
            inp_channels = self.model_cfg.SHARED_CONV_CHANNEL
            outp_channels, groups = inp_channels * len(head_dict), 1
            attr_list = []

            for k in range(num_convs - 1):
                attr_list.append(nn.Conv2d(inp_channels, outp_channels, kernel_size=ksize,
                        stride=1, padding=0, groups=groups, bias=use_bias))
                # NOTE I am not using batch normalization here as it is causing harm
                # to the slicing as the batch sizes are different in training and
                # evaluation. In training, it is full size tensor, in eval, it is
                # batch of slices.
                attr_list.append(nn.ReLU())
                if k == 0:
                    inp_channels = outp_channels
                    groups = len(head_dict)
                if head_idx == 0:
                    self.slice_size = (self.slice_size-1) + ksize

            if num_convs <= 1:
                outp_channels = inp_channels

            attr_list.append(AdaptiveGroupConv(outp_channels, attr_outp_channels, \
                    ksize, stride=1, padding=0, bias=True))

            attr_convs = nn.Sequential(*attr_list)
            if head_idx == 0:
                self.slice_size = (self.slice_size-1) + ksize

            for m in attr_convs:
                if isinstance(m, nn.Conv2d):
                    kaiming_normal_(m.weight.data)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            attr_convs[-1].init_kaiming()

            self.det_heads.append(attr_convs)
        self.calibrated = [False] * num_heads

        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

        self.det_dict_copy = {
            "pred_boxes": torch.zeros([0, 9], dtype=torch.float, device='cuda'),
            "pred_scores": torch.zeros([0], dtype=torch.float, device='cuda'),
            "pred_labels": torch.zeros([0], dtype=torch.int, device='cuda'),
        }

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    # give topk_outps to this guy if it is available!
    def generate_predicted_boxes_eval(self, batch_size, pred_dicts, projections, do_nms=True):
        assert batch_size == 1
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = {
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } # for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            # this loop runs only once for kitti but multiple times for nuscenes (single vs multihead)
            cls_id_map = self.class_id_mapping_each_head[idx]
            if 'center' not in pred_dict:
                # no dets from this dethead, use projections or empty result instead
                if do_nms and projections is not None:
                    # Add the projections if they exist, no need do NMS
                    for k in projections[idx].keys():
                        ret_dict[k].append(projections[idx][k])
                continue

            vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap_sliced(
                heatmap=pred_dict['hm'],
                rot_cos=pred_dict['rot'][..., 0].unsqueeze(dim=-1),
                rot_sin=pred_dict['rot'][..., 1].unsqueeze(dim=-1),
                center=pred_dict['center'], center_z=pred_dict['center_z'],
                dim=pred_dict['dim'].exp(), vel=vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range,
                topk_outp=pred_dict['topk_outp']
            )

            # Assume batch size is 1
            final_dict = final_pred_dicts[0]
            final_dict['pred_labels'] = cls_id_map[final_dict['pred_labels'].long()]

            if do_nms and post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                if projections is not None:
                    # get the projections that match and cat them for NMS
                    for k in projections[idx].keys():
                        final_dict[k] = torch.cat((final_dict[k], projections[idx][k]), dim=0)
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=None
                )

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

            ret_dict['pred_boxes'].append(final_dict['pred_boxes'])
            ret_dict['pred_scores'].append(final_dict['pred_scores'])
            ret_dict['pred_labels'].append(final_dict['pred_labels'])

        #for k in range(batch_size):
        if not ret_dict['pred_boxes']:
            ret_dict = self.get_empty_det_dict()
        else:
            ret_dict['pred_boxes'] = torch.cat(ret_dict['pred_boxes'], dim=0)
            ret_dict['pred_scores'] = torch.cat(ret_dict['pred_scores'], dim=0)
            ret_dict['pred_labels'] = torch.cat(ret_dict['pred_labels'], dim=0) + 1
        return [ret_dict]


    def nms_after_gen(self, batch_dict):
        assert batch_dict['batch_size'] == 1
        post_process_cfg = self.model_cfg.POST_PROCESSING

        ret_dict = {
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } # for k in range(batch_size)]
        for idx, pred_dict in enumerate(batch_dict['projections_nms']):
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=pred_dict['pred_scores'], box_preds=pred_dict['pred_boxes'],
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=None
            )

            ret_dict['pred_boxes'].append(pred_dict['pred_boxes'][selected])
            ret_dict['pred_scores'].append(selected_scores)
            ret_dict['pred_labels'].append(pred_dict['pred_labels'][selected])

        ret_dict['pred_boxes'] = torch.cat(ret_dict['pred_boxes'], dim=0)
        ret_dict['pred_scores'] = torch.cat(ret_dict['pred_scores'], dim=0)
        ret_dict['pred_labels'] = torch.cat(ret_dict['pred_labels'], dim=0) + 1

        batch_dict['final_box_dicts'][0] = ret_dict

        return batch_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def scatter_sliced_tensors(self, chosen_tile_coords, sliced_tensors):
        #Based on chosen_tile_coords, we need to scatter the output
        ctc = chosen_tile_coords
        if self.sched_algo == SchedAlgo.MirrorRR:
            ctc = np.sort(chosen_tile_coords)
        scattered_tensors = []
        ctc_s, ctc_e = ctc[0], ctc[-1]
        if (self.sched_algo == SchedAlgo.RoundRobin and ctc_s <= ctc_e) or \
                (self.sched_algo == SchedAlgo.AdaptiveRR and ctc_s <= ctc_e) or \
                (self.sched_algo == SchedAlgo.MirrorRR and ctc_e - ctc_s + 1 == ctc.shape[0]):
            # contiguous
            num_tiles = ctc_e - ctc_s + 1
        else:
            # Two chunks, find the point of switching
            i = 0
            if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
                while ctc[i] < ctc[i+1]:
                    i += 1
            elif self.sched_algo == SchedAlgo.MirrorRR:
                while ctc[i]+1 == ctc[i+1]:
                    i += 1
            chunk_r = (ctc_s, ctc[i])
            chunk_l = (ctc[i+1], ctc_e)
            num_tiles = (chunk_r[1] - chunk_r[0] + 1) + (chunk_l[1] - chunk_l[0] + 1)

        if len(ctc) == self.tcount:
            return sliced_tensors # no need to scatter

        for tensor in sliced_tensors:
            if (self.sched_algo == SchedAlgo.RoundRobin and ctc_s <= ctc_e) or \
                    (self.sched_algo == SchedAlgo.AdaptiveRR and ctc_s <= ctc_e) or \
                    (self.sched_algo == SchedAlgo.MirrorRR and ctc_e - ctc_s + 1 == ctc.shape[0]):
                # contiguous
                tile_sz = tensor.size(-1) // num_tiles
                full_sz = tile_sz * self.tcount
                tensor_sz = list(tensor.size())
                tensor_sz[-1] = full_sz
                scat_tensor = torch.zeros(tensor_sz, device='cuda', dtype=tensor.dtype)
                scat_tensor[..., (ctc_s * tile_sz):((ctc_e + 1) * tile_sz)] = tensor
            else:
                # Two chunks, find the point of switching
                tile_sz = tensor.size(-1) // num_tiles
                full_sz = tile_sz * self.tcount
                tensor_sz = list(tensor.size())
                tensor_sz[-1] = full_sz
                scat_tensor = torch.zeros(tensor_sz, device='cuda', dtype=tensor.dtype)
                c_sz_l = (chunk_l[1] - chunk_l[0] + 1) * tile_sz
                c_sz_r = (chunk_r[1] - chunk_r[0] + 1) * tile_sz
                #Example: 7 8 2 3 4  -> . . 2 3 4 . . 7 8
                scat_tensor[..., (chunk_r[0]*tile_sz):((chunk_r[1]+1)*tile_sz)] = \
                        tensor[..., :c_sz_r]
                scat_tensor[..., (chunk_l[0]*tile_sz):((chunk_l[1]+1)*tile_sz)] = \
                        tensor[..., -c_sz_l:]
            scattered_tensors.append(scat_tensor)
        return scattered_tensors

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_eval(data_dict)

    def forward_train(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        shr_conv_outp = self.shared_conv(spatial_features_2d)

        heatmaps = self.heatmap_convs(shr_conv_outp)
        heatmaps = self.scatter_sliced_tensors(data_dict['chosen_tile_coords'], heatmaps)

        pred_dicts = [{'hm' : hm} for hm in heatmaps]
        # default padding is 1
        pad_size = p = 1

        # Forward through all heads one by one
        for det_head, pd in zip(self.det_heads, pred_dicts):
            x = shr_conv_outp
            for m in det_head:
                if isinstance(m, nn.Conv2d) or isinstance(m, AdaptiveGroupConv):
                    x = torch.nn.functional.pad(x, (p,p,p,p))
                x = m(x)

            x = self.scatter_sliced_tensors(data_dict['chosen_tile_coords'], x)
            for name, attr in zip(self.attr_conv_names, x):
                pd[name] = attr

        feature_map_size=heatmaps[0].size()[2:]
        target_dict = self.assign_targets(
            data_dict['gt_boxes'], feature_map_size=feature_map_size,
            feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
        )
        self.forward_ret_dict['target_dicts'] = target_dict
        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )
            rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
            data_dict['rois'] = rois
            data_dict['roi_scores'] = roi_scores
            data_dict['roi_labels'] = roi_labels
            data_dict['has_class_labels'] = True

        if not self.training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )
            data_dict['final_box_dicts'] = pred_dicts

        return data_dict

    def forward_eval(self, data_dict):
        data_dict = self.forward_eval_topk(self.forward_eval_pre(data_dict))
        data_dict = self.forward_eval_post(data_dict)

        pred_dicts = self.generate_predicted_boxes_eval(
            data_dict['batch_size'], data_dict['pred_dicts'], data_dict['projections_nms']
        )

        data_dict['final_box_dicts'] = pred_dicts

        return data_dict

    def forward_eval_conv(self, data_dict):
        assert data_dict['batch_size'] == 1
        x = data_dict['spatial_features_2d']

        shr_conv_outp = self.shared_conv(x)
        data_dict['shr_conv_outp'] = shr_conv_outp

        # Run heatmap convolutions and gather the actual channels
        heatmaps = self.heatmap_convs(shr_conv_outp)
        heatmaps = [self.sigmoid(hm) for hm in heatmaps]
        heatmaps = self.scatter_sliced_tensors(data_dict['chosen_tile_coords'], heatmaps)
        data_dict['pred_dicts'] = [{'hm' : hm} for hm in heatmaps]

        return data_dict

    def forward_eval_topk(self, data_dict):
        pred_dicts = data_dict['pred_dicts']

        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thres = post_process_cfg.SCORE_THRESH

        for pd in pred_dicts:
            topk_score, topk_inds, topk_classes, topk_ys, topk_xs = \
                    centernet_utils._topk(pd['hm'], K=self.max_obj_per_sample, \
                    using_slicing=True)
            # stack all of these and then apply the mask
            pd['topk_outp'] = torch.stack((topk_score[0], topk_classes[0], topk_ys[0], topk_xs[0]))
            pd['score_mask'] = topk_score[0] > score_thres

        nonempty_det_head_indexes = []
        for idx, pd in enumerate(pred_dicts):
            masked_topk_vals = pd['topk_outp'][:, pd['score_mask']]
            tensors = torch.chunk(masked_topk_vals, 4)
            pd['topk_outp'] = [t.flatten() for t in tensors]
            if pd['topk_outp'][0].size(0) > 0:
                nonempty_det_head_indexes.append(idx)

        # Code is synched here due to masks
        data_dict['pred_dicts'] = pred_dicts
        data_dict['dethead_indexes'] = np.array(nonempty_det_head_indexes)

        return data_dict

    def forward_eval_post(self, data_dict):


        p = pad_size = self.slice_size//2
        shr_conv_outp = self.scatter_sliced_tensors(data_dict['chosen_tile_coords'], \
                [data_dict['shr_conv_outp']])
        shr_conv_outp = shr_conv_outp[0]
        shr_conv_outp_nhwc = shr_conv_outp.permute(0,2,3,1).contiguous()
        padded_x = torch.nn.functional.pad(shr_conv_outp_nhwc, (0,0,p,p,p,p))



        slc_indices, x_inds, y_inds, i = [], [], [], 0
        for pd in data_dict['pred_dicts']:
            topk_outp = pd['topk_outp']
            scores = topk_outp[0]
            num_slc = scores.size(0)
            slc_indices.append((i, i+num_slc))
            i += num_slc
            if num_slc > 0:
                # thanks to padding, xs and ys now give us the corner position instead of center
                # which is required by the slice and batch kernel
                x_inds.append(topk_outp[2])
                y_inds.append(topk_outp[3])

        if i > 0:
            b_id = torch.full((i,), 0, dtype=torch.short, device='cuda') # since batch size is 1
            indices = torch.stack((b_id, torch.cat(x_inds).short(), torch.cat(y_inds).short()), dim=1)
            all_slices = cuda_slicer.slice_and_batch_nhwc(padded_x, indices, self.slice_size)

        for det_idx, (det_head, pd) in enumerate(zip(self.det_heads, data_dict['pred_dicts'])):
            slc_i_1, slc_i_2 = slc_indices[det_idx]
            if slc_i_1 == slc_i_2: # no slice
                continue

            if not self.calibrated[det_idx] and torch.backends.cudnn.benchmark:
                self.calibrate_for_cudnn_benchmarking(all_slices[slc_i_1:slc_i_2], det_idx)

            # It becomes deterministic with cudnn benchmarking enabled, ~1ms
            outp = det_head(all_slices[slc_i_1:slc_i_2])

            # finally, split the output according to the batches they belong
            for name, attr in zip(self.attr_conv_names, outp):
                pd[name] = attr.flatten(-3)


#        pred_dicts = self.generate_predicted_boxes_eval(
#            data_dict['batch_size'], data_dict['pred_dicts'], data_dict['projections_nms']
#        )
#
#        data_dict['final_box_dicts'] = pred_dicts

        return data_dict

    def get_empty_det_dict(self):
        det_dict = {}
        for k,v in self.det_dict_copy.items():
            det_dict[k] = v.clone().detach()
        return det_dict

    def calibrate_for_cudnn_benchmarking(self, inp, idx):
        max_batch_size = self.max_obj_per_sample
        print(f'Calibrating detection head {idx+1} for cudnn benchmarking,',
                ' max batch size:', max_batch_size, ' ...')

        N, C, H, W = inp.size()
        dummy_inp = torch.rand((max_batch_size, C, H, W), dtype=inp.dtype, device=inp.device)
        for n in range(1, max_batch_size+1):
            x = dummy_inp[:n, ...]
            self.det_heads[idx](x)

        self.calibrated[idx]=True
        print('done.')
