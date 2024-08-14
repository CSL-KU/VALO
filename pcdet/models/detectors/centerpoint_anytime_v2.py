from .anytime_template_v2 import AnytimeTemplateV2

import torch

class CenterPointAnytimeV2(AnytimeTemplateV2):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        if self.model_cfg.get('BACKBONE_3D', None) is None:
            #pillar
            self.is_voxel_enc=False
            self.vfe, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'Sched1': [],
                    'Sched2': [],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead-Pre': [],
                    'CenterHead-Topk': [],
                    'CenterHead-Post': [],
                    'CenterHead-GenBox': [],
                    'CenterHead': []})
        else:
            #voxel
            self.is_voxel_enc=True
            self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'Sched1': [],
                    'Backbone3D':[],
                    'Sched2': [],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead-Pre': [],
                    'CenterHead-Topk': [],
                    'CenterHead-Post': [],
                    'CenterHead-GenBox': [],
                    'CenterHead': []})
        self.cudnn_calibrated = False

    def forward(self, batch_dict):
        # We are going to do projection earlier so the
        # dense head can use its results for NMS
        if self.training:
            return self.forward_train(batch_dict)
        else:
            return self.forward_eval(batch_dict)

    def forward_eval(self, batch_dict):
        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict, model=self)
        self.measure_time_end('VFE')
        self.measure_time_start('Sched1')
        batch_dict = self.schedule1(batch_dict)
        self.measure_time_end('Sched1')
        if self.is_voxel_enc:
            self.measure_time_start('Backbone3D')
            batch_dict = self.backbone_3d(batch_dict)
            self.measure_time_end('Backbone3D')

        if self.is_calibrating():
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()

        self.measure_time_start('Sched2')
        batch_dict = self.schedule2(batch_dict)
        self.measure_time_end('Sched2')

        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.measure_time_end('MapToBEV')

        if not self.cudnn_calibrated and torch.backends.cudnn.benchmark:
            self.calibrate_for_cudnn_benchmarking(batch_dict)

        self.measure_time_start('Backbone2D')
        batch_dict = self.backbone_2d(batch_dict)

        streaming_eval = self.model_cfg.STREAMING_EVAL
        if not streaming_eval and self.enable_projection:
            batch_dict = self.schedule3(batch_dict) # run in parallel with bb2d and dethead
        self.measure_time_end('Backbone2D')
        self.measure_time_start('CenterHead')
        self.measure_time_start('CenterHead-Pre')
        batch_dict = self.dense_head.forward_eval_conv(batch_dict)
        self.measure_time_end('CenterHead-Pre')
        self.measure_time_start('CenterHead-Topk')
        batch_dict = self.dense_head.forward_eval_topk(batch_dict)
        self.measure_time_end('CenterHead-Topk')

        if self.is_calibrating():
            e2 = torch.cuda.Event(enable_timing=True)
            e2.record()
            batch_dict['bb2d_time_events'] = [e1, e2]

        self.measure_time_start('CenterHead-Post')
        batch_dict = self.dense_head.forward_eval_post(batch_dict)
        self.measure_time_end('CenterHead-Post')
        self.measure_time_start('CenterHead-GenBox')
        pred_dicts = self.dense_head.generate_predicted_boxes_eval(
            batch_dict['batch_size'], batch_dict['pred_dicts'], batch_dict.get('projections_nms', None),
            do_nms=(not streaming_eval)
        )
        batch_dict['final_box_dicts'] = pred_dicts
        if streaming_eval and self.enable_projection:
            batch_dict = self.schedule3(batch_dict) # Project ALL
            batch_dict = self.dense_head.nms_after_gen(batch_dict)
        self.measure_time_end('CenterHead-GenBox')
        self.measure_time_end('CenterHead')

        if self.is_calibrating():
            e3 = torch.cuda.Event(enable_timing=True)
            e3.record()
            batch_dict['detheadpost_time_events'] = [e2, e3]

        return batch_dict

    def forward_train(self, batch_dict):
        batch_dict = self.vfe(batch_dict, model=self)
        batch_dict = self.schedule1(batch_dict)
        if self.is_voxel_enc:
            batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict


    def calibrate_for_cudnn_benchmarking(self, batch_dict):
        print('Calibrating bb2d and det head pre for cudnn benchmarking, max num tiles is',
                self.tcount, ' ...')
        # Try out all different chosen tile sizes
        dummy_dict = {'batch_size':1, 'spatial_features': batch_dict['spatial_features']}
        for i in range(1, self.tcount+1):
            dummy_dict['chosen_tile_coords'] = torch.arange(i)
            dummy_dict = self.backbone_2d(dummy_dict)
            self.dense_head.forward_eval_conv(dummy_dict)
        print('done.')
        self.cudnn_calibrated = True
