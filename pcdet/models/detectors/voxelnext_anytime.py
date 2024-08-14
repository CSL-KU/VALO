import torch
from .anytime_template_v2 import AnytimeTemplateV2

class VoxelNeXtAnytime(AnytimeTemplateV2):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.vfe, self.backbone_3d, self.dense_head = self.module_list
        self.update_time_dict( {
                'VFE': [],
                'Sched1': [],
                'Sched2': [],
                'Backbone3D':[],
                'VoxelHead-conv-hm': [],
                'VoxelHead-conv-rest': [],
                'VoxelHead-post': [],
                })


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
        self.measure_time_start('Backbone3D')
        batch_dict = self.backbone_3d(batch_dict)
        self.measure_time_end('Backbone3D')

        self.measure_time_start('VoxelHead-conv-hm')
        batch_dict = self.dense_head.forward_conv_hm(batch_dict)
        self.measure_time_end('VoxelHead-conv-hm')

        if self.is_calibrating() or 'bb3d_layer_time_events' in batch_dict:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            batch_dict['bb3d_layer_time_events'].append(e1)

        self.measure_time_start('Sched2')
        batch_dict = self.schedule2(batch_dict)
        if self.enable_projection:
            batch_dict = self.schedule3(batch_dict) # run projections in parallel with dethead ?
        self.measure_time_end('Sched2')

        self.measure_time_start('VoxelHead-conv-rest')
        batch_dict = self.dense_head.forward_conv_rest(batch_dict)
        self.measure_time_end('VoxelHead-conv-rest')

        self.measure_time_start('VoxelHead-post')
        batch_dict = self.dense_head.forward_post(batch_dict)
        self.measure_time_end('VoxelHead-post')

        if self.is_calibrating():
            e2 = torch.cuda.Event(enable_timing=True)
            e2.record()
            batch_dict['detheadpost_time_events'] = [e1, e2]

        return batch_dict

    def forward_train(self, batch_dict):
        batch_dict = self.vfe(batch_dict, model=self)
        batch_dict = self.schedule1(batch_dict)
        if self.is_voxel_enc:
            batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    #def calibrate(self):
    #    super().calibrate()
    #    return None
