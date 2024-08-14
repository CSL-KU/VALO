import torch
from .detector3d_template import Detector3DTemplate

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        torch.backends.cudnn.benchmark = True
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()

        self.vfe, self.map_to_bev, self.backbone_2d, \
                self.dense_head = self.module_list
        self.update_time_dict( {
                'VFE': [],
                'Backbone3D': [],
                'MapToBEV': [],
                'Backbone2D': [],
                'DetectionHead': [],})

    def forward(self, batch_dict):
        self.measure_time_start('VFE')
        batch_dict = self.vfe.forward_gen_pillars(batch_dict)
        self.measure_time_end('VFE')
        self.measure_time_start('Backbone3D')
        batch_dict = self.vfe.forward_nn(batch_dict)
        self.measure_time_end('Backbone3D')
        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.measure_time_end('MapToBEV')
        self.measure_time_start('Backbone2D')
        batch_dict = self.backbone_2d(batch_dict)
        self.measure_time_end('Backbone2D')
        self.measure_time_start('DetectionHead')
        batch_dict = self.dense_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            return batch_dict
            #pred_dicts, recall_dicts = self.post_processing(batch_dict)
            #return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
    
    def calibrate(self, batch_size=1):
        return super().calibrate(1)
