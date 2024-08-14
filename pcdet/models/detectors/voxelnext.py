import torch
from .detector3d_template import Detector3DTemplate

class VoxelNeXt(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        torch.backends.cudnn.benchmark = True
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()

        self.vfe, self.backbone_3d, self.dense_head = self.module_list
        self.update_time_dict( {
                'VFE': [],
                'Backbone3D':[],
                'VoxelHead-conv': [],
                'VoxelHead-post': [],
                })

    def forward(self, batch_dict):
        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict)
        self.measure_time_end('VFE')

        self.measure_time_start('Backbone3D')
        batch_dict = self.backbone_3d(batch_dict)
        self.measure_time_end('Backbone3D')

        self.measure_time_start('VoxelHead-conv')
        batch_dict = self.dense_head.forward_conv(batch_dict)
        self.measure_time_end('VoxelHead-conv')
        self.measure_time_start('VoxelHead-post')
        batch_dict = self.dense_head.forward_post(batch_dict)
        self.measure_time_end('VoxelHead-post')

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            #pred_dicts, recall_dicts = self.post_processing(batch_dict)
            #return pred_dicts, recall_dicts
            return batch_dict

    def get_training_loss(self):
        
        disp_dict = {}
        loss, tb_dict = self.dense_head.get_loss()
        
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
                box_preds=pred_boxes.cuda(),
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
    
    def calibrate(self, batch_size=1):
        return super().calibrate(1)
