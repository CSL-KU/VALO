from .detector3d_template import Detector3DTemplate
import torch
import numpy as np
import numba

@numba.jit(nopython=True)
def netc_map(tcount, netc):
    mapping = np.zeros(tcount, dtype=np.int64)
    for i in range(netc.shape[0]):
        mapping[netc[i]] = i
    return mapping


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        torch.backends.cudnn.benchmark = True
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()


        if self.model_cfg.get('BACKBONE_3D', None) is None:
            #pillar
            self.is_voxel_enc=False
            self.vfe, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead-Pre': [],
                    'CenterHead-Post': [],
                    'CenterHead-Topk': [],
                    'CenterHead-GenBox': [],
                    'CenterHead': [],})
        else:
            #voxel
            self.is_voxel_enc=True
            self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'Backbone3D':[],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead-Pre': [],
                    'CenterHead-Post': [],
                    'CenterHead-Topk': [],
                    'CenterHead-GenBox': [],
                    'CenterHead': [],})

        self.do_frag_test = False
        if self.do_frag_test:
            self.tcount = 4 # for voxel01
            self.tile_size_voxels = torch.tensor(\
                    self.dataset.grid_size[1] / self.tcount).long().item()

            #self.backbone_3d.sparse_shape[-1] /= self.tcount

    def forward(self, batch_dict):
        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict)
        self.measure_time_end('VFE')

        if self.do_frag_test:
            voxel_coords = batch_dict['voxel_coords']
            batch_dict['voxel_coords'], netc = self.batchify(voxel_coords)
            # sorting might not be necessary but do it anyway
            indexes = torch.argsort(batch_dict['voxel_coords'][:, 0])
            for k in ('voxel_coords', 'voxel_features'):
                batch_dict[k] = batch_dict[k][indexes]
            batch_dict['batch_size'] = netc.shape[0]

        if self.is_voxel_enc:
            self.measure_time_start('Backbone3D')
            batch_dict = self.backbone_3d(batch_dict)
            self.measure_time_end('Backbone3D')

            #est = batch_dict['encoded_spconv_tensor']
            #est.spatial_shape[-1] //= self.tcount
            #print(est.spatial_shape)

        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.measure_time_end('MapToBEV')


        if self.do_frag_test:
            sf = batch_dict['spatial_features']
            #print('spatial features', sf.size())
            slc_sz = sf.size(3) // self.tcount
            scat = torch.zeros((sf.size(0), sf.size(1), sf.size(2), slc_sz),
                    dtype=sf.dtype, device=sf.device)
            #print('scat', scat.size())
            for i, t in enumerate(sf):
                #print('t slice', (netc[i] * slc_sz), ((netc[i]+1) * slc_sz))
                scat[i, ..., :]  = t[..., (netc[i] * slc_sz):((netc[i]+1) * slc_sz)]
            batch_dict['spatial_features'] = scat

        self.measure_time_start('Backbone2D')
        batch_dict = self.backbone_2d(batch_dict)
        self.measure_time_end('Backbone2D')
        self.measure_time_start('CenterHead')
        self.measure_time_start('CenterHead-Pre')
        batch_dict = self.dense_head.forward_pre(batch_dict)
        self.measure_time_end('CenterHead-Pre')
        self.measure_time_start('CenterHead-Post')
        batch_dict = self.dense_head.forward_post(batch_dict)
        self.measure_time_end('CenterHead-Post')

        if self.do_frag_test:
            #I need to scatter the tensors back using netc as maping
            for pd in batch_dict['pred_dicts']:
                for k,v in pd.items():
                    #print(k, v.size())
                    slc_sz = v.size(3)
                    scat = torch.zeros((1, v.size(1), v.size(2), slc_sz * self.tcount),
                            dtype=v.dtype, device=v.device)
                    #print('scat2', scat.size())
                    for i, t in enumerate(v):
                        scat[..., (netc[i] * slc_sz):((netc[i]+1) * slc_sz)] = t
                    pd[k] = scat
            batch_dict['batch_size'] = 1

        self.measure_time_start('CenterHead-Topk')
        batch_dict = self.dense_head.forward_topk(batch_dict)
        self.measure_time_end('CenterHead-Topk')
        self.measure_time_start('CenterHead-GenBox')
        batch_dict = self.dense_head.forward_genbox(batch_dict)
        self.measure_time_end('CenterHead-GenBox')
        self.measure_time_end('CenterHead')

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # let the hooks of parent class handle this
            #pred_dicts, recall_dicts = self.post_processing(batch_dict)
            #return pred_dicts, recall_dicts
            return batch_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
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

    def batchify(self, voxel_coords):
        # Calculate where each voxel resides in which tile

        voxel_tile_coords = torch.div(voxel_coords[:, -1], self.tile_size_voxels, \
                rounding_mode='trunc').long()

        nonempty_tile_coords = torch.unique(voxel_tile_coords, \
                sorted=True, return_counts=False)
        netc = nonempty_tile_coords.cpu().numpy()
        #print('netc   ', netc)
        mapping = torch.from_numpy(netc_map(self.tcount, netc)).cuda()
        #print('mapping', mapping)
        batch_indexes = mapping[voxel_tile_coords]
        #print('batch_indexes', batch_indexes)
        voxel_coords[:, 0] = batch_indexes

        return voxel_coords, netc
