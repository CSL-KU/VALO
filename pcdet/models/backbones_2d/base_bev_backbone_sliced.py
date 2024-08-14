import numpy as np
import torch
import torch.nn as nn
import time

from ..detectors.sched_helpers import SchedAlgo

class BaseBEVBackboneSliced(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.tcount = self.model_cfg.TILE_COUNT
        self.sched_algo = self.model_cfg.METHOD

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """

        spatial_features = data_dict['spatial_features']

        ctc = data_dict['chosen_tile_coords']
        if self.sched_algo == SchedAlgo.MirrorRR:
            ctc = np.sort(ctc)
        ctc_s, ctc_e = ctc[0], ctc[-1]
        tile_sz = spatial_features.size(-1) // self.tcount
        if len(ctc) == self.tcount:
            # Select all
            x = spatial_features
            chunks = [(ctc_s, ctc_e)]
        elif (self.sched_algo == SchedAlgo.RoundRobin and ctc_s <= ctc_e) or \
                 (self.sched_algo == SchedAlgo.AdaptiveRR and ctc_s <= ctc_e) or \
                 (self.sched_algo == SchedAlgo.MirrorRR and ctc_e - ctc_s + 1 == ctc.shape[0]):
            # Contiguous
            x = spatial_features[..., (ctc_s * tile_sz):((ctc_e + 1) * tile_sz)]
            chunks = [(ctc_s, ctc_e)]
        else:
            # Two chunks, find the point of switching
            # Following piece of code take 0.6 ms in jetson agx
            i = 0
            if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
                while ctc[i] < ctc[i+1]:
                    i += 1
            elif self.sched_algo == SchedAlgo.MirrorRR:
                while ctc[i]+1 == ctc[i+1]:
                    i += 1
            chunk_r = (ctc_s, ctc[i])
            chunk_l = (ctc[i+1], ctc_e)
            c_sz_l = (chunk_l[1] - chunk_l[0] + 1) * tile_sz
            c_sz_r = (chunk_r[1] - chunk_r[0] + 1) * tile_sz
            c_sz = list(spatial_features.size())
            c_sz[-1] = c_sz_r + c_sz_l
            x = torch.empty(c_sz, device='cuda', dtype=spatial_features.dtype)
            #Example: . . 2 3 4 . . 7 8 -> 7 8 2 3 4
            x[..., :c_sz_r] = spatial_features[..., \
                    (chunk_r[0]*tile_sz):((chunk_r[1]+1)*tile_sz)]
            x[..., -c_sz_l:] = spatial_features[..., \
                    (chunk_l[0]*tile_sz):((chunk_l[1]+1)*tile_sz)]
            chunks = [chunk_r, chunk_l]
        data_dict['tile_chunks'] = chunks

        ups = []
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            #stride = int(spatial_features.shape[2] / x.shape[2])
            #ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        # Don't scatter the results now, let detection head process the reduced
        # input as well.

        return data_dict
