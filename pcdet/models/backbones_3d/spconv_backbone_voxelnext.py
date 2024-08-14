from functools import partial
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelResBackBone8xVoxelNeXt(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        
        self.conv6 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

        self.num_layer_groups = 8

    def get_inds_dividers(self, tile_size_voxels):
        # numbers here are determined with respect to strides
        return [tile_size_voxels / float(s) for s in (2,4,8,8,8,8,8)]

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def print_min_max(self, tensor):
        tensor = tensor.cpu()
        min_x = torch.min(tensor[:,-1]).item()
        max_x = torch.max(tensor[:,-1]).item()
        min_y = torch.min(tensor[:,-2]).item()
        max_y = torch.max(tensor[:,-2]).item()
        print(min_x, max_x, min_y, max_y)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        record_time = batch_dict.get('record_time', False)
        record_vcounts = batch_dict.get('record_int_vcounts', False)
        # int: intermediary
        record_int_vcoords = batch_dict.get('record_int_vcoords', False)
        record_int_indices = batch_dict.get('record_int_indices', False)

        ## BLOCK 1
        if record_time:
            events=[torch.cuda.Event(enable_timing=True)]
            events[-1].record()
        if record_vcounts:
            num_voxels=[voxel_coords.size(0)]
        if record_int_vcoords:
            vcoords = []
            num_tiles = batch_dict['num_tiles']
            dividers = self.get_inds_dividers(batch_dict['tile_size_voxels'])
        if record_int_indices:
            vinds = []

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2_0 = self.conv2[0][0](x_conv1)
        #x_conv2 = self.conv2(x_conv1)
        #x_conv3 = self.conv3(x_conv2)
        #x_conv4 = self.conv4(x_conv3)
        #x_conv5 = self.conv5(x_conv4)
        #x_conv6 = self.conv6(x_conv5)

        ## BLOCK 2
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv2_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv2_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv2_0.indices[:, -1], dividers[0], \
                    rounding_mode='trunc').int()
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)
            vcoords.append(voxel_dist.unsqueeze(0))

        # since next to ops are not spconv op
        x_conv2_0 = x_conv2_0.replace_feature(self.conv2[0][1](x_conv2_0.features))
        x_conv2_0 = x_conv2_0.replace_feature(self.conv2[0][2](x_conv2_0.features))
        x_conv2_1 = self.conv2[1](x_conv2_0)
        x_conv2 = self.conv2[2](x_conv2_1)
        x_conv3_0 = self.conv3[0][0](x_conv2)

        ## BLOCK 3
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv3_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv3_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv3_0.indices[:, -1], dividers[1], \
                    rounding_mode='trunc').int()
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv3_0 = x_conv3_0.replace_feature(self.conv3[0][1](x_conv3_0.features))
        x_conv3_0 = x_conv3_0.replace_feature(self.conv3[0][2](x_conv3_0.features))
        x_conv3_1 = self.conv3[1](x_conv3_0)
        x_conv3 = self.conv3[2](x_conv3_1)
        x_conv4_0 = self.conv4[0][0](x_conv3)

        ## BLOCK 4
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv4_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv4_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv4_0.indices[:, -1], dividers[2], \
                    rounding_mode='trunc').int()
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv4_0 = x_conv4_0.replace_feature(self.conv4[0][1](x_conv4_0.features))
        x_conv4_0 = x_conv4_0.replace_feature(self.conv4[0][2](x_conv4_0.features))
        x_conv4_1 = self.conv4[1](x_conv4_0)
        x_conv4 = self.conv4[2](x_conv4_1)
        x_conv5_0 = self.conv5[0][0](x_conv4)

        ## BLOCK 5
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv5_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv5_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv5_0.indices[:, -1], dividers[3], \
                    rounding_mode='trunc').int()
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv5_0 = x_conv5_0.replace_feature(self.conv5[0][1](x_conv5_0.features))
        x_conv5_0 = x_conv5_0.replace_feature(self.conv5[0][2](x_conv5_0.features))
        x_conv5_1 = self.conv5[1](x_conv5_0)
        x_conv5 = self.conv5[2](x_conv5_1)
        x_conv6_0 = self.conv6[0][0](x_conv5)

        ## BLOCK 6
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv6_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv6_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv6_0.indices[:, -1], dividers[4], \
                    rounding_mode='trunc').int()
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv6_0 = x_conv6_0.replace_feature(self.conv6[0][1](x_conv6_0.features))
        x_conv6_0 = x_conv6_0.replace_feature(self.conv6[0][2](x_conv6_0.features))
        x_conv6_1 = self.conv6[1](x_conv6_0)
        x_conv6 = self.conv6[2](x_conv6_1)

        # the ops here should be fast, we don't need to create a new block for them
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        ## BLOCK 7
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv4.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv4.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv4.indices[:, -1], dividers[5], \
                    rounding_mode='trunc').int()
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        out = self.bev_out(x_conv4)

        ## BLOCK 8
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(out.indices.size(0))
        if record_int_indices:
            vinds.append(out.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(out.indices[:, -1], dividers[6], \
                    rounding_mode='trunc').int()
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)
            vcoords.append(voxel_dist.unsqueeze(0))

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
            batch_dict['bb3d_layer_time_events'] = events
        if record_vcounts:
            batch_dict['bb3d_num_voxels'] = num_voxels
        if record_int_indices:
            batch_dict['bb3d_intermediary_vinds'] = vinds
        if record_int_vcoords:
            batch_dict['bb3d_intermediary_vcoords'] = vcoords

        return batch_dict
