import torch

from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class DynamicFilterMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        #print(f'grid_size {grid_size}') # [1024 1024 40]
        #print(f'voxel_size {voxel_size}') # [0.1 0.1 0.2]
        #print(f'point_cloud_range {point_cloud_range}') # [-51.2 -51.2 -5. 51.2 51.2 3.]
        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.tile_begin_idx=0

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        # point_coords are cell indexes
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) \
                / self.voxel_size).long()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        point_coords = torch.cat((points[:, :1].long(), point_coords), dim=1)

        points = points[mask]
        point_coords = point_coords[mask]

        ####
        batch_dict['points'], batch_dict['point_coords'] = points, point_coords
        batch_dict = kwargs['model'].schedule(batch_dict)
        points, point_coords = batch_dict['points'], batch_dict['point_coords']
        ####

        #merge_coords = points[:, 0].long() * self.scale_xyz + \
        merge_coords = point_coords[:, 0] * self.scale_xyz + \
                        point_coords[:, 1] * self.scale_yz + \
                        point_coords[:, 2] * self.scale_z + \
                        point_coords[:, 3]

        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True,
                return_counts=False)
        unq_coords = unq_coords.long()

        #unq_coords are the voxel coords

        voxel_coords = torch.stack((torch.div(unq_coords, self.scale_xyz, rounding_mode='trunc'),
                torch.div((unq_coords % self.scale_xyz), self.scale_yz, rounding_mode='trunc'),
                torch.div((unq_coords % self.scale_yz), self.scale_z, rounding_mode='trunc'),
                unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        batch_dict['voxel_coords'] = voxel_coords.contiguous()

        points_data = points[:, 1:].contiguous()
        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
