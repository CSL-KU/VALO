"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
]

vis = None
def initialize_visualizer():
    global vis
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.ones(3)

def destroy_visualizer():
    global vis
    vis.destroy_window()

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, max_num_tiles=18, pc_range=None, nonempty_tile_coords=None, tile_coords=None, clusters=None, ground_points=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    global vis
    vis.clear_geometries()
    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()

#    #NOTE Only display most recent sweep start
#    mrs_mask = (points[:, -1] == 0.)
#    points = points[mrs_mask]
    # filter point outside of pc_range
    points_x = points[:, 0]
    points_y = points[:, 1]
    x_range_mask = np.logical_and(points_x > pc_range[0], points_x < pc_range[3])
    y_range_mask = np.logical_and(points_y > pc_range[1], points_y < pc_range[4])
    points = points[np.logical_and(x_range_mask, y_range_mask)]

    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    if point_colors is None:
        cornflower_blue = np.array([[100., 149., 237.]], dtype=np.float64)/255.
        clrs = np.repeat(cornflower_blue, points.shape[0], axis=0)
        #3# make the most recent sweep pink
        ##mrs_mask = (points[:, -1] == 0.)
        ##clrs[mrs_mask] = np.array([255.,192.,203.])/255.
        pts.colors = open3d.utility.Vector3dVector(clrs)
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)


    if ground_points is not None:
        gpts= open3d.geometry.PointCloud()
        gpts.points = open3d.utility.Vector3dVector(ground_points[:, :3])
        clrs = np.repeat([[0.,1.,0.]], ground_points.shape[0], axis=0)
        gpts.colors = open3d.utility.Vector3dVector(clrs)
        vis.add_geometry(gpts)

    use_voxels=False
    if use_voxels:
        pts = open3d.geometry.VoxelGrid.create_from_point_cloud(pts, 0.1)

    vis.add_geometry(pts)

    #print(gt_boxes)
    #if gt_boxes is not None:
    #    vis = draw_box(vis, gt_boxes, (1., 0., 0.))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0., 1., 0.), ref_labels, ref_scores)

    if tile_coords is not None:
        #Draw all tiles, but change the color of chosen ones
        #pc range: -x -y -z +x +y +z
        tc = torch.from_numpy(tile_coords)
        tile_w = (pc_range[3] - pc_range[0]) / max_num_tiles
        tile_h = (pc_range[4] - pc_range[1])

        v_top = np.array((pc_range[0], pc_range[1]))
        v_bot = np.array((pc_range[0], pc_range[4]))
        vertices_top = np.array([v_top + (tile_w*i, 0) for i in range(max_num_tiles+1)])
        vertices_bot = np.array([v_bot + (tile_w*i, 0) for i in range(max_num_tiles+1)])
        vertices = np.concatenate((vertices_top, vertices_bot), axis=0)
        vertices = np.concatenate((vertices, np.full((vertices.shape[0],1), -3,
            dtype=vertices.dtype)), axis=1)

        # The whole area
        s = vertices.shape[0]//2
        lines_area = [(0,s), (0,s-1), (s,2*s-1), (s-1,2*s-1)]

        #  Nonempty area
        netc = nonempty_tile_coords
        st, et = netc[0], netc[-1]
        lines_nonempty_area = [(st, st+s), (st, et+1), (st+s, et+s+1), (et+1, et+s+1)]

        lines_chosen = []
        for t in range(max_num_tiles):
            if t in tile_coords:
                vl =[(t,t+1), (t,t+s), (t+s,t+s+1), (t+1,t+s+1)]
                lines_chosen.extend(vl)

        #print('Tile coords:', tile_coords)
        vertices_ = vertices.copy()
        vertices_[:s] += (0, 0.2, 0)
        vertices_[s:] += (0, -0.2, 0)
        all_vertices = (vertices, vertices_)
        all_lines = (lines_area, lines_chosen)
        all_colors = (np.array([0.,0.,0.], dtype=np.float64), \
                np.array([125., 206., 160.], dtype=np.float64)/255.)
        for vertices, lines, colors in zip(all_vertices, all_lines, all_colors):
            open3d_vertices = open3d.utility.Vector3dVector(vertices)
            open3d_vertex_pairs = open3d.utility.Vector2iVector(np.array(lines))
            rectangles = open3d.geometry.LineSet(open3d_vertices, open3d_vertex_pairs)
            rectangles.paint_uniform_color(colors)
            vis.add_geometry(rectangles)

        if clusters is not None:
            for clu in clusters:
                try:
                    clu_v = open3d.utility.Vector3dVector(clu.astype(np.float64))
                    bb = open3d.geometry.AxisAlignedBoundingBox.create_from_points(clu_v)
                    bb.color= (1., 0., 0.) # red
                    vis.add_geometry(bb)
                except:
                    print('Bad cluster')


    else:
        # Assume there are 16 vertical tiles
        pass

    vis.poll_events()
    vis.update_renderer()
    #vis.run()
    #vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0., 1., 0.), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        #if ref_labels is None:
        #    line_set.paint_uniform_color(color)
        #else:
        #    line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        #print(score)
        if score is not None: 
            c = np.array([142., 68., 173.], dtype=np.float64)/255. if score[i] <= 0.3 else \
                    np.array([39., 174., 96.], dtype=np.float64)/255.
        else:
            c = np.array(color, dtype=np.float64)
        line_set.paint_uniform_color(c)
        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
