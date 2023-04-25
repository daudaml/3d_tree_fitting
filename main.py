import open3d as o3d
import numpy as np
from tree_segmentation import TreeSegmentation
from tree_fitter import TreeFitter

pc_path = '20230322-191855.ply'


def get_trunk_pc(pc):
    points = np.array(pc.points)
    colors = np.array(pc.colors)
    normals = np.array(pc.normals)
    points = points[np.all(colors == [0, 0, 1], axis=1)]
    normals = normals[np.all(colors == [0, 0, 1], axis=1)]
    colors = colors[np.all(colors == [0, 0, 1], axis=1)]
    trunk_pc = o3d.geometry.PointCloud()
    trunk_pc.points = o3d.utility.Vector3dVector(points)
    trunk_pc.normals = o3d.utility.Vector3dVector(normals)
    trunk_pc.colors = o3d.utility.Vector3dVector(colors)
    return trunk_pc


def get_branch_pc(pc):
    points = np.array(pc.points)
    colors = np.array(pc.colors)
    normals = np.array(pc.normals)
    points = points[np.all(colors == [0, 1, 0], axis=1)]
    normals = normals[np.all(colors == [0, 1, 0], axis=1)]
    colors = colors[np.all(colors == [0, 1, 0], axis=1)]
    branch_pc = o3d.geometry.PointCloud()
    branch_pc.points = o3d.utility.Vector3dVector(points)
    branch_pc.normals = o3d.utility.Vector3dVector(normals)
    branch_pc.colors = o3d.utility.Vector3dVector(colors)
    return branch_pc


# loading pointcloud
input_pc = o3d.io.read_point_cloud(pc_path)
input_pc_points = np.array(input_pc.points)[:, (0, 2, 1)]
input_pc_points[:, 2] = -input_pc_points[:, 2]
input_pc.points = o3d.utility.Vector3dVector(input_pc_points)
o3d.visualization.draw_geometries([input_pc])
trunk = get_trunk_pc(input_pc)
branch = get_branch_pc(input_pc)
# segmentation
point_cloud_segmentation = TreeSegmentation()
trunk_segmented_pc = point_cloud_segmentation.process_trunk(trunk)
branch_trunk_segmented = point_cloud_segmentation.process_branch(branch, trunk_segmented_pc)
tree_fitter = TreeFitter()
tree_fitter.process(branch_trunk_segmented)
# tree_fitter.plot()
