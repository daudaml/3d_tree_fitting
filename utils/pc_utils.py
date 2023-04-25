import open3d as o3d
import numpy as np


def get_class_pc(points, class_colors):
    for key, value in class_colors.items():
        print(key, value)


def visualize_multiple_pc(pcd_arr):
    o3d.visualization.draw_geometries(pcd_arr)


def create_pc(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def view_pc(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
