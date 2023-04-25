import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import open3d as o3d


class SegmentationController:
    """
    This class performs tree point cloud segmentation using DBSCAN clustering algorithm.
    """

    def __init__(self, config):
        self.existing_colors = []  # list to keep track of existing colors used for visualization
        # tree structure for point cloud visualization
        self.tree_structure = {"trunk": [], "left_branches": [], "right_branches": []}
        # coordinate frame for visualization
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self.min_trunk_height = config['min_trunk_height']
        self.voxel_size = config['voxel_size']
        self.dbscan_eps = config['dbscan_eps']
        self.dbscan_min_samples = config['dbscan_min_samples']
        self.min_branch_width = config['min_branch_width']
        self.min_branch_trunk_distance = config['min_branch_trunk_distance']

    def process(self, trunk_pc, branch_pc):
        self.process_trunk(trunk_pc)
        self.process_branch(branch_pc, self.tree_structure['trunk'])
        return self.tree_structure

    def process_trunk(self, pc_arr):
        segmented_clouds = []
        for pcd in pc_arr:
            pcd = self.voxel_down_sample(pcd)
            labels = self.dbscan_cluster(pcd)
            # Postprocess the segmented clusters
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    continue  # ignore noise points
                cloud = pcd.select_by_index(np.where(labels == label)[0])
                cloud_points = np.array(cloud.points)
                # filter out small segments based on their length along z-axis
                if self.length_along_axis(cloud_points, 'y') > self.min_trunk_height:
                    cloud.paint_uniform_color(self.generate_unique_color())
                    segmented_clouds.append(cloud)
        index = self.closest_to_origin_point_cloud(segmented_clouds)  # find the segment closest to the origin
        self.tree_structure['trunk'] = segmented_clouds[index]

    def voxel_down_sample(self, pcd):
        """
        Downsample the input point cloud using voxel grid with a fixed voxel size.
        """
        return pcd.voxel_down_sample(voxel_size=0.008)

    def dbscan_cluster(self, pcd):
        """
        Cluster the input point cloud using DBSCAN algorithm and return the cluster labels.
        """
        # Preprocess the data
        points = np.asarray(pcd.points)
        points = StandardScaler().fit_transform(points)  # scale the points to have zero mean and unit variance

        # Apply DBSCAN clustering algorithm
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, algorithm='kd_tree', n_jobs=-1)
        labels = dbscan.fit_predict(points)
        return labels

    def closest_to_origin_point_cloud(self, point_clouds):
        """
        Find the index of the point cloud in the list that is closest to the origin.
        """
        # convert the list of point clouds to a list of numpy arrays
        center_clouds = np.asarray([np.mean(point_cloud.points, axis=0) for point_cloud in point_clouds])
        # return the index of the point cloud with smallest x-coordinate (closest to origin)
        return np.argmin(np.abs(center_clouds[:, 0]))

    def process_branch(self, branches, trunk):
        """
        Given a set of branches and trunk point clouds, this function performs segmentation of branches into left and
        right clusters with respect to the trunk. It returns a dictionary containing segmented left and right branches
        as point clouds.

        Args:
            branches (open3d.geometry.PointCloud): Point cloud data for all branches
            trunk (open3d.geometry.PointCloud): Point cloud data for the trunk
        """
        left_branches = []
        right_branches = []
        # Convert the trunk and branch point clouds into numpy arrays
        source_points = np.array(trunk.points)
        for branch in branches:
            branch = self.voxel_down_sample(branch)
            target_points = np.array(branch.points)
            is_left_branch = self.is_branch_on_left(source_points, target_points)
            labels = self.dbscan_cluster(branch)
            # Postprocess the segmented clusters
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    continue  # ignore noise points
                cloud = branch.select_by_index(np.where(labels == label)[0])
                cloud_points = np.array(cloud.points)
                cluster_distance_trunk = min(cloud.compute_point_cloud_distance(trunk))
                accumulated_length = self.length_along_axis(cloud_points, 'x') + \
                                     self.length_along_axis(cloud_points, 'y') + \
                                     self.length_along_axis(cloud_points, 'z')
                # filter out small segments based on their length along z-axis
                if accumulated_length > self.min_branch_width and cluster_distance_trunk < self.min_branch_trunk_distance:
                    if is_left_branch:
                        cloud.paint_uniform_color([1, 0, 0])
                        left_branches.append(cloud)
                    else:
                        cloud.paint_uniform_color([0, 1, 0])
                        right_branches.append(cloud)

        self.tree_structure['left_branches'] = left_branches

        self.tree_structure['right_branches'] = right_branches

    def is_branch_on_left(self, trunk_points, branch_points):
        """
        Masks the left branches of a tree based on their position relative to the main trunk.

        Args:
            trunk_points (np.array): A point cloud containing the main trunk of the tree.
            branch_points (np.array): A point cloud containing the branches of the tree.

        Returns:
            A boolean array indicating which points in the branch point cloud correspond to left branches.
        """
        # Find the centroid of the main trunk
        trunk_centroid = np.mean(trunk_points, axis=0)
        branch_centroid = np.mean(branch_points, axis=0)

        ref_point = np.array([trunk_centroid[0], np.min(trunk_points[:, 1]), trunk_centroid[2]])
        branch_vec = branch_centroid - trunk_centroid
        ref_vec = ref_point - trunk_centroid
        cross_prods = np.cross(branch_vec, ref_vec)
        left_mask = cross_prods[2] > 0
        return left_mask

    def generate_unique_color(self):
        """
        Generates a unique RGB color.

        Returns:
            A numpy array representing the RGB color (values between 0 and 1).
        """
        while True:
            # Generate a random RGB color
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            new_color = [r, g, b]

            # Check if the new color is unique
            if new_color not in self.existing_colors:
                self.existing_colors.append(new_color)
                return np.array(new_color) / 255

    def length_along_axis(self, points: np.ndarray, axis: str = 'x'):
        """
        Returns the length of a 3D point cloud along the specified axis.

        Args:
            points: A numpy array of 3D points.
            axis: The axis along which to calculate the length. Must be one of 'x', 'y', or 'z'.

        Returns:
            The length of the point cloud along the specified axis.
        """
        if axis == 'x':
            return np.ptp(points[:, 0])
        elif axis == 'y':
            return np.ptp(points[:, 1])
        elif axis == 'z':
            return np.ptp(points[:, 2])
        else:
            raise ValueError("Invalid axis input. Accepted inputs are 'x', 'y', and 'z'.")
