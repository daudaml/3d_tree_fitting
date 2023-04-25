import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
import open3d.visualization.gui as gui
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize


class TreeFittingController:
    """Class for fitting a tree structure to a point cloud"""

    def __init__(self, config):
        """Initialize class attributes"""

        self.plane_width = config['plane_width']  # Width of the plane used to fit branch structure
        self.plane_length = config['plane_length']  # Length of the plane used to fit branch structure
        self.trunk_geometric_cluster = config['trunk_geometric_cluster']
        self.branch_geometric_cluster = config['branch_geometric_cluster']
        self.tree_data = {}  # Dictionary to store segmented point clouds and center points of the tree
        self.trunk_kdtree = None  # KDTree object for trunk points
        self.trunk_points = []  # Array to store trunk points
        self.this = False
        self.CfEO = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    def process(self, tree_dict):
        """Processes a tree and fits its structure to the point cloud

        Args:
            tree_dict (dict): Dictionary containing information about the tree structure
        """

        # Extract trunk points and create a KDTree for efficient search

        # Fit the trunk and branches
        self.fit_branch(tree_dict['trunk'][0])
        # self.trunk_points = np.array(self.tree_data['tree_centers'][0].points)
        self.trunk_points = np.array(tree_dict['trunk'][0].points)
        self.trunk_kdtree = KDTree(self.trunk_points)
        for branch in tree_dict['left_branches']:
            self.fit_branch(branch, is_branch=True, direction='left')
        for branch in tree_dict['right_branches']:
            self.fit_branch(branch, is_branch=True, direction='right')

        # self.plot_tree_open3d()
        # self.plot_tree_matplot()
        o3d.visualization.draw_geometries(self.tree_data['segmented_pc'])
        self.plot_tree_open3d()

    def plot_tree_open3d(self):
        gui.Application.instance.initialize()
        w = gui.Application.instance.create_window("Lots of lines", 1024, 768)
        widget3d = gui.SceneWidget()
        scene = o3d.visualization.rendering.Open3DScene(w.renderer)
        widget3d.scene = scene
        w.add_child(widget3d)

        trunk_mat = o3d.visualization.rendering.MaterialRecord()
        trunk_mat.shader = "unlitLine"
        trunk_mat.base_color = np.array([92, 67, 34, 255]) / 255
        trunk_mat.line_width = 20

        branch_mat = o3d.visualization.rendering.MaterialRecord()
        branch_mat.shader = "unlitLine"
        branch_mat.base_color = np.array([150, 75, 0, 255]) / 255
        branch_mat.line_width = 8

        for i in range(len(self.tree_data['tree_centers'])):
            points = np.array(self.tree_data['tree_centers'][i].points)
            points_len = len(points)
            indices = np.column_stack([np.arange(0, points_len - 1, 1), np.arange(1, points_len, 1)])
            self.tree_data['tree_centers'][i] = self.create_lines(points, indices)
            if i == 0:
                scene.add_geometry("line " + str(i), self.tree_data['tree_centers'][i], trunk_mat)
            else:
                scene.add_geometry("line " + str(i), self.tree_data['tree_centers'][i], branch_mat)
        widget3d.setup_camera(60, scene.bounding_box, (0, 0, 0))

        gui.Application.instance.run()
        # Visualize the segmented point clouds and center points of the tree
        # note that this is scaled with respect to pixels,
        # so will give different results depending on the
        # scaling values of your system

    def plot_tree_matplot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(self.tree_data['tree_centers'])):
            if i == 0:
                line_width = 10
                line_color = '#5c4322'
            else:
                line_width = 4
                line_color = '#964B00'  # Hex code for brown

            points = np.array(self.tree_data['tree_centers'][i].points)
            # Plot the curve line with the specified line width and color
            ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=line_width, color=line_color)

        # Show the plot
        plt.show()

    def fit_spline(self, points, is_branch=False):
        # Fit a spline to the points
        if is_branch:
            tck, u = splprep(points.T, s=0.001)
        else:
            tck, u = splprep(points.T, s=0.005)

        # Evaluate the spline at 100 points
        u_new = np.linspace(u.min(), u.max(), 200)
        x_new, y_new, z_new = splev(u_new, tck)

        # Create a point cloud from the spline points
        spline_points = np.column_stack((x_new, y_new, z_new))
        return spline_points

    def fit_branch(self, point_cloud, is_branch=False, direction='left'):
        """Fits a branch structure to the given point cloud

        Args:
            point_cloud (open3d.geometry.PointCloud): Point cloud to fit the branch structure to
            is_branch (bool, optional): Flag to indicate whether the point cloud represents a branch or not.
                Defaults to False.
            direction (str, optional): Direction of the branch. Either 'left' or 'right'. Defaults to 'left'.
        """

        # Convert point cloud to array
        points = np.array(point_cloud.points)
        normals = np.array(point_cloud.normals)
        # Get the center points and fit the branch structure
        self.get_center_points(points, is_branch=is_branch, direction=direction)

        # Store the segmented point cloud in the tree_data dictionary
        if 'segmented_pc' in self.tree_data:
            self.tree_data['segmented_pc'].append(point_cloud)
        else:
            self.tree_data['segmented_pc'] = [point_cloud]

    def get_center_points(self, points, is_branch=False, direction='left'):
        """
        Find the center points of each box in the point cloud.

        Args:
            points (np.ndarray): The points in the point cloud.
            is_branch (bool): True if the point cloud is a branch, False otherwise.
            direction (str): The direction of the branch ('left' or 'right').

        Returns:
            None
        """

        # If the point cloud is a branch, swap x and y coordinates
        if is_branch:
            points = points[:, (1, 0, 2)]

        # Get rotation matrix to align points with y-axis
        rot = np.eye(3, 3)

        inside_points_arr = []
        # Initialize empty lists for box centers, rotated box centers, and line segments
        box_centers = []
        rotated_box_centers = []
        line_arr = []
        line_indices = []

        # Find the minimum and maximum y coordinates of the point cloud
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])

        # Loop over the boxes from y_min to y_max
        for i, y in enumerate(np.arange(y_min, y_max + self.plane_length, self.plane_length)):

            # Find the points that are inside the current box
            box_min = [np.min(points[:, 0]) - self.plane_width, y, np.min(points[:, 2]) - self.plane_width]
            box_max = [np.max(points[:, 0]) + self.plane_width, y + self.plane_width,
                       self.plane_width + np.max(points[:, 2])]
            inside_points = points[(points[:, 0] >= box_min[0]) & (points[:, 0] <= box_max[0]) &
                                   (points[:, 1] >= box_min[1]) & (points[:, 1] <= box_max[1]) &
                                   (points[:, 2] >= box_min[2]) & (points[:, 2] <= box_max[2])]
            # Find the center of the current box
            if len(inside_points) > 0:
                inside_points_arr.append(np.array(inside_points))
                box_center = np.mean(inside_points, axis=0)
                line_arr.append([np.min(inside_points[:, 0]) - self.plane_width, y, np.max(inside_points[:, 2])])
                line_arr.append([np.max(inside_points[:, 0]) + self.plane_width, y, np.max(inside_points[:, 2])])
                line_indices.append([i * 2, (i * 2) + 1])
                box_centers.append(box_center)

        # Convert lists to NumPy arrays
        box_centers = np.array(box_centers)
        rotated_box_centers = np.array(box_centers)
        line_arr = np.array(line_arr)

        # Rotate box centers and line segments back to original orientation
        box_centers = np.dot(box_centers, rot.T)
        rotated_box_centers = np.dot(rotated_box_centers, rot.T)
        line_arr = np.dot(line_arr, rot.T)

        # If the point cloud is a branch, adjust the box centers to align with trunk
        if is_branch:
            points = points[:, (1, 0, 2)]
            box_centers = box_centers[:, (1, 0, 2)]
            rotated_box_centers = rotated_box_centers[:, (1, 0, 2)]
            line_arr = line_arr[:, (1, 0, 2)]
            if direction == 'left':
                _, index = self.trunk_kdtree.query(box_centers[np.argmax(box_centers[:, 0])])
                box_centers = np.row_stack([box_centers, self.trunk_points[index]])
            else:
                _, index = self.trunk_kdtree.query(box_centers[np.argmin(box_centers[:, 0])])
                box_centers = np.row_stack([self.trunk_points[index], box_centers])

        box_centers = self.fit_spline(box_centers, is_branch=is_branch)
        line_arr, theta_arr = self.orientation_correction(line_arr, line_indices, box_centers, is_branch=is_branch)
        # self.calculate_diameters(line_arr, line_indices, theta_arr, points, is_branch=is_branch, direction=direction)
        self.add_to_data(box_centers, rotated_box_centers, line_arr, line_indices)

    def calculate_diameters(self, line_arr, line_ind, theta_arr, points, is_branch=False, direction='left'):
        diameters = []
        lengths = []
        corners = []
        points_pc = []
        colors_pc = []
        if is_branch:
            pass

        if direction == 'right':
            if self.this:
                # plt.scatter(points[:, 0], points[:, 1])
                # plt.show()
                for i in range(len(line_ind)):
                    inside_points = self.get_points_inside_rotated_rect(
                        (line_arr[line_ind[i][0], 0] + line_arr[line_ind[i][1], 0]) / 2,
                        (line_arr[line_ind[i][0], 1] + line_arr[line_ind[i][1], 1]) / 2,
                        0.02, 0.1, theta_arr[i], points)
                    points_pc.extend(inside_points)
                    colors_pc.extend(self.random_color_array(len(inside_points)))
                print(np.array(points_pc).shape)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_pc)
                pcd.colors = o3d.utility.Vector3dVector(colors_pc)
                o3d.visualization.draw_geometries([pcd])
                # exit()
            self.this = True


        else:
            pass

    def random_color_array(self, n):
        rgb = np.random.rand(3)
        return np.tile(rgb, (n, 1))

    def get_points_inside_rotated_rect(self, center_x, center_y, width, height, theta, points):
        # Define the corners of the unrotated rectangle
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y - height / 2
        x3 = center_x + width / 2
        y3 = center_y + height / 2
        x4 = center_x - width / 2
        y4 = center_y + height / 2

        # Define the rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        # Rotate the corners of the rectangle
        corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        center = np.array([center_x, center_y])
        rotated_corners = np.matmul(R, (corners - center).T).T + center
        # Generate a regular grid of points inside the bounding box of the rotated rectangle
        x_min, x_max = min(rotated_corners[:, 0]), max(rotated_corners[:, 0])
        y_min, y_max = min(rotated_corners[:, 1]), max(rotated_corners[:, 1])

        # Test whether each point lies inside the rectangle using the winding number algorithm
        def point_inside_rect(point):
            point = point[:2]
            x, y = point
            if x < x_min or x > x_max or y < y_min or y > y_max:
                return False
            v1 = rotated_corners[1] - rotated_corners[0]
            v2 = point - rotated_corners[0]
            if np.cross(v1, v2) < 0:
                return False
            v1 = rotated_corners[2] - rotated_corners[1]
            v2 = point - rotated_corners[1]
            if np.cross(v1, v2) < 0:
                return False
            v1 = rotated_corners[3] - rotated_corners[2]
            v2 = point - rotated_corners[2]
            if np.cross(v1, v2) < 0:
                return False
            v1 = rotated_corners[0] - rotated_corners[3]
            v2 = point - rotated_corners[3]
            if np.cross(v1, v2) < 0:
                return False
            return True

        # Filter points that lie inside the rectangle
        inside_points = list(filter(point_inside_rect, points))

        return inside_points

    def orientation_correction(self, line_end_points, line_indices, box_centers, is_branch=False):
        theta_arr = []
        if is_branch:
            for ind in line_indices:
                prev_p, next_p = self.find_closest_points(line_end_points[ind[0]], box_centers, axis=0)
                theta = self.rotation_vector(prev_p, next_p, axis=0)
                theta_arr.append(theta)
                center_point = (line_end_points[ind[0]] + line_end_points[ind[1]]) / 2
                line_end_points[ind[0]] = self.rotate_point(theta, line_end_points[ind[0]], center_point, axis=0)
                line_end_points[ind[1]] = self.rotate_point(theta, line_end_points[ind[1]], center_point, axis=0)
        else:
            for ind in line_indices:
                prev_p, next_p = self.find_closest_points(line_end_points[ind[0]], box_centers)
                theta = self.rotation_vector(prev_p, next_p)
                theta_arr.append(theta)
                center_point = (line_end_points[ind[0]] + line_end_points[ind[1]]) / 2
                line_end_points[ind[0]] = self.rotate_point(theta, line_end_points[ind[0]], center_point)
                line_end_points[ind[1]] = self.rotate_point(theta, line_end_points[ind[1]], center_point)

        return line_end_points, np.array(theta_arr)

    def rotate_point(self, theta, p, center_p, axis=1):
        # Calculate the rotation matrix from the rotation vector

        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])

        # Apply the rotation to the point
        if axis == 1:
            rotated_p = np.matmul(R.T, (p - center_p).T).T + center_p
        else:
            rotated_p = np.matmul(R, (p - center_p).T).T + center_p
        return rotated_p

    def rotation_vector(self, p1, p2, axis=1):
        if axis == 1:
            return np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
        else:
            return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def find_closest_points(self, A, B, axis=1):
        # Convert input points to numpy arrays
        A = np.array(A)
        B = np.array(B)
        # Calculate distances between A and all points in B
        distances = np.abs(B[:, axis] - A[axis])
        # Find indices of two closest points
        indices = np.argsort(distances)[:2]

        # Return the two closest points
        closest_points = B[indices]
        previous_point = closest_points[0]
        next_point = closest_points[1]
        if previous_point[axis] > next_point[axis]:
            previous_point, next_point = next_point, previous_point
        return previous_point, next_point

    def add_to_data(self, box_centers, rotated_box_centers, line_arr, line_indices):
        """
        Add box centers, rotated box centers, and lines to the tree data.

        Args:
            box_centers (numpy.ndarray): A NumPy array of box centers.
            rotated_box_centers (numpy.ndarray): A NumPy array of rotated box centers.
            line_arr (numpy.ndarray): A NumPy array of line points.
            line_indices (list): A NumPy array of line indices.
        """
        # Create point clouds and line set
        box_centers_pcd = self.create_pcd(box_centers)
        rotated_box_centers_pcd = self.create_pcd(rotated_box_centers)
        line_set = self.create_lines(line_arr, line_indices)

        self.tree_data.setdefault('segmented_pc', []).extend([box_centers_pcd, line_set])
        self.tree_data.setdefault('tree_centers', []).append(box_centers_pcd)
        self.tree_data.setdefault('rot_tree_centers', []).append(rotated_box_centers_pcd)
        self.tree_data.setdefault('line_set', []).append(line_set)

    def create_pcd(self, points, color=[1, 0, 0]):
        """
        Create a point cloud from a NumPy array of points.

        Args:
            points (numpy.ndarray): A NumPy array of points.
            color (list): [r,g,b]

        Returns:
            o3d.geometry.PointCloud: A point cloud object created from the input points.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        return pcd

    def create_lines(self, points, indices):
        """Creates a line set from a list of points and a list of line indices.

        Args:
            points (numpy.ndarray): A numpy array of shape (n, 3) representing the point coordinates.
            indices (list): A numpy array of shape (m, 2) representing the indices of the start and end points
                of the line segments.

        Returns:
            o3d.geometry.LineSet: An Open3D line set object representing the line segments.
        """
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(indices)
        return line_set

    def find_center(self, min_pt, max_pt, points, R):
        # Find the center point of the rotated box
        rotated_box_min = np.dot(min_pt, R)
        rotated_box_max = np.dot(max_pt, R)

        corners = np.array([rotated_box_min, [rotated_box_min[0], rotated_box_max[1], rotated_box_min[2]],
                            rotated_box_max, [rotated_box_max[0], rotated_box_min[1], rotated_box_max[2]]])

        # Use the winding number algorithm to check if each point is inside the rectangle
        is_inside = []
        for point in points:
            is_inside.append(self.is_inside_rotated_rectangle(point, corners, R))

        # Compute the center point of the local points
        local_points = points[is_inside]
        local_center = np.mean(local_points, axis=0)

        # Transform the center point back to the global coordinate system
        center = local_center

        return center

    def is_inside_rotated_rectangle(self, point, corners, R):
        # Rotate the point to the local coordinate system of the rectangle
        local_point = point

        # Check if the point is inside the rectangle using the winding number algorithm
        winding_number = 0
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            if p1[1] <= local_point[1]:
                if p2[1] > local_point[1]:
                    if (p2[0] - p1[0]) * (local_point[1] - p1[1]) - (local_point[0] - p1[0]) * (p2[1] - p1[1]) > 0:
                        winding_number += 1
            else:
                if p2[1] <= local_point[1]:
                    if (p2[0] - p1[0]) * (local_point[1] - p1[1]) - (local_point[0] - p1[0]) * (p2[1] - p1[1]) < 0:
                        winding_number -= 1
        return winding_number != 0

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """Find the rotation matrix that aligns vec1 to vec2.

        Args:
            vec1 (numpy.ndarray): A 3d "source" vector.
            vec2 (numpy.ndarray): A 3d "destination" vector.

        Returns:
            numpy.ndarray: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def simplePCA(self, arr):
        """Calculate the mean, eigenvalues, and eigenvectors of the input array.

        Args:
            arr (numpy.ndarray): The array for which to calculate the PCA.

        Returns:
            tuple: A tuple containing the mean, eigenvalues, and eigenvectors.
        """
        # calculate mean
        m = np.mean(arr, axis=0)
        sd = np.std(arr, axis=0)

        # center data
        arrm = arr - m

        Cov = np.cov(arrm.T)
        eigval, eigvect = np.linalg.eig(Cov.T)

        # return mean, eigenvalues, eigenvectors
        return m, eigval, eigvect
