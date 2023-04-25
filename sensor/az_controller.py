from utils.az_utils import convert_to_bgra_if_required
import pyk4a
import copy
import numpy as np
from utils.common_utils import *
from utils.pc_utils import *
from ai.ai_controller import AIController
from segmentation.segmentation_controller import SegmentationController
from fitting.tree_fitting_controller import TreeFittingController

class AzureController:
    def __init__(self, playback, config):
        playback.open()
        self.playback = playback
        self.distance_threshold = config['distance_threshold']
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        self.segmentation_controller = SegmentationController(config['segmentation'])
        self.tree_fitting_controller = TreeFittingController(config['fitting'])
        self.ai_controller = AIController(
            os.path.join(config['project_path'], config['model_path']),
            config)

    def info(self):
        print(f"Record length: {self.playback.length / 1000000: 0.2f} sec")

    def play(self):
        count = 0
        while True:
            try:
                branch_pc = []
                trunk_pc = []
                capture = self.playback.get_next_capture()

                if capture.color is None:
                    continue
                if capture.depth is None:
                    continue

                res = input("Segment image? Y or N ")
                if res == "N":
                    print("Exiting the program!")
                    break
                print('Capturing next frame and passing to ai model')
                image = convert_to_bgra_if_required(self.playback.configuration["color_format"], capture.color)
                depth = capture.transformed_depth
                predicted_instances, class_ids = self.ai_controller.run_inference(image, count)
                number_of_instances = predicted_instances.shape[2]
                capture._color = cv2.cvtColor(cv2.imdecode(capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
                capture._color_format = pyk4a.ImageFormat.COLOR_BGRA32
                for i in range(number_of_instances):
                    mask = predicted_instances[:, :, i]
                    sub_depth = copy.deepcopy(depth)
                    sub_depth[~mask] = 0
                    sub_depth[sub_depth > self.distance_threshold] = 0
                    points = pyk4a.transformation.depth_image_to_point_cloud(
                        sub_depth,
                        capture._calibration,
                        capture.thread_safe,
                        calibration_type_depth=False
                    ).reshape((-1, 3)) / 1000
                    colors = capture.color[..., (2, 1, 0)].reshape((-1, 3)) / 255
                    colors = colors[~np.all(points == 0, axis=1)]
                    points = points[~np.all(points == 0, axis=1)]
                    # points = np.array(points)[:, (0, 1, 2)]
                    # points[:, 2] = -points[:, 2]
                    pc = create_pc(points, colors)
                    if points.shape[0] > 50:
                        if class_ids[i] == 0:
                            trunk_pc.append(pc)
                        elif class_ids[i] == 1:
                            branch_pc.append(pc)
                segmented_tree = self.segmentation_controller.process(trunk_pc, branch_pc)
                # self.tree_segmentation.association_algorithm(trunk_pc, branch_pc)
                count += 1
            except EOFError:
                break
