import os.path

from ai.yolov8 import YoloInference
from utils.common_utils import *


class AIController:
    def __init__(self, model_path, config):
        self.yolo_inference = YoloInference(model_path)
        self.input_image_folder = os.path.join(
            config['project_path'],
            config['data_path'],
            config['input_image_folder']
        )

        self.output_image_folder = os.path.join(
            config['project_path'],
            config['data_path'],
            config['image_segmentation_output_path']
        )
        create_paths(self.input_image_folder, self.output_image_folder)

    def run_inference(self, image, image_id):
        save_image_path = os.path.join(self.input_image_folder, '{:04d}.jpg'.format(image_id))
        save_image(image, save_image_path)

        plotted_image, predicted_instances, class_ids, scores = self.yolo_inference.predict(image)
        save_image_path = os.path.join(self.output_image_folder, '{:04d}.jpg'.format(image_id))
        save_image(plotted_image, save_image_path)

        return predicted_instances, class_ids
