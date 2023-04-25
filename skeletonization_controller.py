import os.path
import pyk4a
import yaml
from sensor.az_controller import AzureController


class SkeletonizationController:
    def __init__(self, config):
        self.config = config
        pass

    def run(self):
        input_file_path = os.path.join(
            self.config['data_path'],
            self.config['input_file']
        )
        if self.config['current_sensor'] == 0:
            # Azure Sensor
            print('Running on azure sensor')
            playback = pyk4a.PyK4APlayback(input_file_path)
            az_controller = AzureController(playback, self.config)
            az_controller.play()
            playback.close()
        elif self.config['current_sensor'] == 1:
            # Realsense Sensor
            pass


if __name__ == '__main__':
    with open("settings.yaml", "r") as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    skeletonization_controller = SkeletonizationController(settings)
    skeletonization_controller.run()
