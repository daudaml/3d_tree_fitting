from ultralytics import YOLO
import numpy as np
from ultralytics.yolo.utils.ops import scale_image


class YoloInference:
    def __init__(self, model_path):
        """initialization
        """
        self.model = YOLO(model_path)  # load a pretrained YOLOv8 segmentation model
        print("Model weights loaded")

    def predict(self, image):
        """predicting image
        """
        result = self.model(source=image, retina_masks=True)[0]
        res_plotted = result.plot(font_size=7, line_width=3)
        masks = result.masks.masks.cpu().numpy()  # masks, (N, H, W)
        masks = np.moveaxis(masks, 0, -1)  # masks, (H, W, N)
        masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
        masks = np.array(masks, dtype=bool)
        cls = result.boxes.cls.cpu().numpy()  # cls, (N, 1)
        probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)

        return res_plotted, masks, cls, probs
