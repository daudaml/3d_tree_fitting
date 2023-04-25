import os
import cv2


def create_paths(*args):
    for folder in args:
        if not os.path.exists(folder):
            os.mkdir(folder)


def save_image(image, image_path):
    cv2.imwrite(image_path, image)
