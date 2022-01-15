from typing import List

import torch
import numpy as np
import cv2
from time import time

from model.IdCardObject import IdCardObject
from model.ImageReader import ImageReader
from model.LabeledImage import LabeledImage

PRETRAINED_MODEL_PATH = 'weights/best.pt'
PERCENTAGE_LIMIT = 0.3
DEBUG = False


class DetectionModel:

    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.id_card = IdCardObject()
        self.text_detector = ImageReader()

    def read_image_from_path(self, path):
        image = cv2.imread(path)
        return image

    def load_model(self):
        return torch.hub.load('ultralytics/yolov5', 'custom', path=PRETRAINED_MODEL_PATH, force_reload=True)

    def detect_image(self, image):
        self.model.to(self.device)

        results = self.model(image)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, image):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = image.shape[1], image.shape[0]
        for i in range(n):
            row = cord[i]

            if row[4] >= PERCENTAGE_LIMIT:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 0, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 1)
                cv2.putText(image, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                            1)

        return image

    def crop_detected_areas(self, results, image):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = image.shape[1], image.shape[0]

        cropped_images = []
        for i in range(n):
            row = cord[i]

            if row[4] >= PERCENTAGE_LIMIT:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                label_name = self.class_to_label(labels[i])

                crop_img = image[y1:y2, x1:x2]
                cropped_images.append(LabeledImage(crop_img, label_name))

        return cropped_images

    def process_images_for_reading(self, images: List[LabeledImage]):
        processed_images: List[LabeledImage] = []
        for img in images:
            processed_images.append(self.process_image_for_reading(img))

        return processed_images

    def process_image_for_reading(self, labeled_image: LabeledImage):
        # gray
        gray = cv2.cvtColor(labeled_image.image, cv2.COLOR_BGR2GRAY)

        # blur
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)

        # divide
        divide = cv2.divide(gray, blur, scale=255)

        # otsu threshold
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        setattr(labeled_image, "image", morph)
        return labeled_image

    def read_text_and_save(self, images: List[LabeledImage]):
        for image in images:
            result = self.read_text_from_image(image.image)
            txt = "".join(result) if not (image.label == "birthPlace" and len(result) == 2) else " ".join(
                result)  # due to birthPlace has to rows
            setattr(self.id_card, image.label, txt)

    def read_text_from_image(self, image):
        return self.text_detector.read_text(image)

    def process_image(self, path: str):
        self.id_card.reset()

        start_time = time()
        if DEBUG:
            print(f"Start: {start_time}")

        image = self.read_image_from_path(path)
        results = self.detect_image(image)

        if DEBUG:
            debug_image = image.copy()
            debug_image = self.plot_boxes(results, debug_image)
            cv2.imwrite("./output/detectedAreas.jpg", debug_image)

        cropped_images: List[LabeledImage] = self.crop_detected_areas(results, image)

        images_for_reading: List[LabeledImage] = self.process_images_for_reading(cropped_images)

        if DEBUG:
            for img in images_for_reading:
                cv2.imwrite("./output/" + img.label + ".jpg", img.image)

        self.read_text_and_save(images_for_reading)

        json = self.id_card.convert_to_json()
        print(json)

        end_time = time()
        if DEBUG:
            print(f"End: {end_time}")

        processing_time = np.round(end_time - start_time, 3)
        print(f"Processing took : {processing_time} seconds")


if __name__ == '__main__':
    image_path = "cards/"  # path to ID card image

    model = DetectionModel()
    model.process_image(image_path)
