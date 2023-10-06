import cv2

# import argparse
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from ultralytics.utils.plotting import Annotator
import torch
from loguru import logger
import json
import time
import os


class OcrOnEdge:
    def __init__(self):
        # Initialize detection model and OCR reader
        logger.info("Initializing OcrOnEdge...")
        self.detection_model = YOLO("./models/model.onnx", task="detect")
        logger.info("Detection model initialized.")
        self.reader = easyocr.Reader(
            ["nl"],
            gpu=True,
            detector=False,
            download_enabled=False,
            model_storage_directory="models",
        )
        self.reader.recognizer.eval()
        logger.info("OCR reader initialized.")

        # Warmup the OCR recognizer
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(100):
            with torch.no_grad():
                self.reader.recognize(dummy_image, batch_size=5)
        logger.info("OCR recognizer warmed up.")
        logger.success("Initialization complete.")

    def read_config(self, filename):
        with open(filename, "r") as config_file:
            self.config = json.load(config_file)

    def ocr_detection(self, image):
        results = self.detection_model(image)
        cropped_image = None
        highest_confidence = 0
        highest_confidence_box = None

        for r in results:
            boxes = r.boxes
            if len(boxes) < 1:
                logger.critical("No label detected!")
                return None
            for box in boxes:
                b = box.xyxy[0]
                c = box.conf[0]

                if c > highest_confidence:
                    highest_confidence = c
                    highest_confidence_box = b

        if highest_confidence_box is not None:
            cropped_image = self.crop_date_region(image, highest_confidence_box)

        if cropped_image is not None:
            return cropped_image, highest_confidence_box, highest_confidence
        else:
            return None

    def crop_date_region(self, image, box, padding=5):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Calculate the dimensions of the cropped region
        crop_width = x2 - x1
        crop_height = y2 - y1

        background_width = crop_width + 2 * padding
        background_height = crop_height + 2 * padding
        background = (
            np.ones((background_height, background_width, 3), dtype=np.uint8) * 255
        )  # white border

        bg_x1 = padding
        bg_y1 = padding
        bg_x2 = bg_x1 + crop_width
        bg_y2 = bg_y1 + crop_height

        background[bg_y1:bg_y2, bg_x1:bg_x2] = image[y1:y2, x1:x2]

        output_folder = "results-cropped"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        base_filename = "result-cropped.jpeg"
        output_path = os.path.join(output_folder, base_filename)

        count = 1
        while os.path.exists(output_path):
            base_filename = f"result-cropped_{count}.jpeg"
            output_path = os.path.join(output_folder, base_filename)
            count += 1

        cv2.imwrite(output_path, background)
        return background

    def pre_processing(self, image):
        # NOTE: Pre-processing not needed - consider removing config file
        dot_matrix = self.config["runtime_configuration"]["dot_matrix"]

        if dot_matrix:
            # Kernel
            kernel = np.ones((4, 4), np.uint8)

            # Dilation
            image_dilate = cv2.dilate(image, kernel, iterations=1)

            # Morphological Closing
            image_morph = cv2.morphologyEx(image_dilate, cv2.MORPH_CLOSE, kernel)

            # Blur
            image_blur = cv2.blur(image_morph, (5, 5))

            return image_blur

        # If dot_matrix is False, return the original image without additional processing
        return image

    def ocr_recognition(self, image):
        pre_recognition_image = image.copy()

        start_time = time.time()
        with torch.no_grad():
            easyocr_result = self.reader.recognize(
                pre_recognition_image, batch_size=5, paragraph=True
            )
        end_time = time.time()
        logger.info(f"EasyOCR read time: {(end_time-start_time):.2f} seconds")

        for result in easyocr_result:
            possible_date = result[1]
            valid_date = self.is_valid_date(possible_date)
        if valid_date:
            logger.success(f"Valid date: {valid_date}")
            return valid_date
        else:
            logger.warning(f"No valid date found. Found: {possible_date}")
            return possible_date

    def is_valid_date(self, possible_date):
        date_patterns = [
            r"(\d{1,2}/\d{1,2}/\d{2,4})",  # Matches dates in the format dd/mm/yyyy or dd/mm/yy
            r"(\d{1,2}-\d{1,2}-\d{2,4})",  # Matches dates in the format dd-mm-yyyy or dd-mm-yy
            r"(\d{1,2}\.\d{1,2}\.\d{2,4})",  # Matches dates in the format dd.mm.yyyy or dd.mm.yy
            r"(\d{2,4}-\d{1,2}-\d{1,2})",  # Matches dates in the format yyyy-mm-dd
            r"(\d{1,2}' '\d{2,4})",  # Matches dates in the format mm' 'yyyy or mm' 'yy
            r"(\d{2}/\d{4})",  # Matches dates in the format mm/yyyy
            r"(\d{2}-\d{4})",  # Matches dates in the format mm-yyyy
            r"(\d{2}.\d{4})",  # Matches dates in the format mm.yyyy
        ]

        for pattern in date_patterns:
            match = re.search(pattern, possible_date)
            if match:
                valid_date = match.group(1)
                # Remove all characters except for the matched date
                valid_date = re.sub(r"[^0-9/.\'-]", "", valid_date)
                valid_date = valid_date.replace("D", "0").replace("?", "2")

                ## Month check for mm?year
                if pattern in date_patterns[-4:]:
                    try:
                        first_two_digits = int(valid_date[:2])
                        if first_two_digits not in range(1, 13):
                            logger.warning("Month cannot be 0 nor exceed 12.")
                            return None
                    except ValueError:
                        logger.warning(f"Month expected 1->12. Got: {valid_date[:2]}")

                ## Day check for dd?(month)?year
                elif pattern in date_patterns[:4]:
                    try:
                        first_two_digits = int(valid_date[:2])
                        if first_two_digits not in range(1, 32):
                            logger.warning("Day cannot be 0 nor exceed 31.")
                            return None
                    except ValueError:
                        logger.warning(f"Day expected 1->31. Got: {valid_date[:2]}")

                return valid_date

        return None  # No valid date found

    def visualization(self, image, highest_confidence_box, date, conf):
        annotator = Annotator(image)
        annotator.box_label(highest_confidence_box, date)

        output_folder = "results-annotated"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        base_filename = "result-annotated.jpeg"
        output_path = os.path.join(output_folder, base_filename)

        count = 1
        while os.path.exists(output_path):
            base_filename = f"result-annotated_{count}.jpeg"
            output_path = os.path.join(output_folder, base_filename)
            count += 1

        cv2.imwrite(output_path, image)


if __name__ == "__main__":
    performOCR = OcrOnEdge()  # Instantiate OcrOnEdge once

    while True:
        file_path = input("Enter the file name (or 'exit' to quit): ")
        if file_path.lower() == "exit":
            break

        performOCR.read_config("ysco_config.json")

        # 1 | Read image
        image = cv2.imread(f"./sampleset/{file_path}")

        # 2 | OCR detection --> crop
        detection_result = performOCR.ocr_detection(image)

        if detection_result is None:
            continue  # Skip subsequent processing steps if no label is detected
        cropped_image, annotation_box, annotation_confidence = detection_result

        # 3 | Crop pre-processing
        pre_processed_img = performOCR.pre_processing(cropped_image)

        # 4 | OCR date recognition
        date = performOCR.ocr_recognition(pre_processed_img)

        # 5 | Visualization
        performOCR.visualization(image, annotation_box, date, annotation_confidence)
