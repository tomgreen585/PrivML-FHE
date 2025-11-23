import numpy as np
import os
import cv2
import face_recognition
from fc_config import (DATASET_PATH, DATASET_SIZE, IMAGE_SIZE)

"""
data_loader.py

Loads face images and automatically annoteates them using the 'face_recognition' library
to extract bounding box coordinates. The output is used to to train a regression model that predicts
normalised bounding box coordinates (cx, cy, bw, bh) for detected faces.
"""

class ML_Data_Loader:
    """
    Loads and prepares a dataset of face images with bounding box annotations.

    Attributes:
    image (List): loaded and normalized face images.
    image_files (List): file paths of all loaded images.
    x_data (np array): normalized image data for model input (N, H, W, 3).
    y_data (np array): normalized bounding boxes [cx, cy, bw, bh] (N, 4).
    dataset_path (str): path to image dataset.
    dataset_size (int): maximum number of images to load.
    image_size (int): target image size (assumes square).
    """
    
    def __init__(self):
        self.image = []
        self.image_files = []
        self.x_data = []
        self.y_data = []
        self.dataset_path = DATASET_PATH
        self.dataset_size = DATASET_SIZE
        self.image_size = IMAGE_SIZE

    def load_dataset(self):
        """
        Loads and preprocesses face images from the dataset directory. Filters image files 
        with extensions: .jpg, .jpeg, .png. Resizes each image to (image_size x image_size).
        Converts BGR to RGB. Normalizes pixel values to [0, 1].

        Populates:
        self.image: list of images
        self.image_files: list of file paths
        """
        print(f'[INFO] Loading images from: {self.dataset_path}')

        for f in os.listdir(self.dataset_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(self.dataset_path, f)
                self.image_files.append(full_path)
        self.image_files = self.image_files[:self.dataset_size]

        for idx, img_path in enumerate(self.image_files):
            img = cv2.imread(img_path)
            if img is None:
                print(f'[ERR] Skipped unreadable image: {img_path}')
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.image_size and self.image_size > 0:
                img = cv2.resize(img, (self.image_size, self.image_size))
            img = img.astype(np.float32) / 255.0
            self.image.append(img)
        print(f'[INFO] Data Loaded {len(self.image)} images.')

    def generating_model_datasets(self):
        """
        Generates bounding box labels for loaded face images using face detection library.
        Extracts the bounding box of the first face found. Normalizes the coordinates (cx, cy, bw, bh)

        Returns:
        x_data: shape (N, H, W, 3)
        y_data: shape (N, 4)
        """
        print(f'[INFO] Generating annotated images with bounding boxes.')
        original_images = []
        annotated_images = []

        for img_path in self.image_files:
            index = self.image_files.index(img_path)
            image_data = self.image[index]
            if image_data is None:
                print(f'[ERR] No image data for {img_path}, skipping.')
                continue

            #convert image to uint8 for face_recognition
            image_uint8 = (image_data * 255).astype(np.uint8)
            face_locations = face_recognition.face_locations(image_uint8)

            if not face_locations:
                print(f'[WARN] No face found in {img_path}, skipping.')
                continue

            top, right, bottom, left = face_locations[0]
            h, w, _ = image_uint8.shape
            cx = ((left + right) / 2) / w
            cy = ((top + bottom) / 2) / h
            bw = (right - left) / w
            bh = (bottom - top) / h
            bbox = [cx, cy, bw, bh]

            original_images.append(image_data)
            annotated_images.append(bbox)

        self.x_data = np.array(original_images, dtype=np.float32)
        self.y_data = np.array(annotated_images, dtype=np.float32)

        print(f'[INFO] Generated {len(self.y_data)} annotated image samples.')
        return self.x_data, self.y_data

    def display_dataset_metrics(self):
        """
        Displays stats about loaded/annotated dataset.
        """
        print(f'[INFO] Total images loaded: {len(self.image)}')
        if self.x_data is not None and self.y_data is not None:
            print(f'[INFO] x_data shape: {self.x_data.shape}')
            print(f'[INFO] y_data shape: {self.y_data.shape}')
        else:
            print(f'[ERR] Failed to display metrics')