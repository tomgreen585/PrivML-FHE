import numpy as np
import os
import cv2
from br_config import (DATASET_SIZE, IMAGE_SIZE, DATASET_PATH, BORDER_COLOR, BORDER_THICKNESS)

"""
br_data_loader.py

Provides functionality to load grayscale images from the dataset directory, 
preprocess them, and generate input/output pairs for training the model.

Each image is loaded, resized and normalized to simulate ground truth labels 
for a supervised learning task.
"""

class ML_Data_Loader:
    """
    Data loader class for grayscale border prediction task.
    """
    
    def __init__(self):
        self.image = []
        self.image_files = []
        self.x_data = []
        self.y_data = []
        self.dataset_path = DATASET_PATH
        self.image_size = IMAGE_SIZE
        self.dataset_size = DATASET_SIZE
        self.border_thickness = BORDER_THICKNESS
        self.border_color = BORDER_COLOR

    def load_dataset(self):
        """
        Loads grayscale images from the dataset directory. Filters files with `.jpg`, 
        `.jpeg`, or `.png` extensions. Resizes each image to (image_size x image_size).
        Normalizes pixel values to [0, 1].

        Populates:
        self.image: list of preprocessed grayscale images
        self.image_files: list of successfully loaded image paths
        """
        print(f'[INFO] Loading grayscale images from: {self.dataset_path}')
        
        for f in os.listdir(self.dataset_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(self.dataset_path, f)
                self.image_files.append(full_path)
        self.image_files = self.image_files[:self.dataset_size]
        
        for idx, img_path in enumerate(self.image_files):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'[ERR] Skipped unreadable image: {img_path}')
                continue
            if self.image_size and self.image_size > 0:
                img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            self.image.append(img)
        print(f'[INFO] Data Loaded {len(self.image)} grayscale images.')

    def generating_model_datasets(self):
        """
        Generates training data for the model. For each grayscale image it 
        creates an original image (input) and a target image with a rectangular border (output)

        Returns:
        x_data: original images with shape (N, H, W, 1)
        y_data: images with borders added, same shape as x_data
        """
        
        print("[INFO] Performing grayscale border generation")
        
        for img in self.image:
            original = img.copy()
            bordered = img.copy()

            bordered[:self.border_thickness, :] = self.border_color
            bordered[-self.border_thickness:, :] = self.border_color
            bordered[:, :self.border_thickness] = self.border_color
            bordered[:, -self.border_thickness:] = self.border_color
            
            self.x_data.append(original[..., np.newaxis])
            self.y_data.append(bordered[..., np.newaxis])
        
        self.x_data = np.stack(self.x_data)
        self.y_data = np.stack(self.y_data)

        self.generated_test_data = True
        return self.x_data, self.y_data

    def display_dataset_metrics(self):
        """
        Prints basic statistics about the loaded dataset.
        """
        print(f'[INFO] Total images loaded: {len(self.image)}')
        if self.image:
            print(f'[INFO] Sample grayscale image shape: {self.image[0].shape}')
