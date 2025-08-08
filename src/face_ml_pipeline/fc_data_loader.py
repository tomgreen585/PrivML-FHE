import numpy as np
import os
import cv2
import face_recognition
from fc_config import (DATASET_SIZE)

# data_loader.py
# - Loads and precprocesses datasets (plaintext format). 
# - Applies scaling/normalization and optionally converts to encrypted format. 
# - Performs train/test split and returns data in usable format (e.g. NumPy arrays, tensors, encrytped vectors).
# - Metrics display numpy arrays of shape (96, 96, 3) -> (Height, width, channel)

class ML_Data_Loader:
    def __init__(self, dataset_path, image_size):
        self.data = []
        self.image_files = []
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.dataset_size = DATASET_SIZE
    
    def loading_dataset(self):
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
            self.data.append(img)
            
        print(f'[INFO] Data Loaded {len(self.data)} images.')
    
    def get_dataset_with_bounding_boxes(self):
        print(f'[INFO] Generating bounding boxes from: {self.dataset_path}')
        x_data = []
        y_data = []

        for img_path in self.image_files:
            index = self.image_files.index(img_path)
            image_data = self.data[index]
            if image_data is None:
                print(f'[ERR] No image data for {img_path}, skipping.')
                continue

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
            x_data.append(image_data)
            y_data.append(bbox)

        x = np.array(x_data, dtype=np.float32)
        y = np.array(y_data, dtype=np.float32)

        print(f'[INFO] Generated {len(x)} samples with bounding boxes.')
        return x, y
        
    def display_dataset_metrics(self):
        print(f'[INFO] Total images loaded: {len(self.data)}')
        if self.data:
            print(f'[INFO] Sample image shape: {self.data[0].shape}')