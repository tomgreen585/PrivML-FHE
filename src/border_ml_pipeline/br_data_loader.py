import numpy as np
import os
import cv2
from br_config import (DATASET_SIZE)

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
        return self.data
        
    def display_dataset_metrics(self):
        print(f'[INFO] Total images loaded: {len(self.data)}')
        if self.data:
            print(f'[INFO] Sample image shape: {self.data[0].shape}')