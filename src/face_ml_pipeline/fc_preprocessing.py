import numpy as np
import os
import cv2
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

# preprocessing.py
# - Generates target test data from original dataset (consisting of original images with red border set in order top, bottom, left, right)
# - Applies image data augmentation techniques
# - Performs dataset split to generate training, validation and testing datasets
# - preprocesseng stacked to shape (N, 96, 96, 3) -> (N, Height, width, channel)
# - Converted to tensor version -> still (N, Height, width, channel)

class ML_Preprocessing:
    def __init__(self):
        self.processing_completed = False
        self.generated_test_data = False
        self.data_augmentation_completed = False
        self.model_datasets_created = False
    
    def preview_dataset(self, x_data, y_data):
        for i in range(5):
            x_sample = (x_data[i] * 255).astype(np.uint8)
            bbox = y_data[i]
            h, w, _ = x_sample.shape
            cx, cy, bw, bh = bbox
            left = int((cx - bw / 2) * w)
            top = int((cy - bh / 2) * h)
            right = int((cx + bw / 2) * w)
            bottom = int((cy + bh / 2) * h)
            x_sample_drawn = x_sample.copy()
            print(f'[INFO] Sample y_data (bounding box [cx, cy, w, h]): {bbox}')
            cv2.rectangle(x_sample_drawn, (left, top), (right, bottom), (0, 255, 0), 2)
            plt.imshow(x_sample_drawn)
            plt.title("Sample x_data with Target Box (target y_data)")
            plt.axis('off')
            plt.show()
        self.generated_test_data = True
    
    # NOT IMPLEMENTED YET (Only will be integrated if needed)
    def perform_data_augmentation(self, image):
        print("[INFO] Performing data augmentation techniques")
        flipped_image = tf.flip_left_right(image)
        rotated_image = tf.image.rot90(image)
        zoom_factor = np.random.uniform(0.6, 0.8)
        zoomed_image = tf.image.central_crop(image, zoom_factor)
        
        self.data_augmentation_completed = True
        return flipped_image.numpy(), rotated_image.numpy(), zoomed_image.numpy()
        
    def perform_dataset_split(self, x_data, y_data, train_ratio, val_ratio, seed):
        print("[INFO] Creating training, validation, testing datasets")
        
        np.random.seed(seed)
        dataset = np.arange(len(x_data))
        np.random.shuffle(dataset)
        
        dataset_size = len(x_data)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        
        train_set = dataset[:train_size]
        val_set = dataset[train_size:train_size + val_size]
        test_set = dataset[train_size + val_size:]
        
        x_train = torch.from_numpy(x_data[train_set]).float()
        y_train = torch.from_numpy(y_data[train_set]).float()
        
        x_val = torch.from_numpy(x_data[val_set]).float()
        y_val = torch.from_numpy(y_data[val_set]).float()
        
        x_test = torch.from_numpy(x_data[test_set]).float()
        y_test = torch.from_numpy(y_data[test_set]).float()
        
        self.model_datasets_created = True
        return x_train, y_train, x_val, y_val, x_test, y_test   
        
    def preprocessing_steps(self, x_data, y_data, train_ratio, val_ratio, seed, run_type):
        print("[INFO] Performing Preprocessing Steps")
        
        if run_type == "Testing":
            self.preview_dataset(x_data, y_data)
        
        x_train, y_train, x_val, y_val, x_test, y_test = self.perform_dataset_split(x_data, y_data, train_ratio, val_ratio, seed)
        
        print("[INFO] Finished Preprocessing Steps")
        self.processingcompleted = True
        return x_train, y_train, x_val, y_val, x_test, y_test