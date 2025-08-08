import numpy as np
import os
import cv2
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from mn_config import (BORDER_THICKNESS, BORDER_COLOR)

# preprocessing.py
# - Generates target test data from original dataset (consisting of original images with red border set in order top, bottom, left, right)
# - Applies image data augmentation techniques
# - Performs dataset split to generate training, validation and testing datasets
# - preprocesseng stacked to shape (N, 96, 96, 3) -> (N, Height, width, channel)
# - Converted to tensor version -> still (N, Height, width, channel)

class ML_Preprocessing:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        
        self.processing_completed = False
        self.data_augmentation_completed = False
        self.model_datasets_created = False
    
    def visualize_data(self, images, labels):
        print("[INFO] Performing border generation from normal images")
        for i in range(5):
            x_sample = images[i]
            plt.imshow(x_sample, cmap='gray')
            plt.title(f"Sample x_data {labels[i]}")
            plt.axis('off')
            plt.show()
    
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
        
    def preprocessing_steps(self, images, labels, train_ratio, val_ratio, seed, run_type):
        print("[INFO] Performing Preprocessing Steps")
        
        if run_type == "Testing":
            self.visualize_data(images, labels)
        
        x_train, y_train, x_val, y_val, x_test, y_test = self.perform_dataset_split(images, labels, train_ratio, val_ratio, seed)
        
        print("[INFO] Finished Preprocessing Steps")
        self.processingcompleted = True
        return x_train, y_train, x_val, y_val, x_test, y_test