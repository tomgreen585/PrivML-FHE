import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from fc_config import (SEED, TRAINING_SIZE, VALIDATION_SIZE, TESTING_SIZE)

"""
preprocessing.py

Performs data processing. Splits dataset into training, validation, and test sets using fixed sizes. Converts
image data into torch tensors and visualizes input-output image pairs.
"""

class ML_Preprocessing:
    """
    Preprocessing class for splitting and converting bounding-box datasets.

    Attributes:
    seed (int): random seed for reproducibility.
    training_size (float): size of dataset for training.
    validation_size (float): size of dataset for validation.
    testing_size (float): size of dataset for testing.
    processing_completed (bool): flag indicating completion of preprocessing.
    generated_test_data (bool): placeholder for future feature.
    data_augmentation_completed (bool): placeholder for augmentation logic.
    model_datasets_created (bool): flag indicating that split datasets were created.
    """
    
    def __init__(self):
        self.seed = SEED
        self.training_size = TRAINING_SIZE
        self.validation_size = VALIDATION_SIZE
        self.testing_size = TESTING_SIZE
        
        self.processing_completed = False
        self.generated_test_data = False
        self.data_augmentation_completed = False
        self.model_datasets_created = False
    
    def visualize_data(self, x_data, y_data):
        """
        Displays 5 sample input images with their target bounding boxes drawn.

        Args:
        x_data (np array): Image data in shape (N, H, W, 3)
        y_data (np array): Bounding box targets in shape (N, 4) [cx, cy, bw, bh]
        """
        print("[INFO] Visualizing data")
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
        
    def perform_dataset_split(self, x_data, y_data):
        """
        Splits the dataset into training, validation, and testing sets.

        Args:
        x_data (np array): Input images, shape (N, H, W, 3)
        y_data (np array): Target images, shape (N, 4)

        Returns:
        (x_train, y_train, x_val, y_val, x_test, y_test)
        """
        print("[INFO] Creating training, validation, testing datasets")
        
        np.random.seed(self.seed)
        dataset = np.arange(len(x_data))
        np.random.shuffle(dataset)
        
        plain_train = self.training_size
        plain_val = self.validation_size
        plain_test = self.testing_size
        
        #calcualtes sizes
        dataset_size = len(x_data)
        train_plain_size = int(plain_train * dataset_size)
        val_plain_size = int(plain_val * dataset_size)
        test_plain_size = int(plain_test * dataset_size)
        
        #split based on indices
        dataset_start = 0
        train_plain_set = dataset[dataset_start:dataset_start+train_plain_size]
        dataset_start += train_plain_size
        val_plain_set = dataset[dataset_start:dataset_start+val_plain_size]
        dataset_start += val_plain_size
        test_plain_set = dataset[dataset_start:dataset_start+test_plain_size]
        dataset_start += test_plain_size
        
        #index into data and convert to torch tensors
        x_plain_train = torch.from_numpy(x_data[train_plain_set]).float()
        y_plain_train = torch.from_numpy(y_data[train_plain_set]).float()
        x_plain_val = torch.from_numpy(x_data[val_plain_set]).float()
        y_plain_val = torch.from_numpy(y_data[val_plain_set]).float()
        x_plain_test = torch.from_numpy(x_data[test_plain_set]).float()
        y_plain_test = torch.from_numpy(y_data[test_plain_set]).float()
        
        self.model_datasets_created = True
        return x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test
        
    def preprocessing_steps(self, x_data, y_data, run_type):
        """
        Method that is called that runs the preprocessing pipeline. Visualizes image pairs 
        if `run_type` is "Testing". Splits data into train/val/test sets

        Args:
        x_data (np array): Input grayscale images
        y_data (np array): Target grayscale images
        run_type (str): Either "Testing" or "Final_Model"

        Returns:
        training, validation, and test sets
        """
        print("[INFO] Performing Preprocessing Steps")
        
        if run_type == "Testing":
            self.visualize_data(x_data, y_data)
        
        x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test = self.perform_dataset_split(x_data, y_data)
        
        print("[INFO] Finished Preprocessing Steps")
        self.processingcompleted = True
        
        return x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test