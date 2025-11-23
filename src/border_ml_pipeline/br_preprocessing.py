import numpy as np
import torch
import matplotlib.pyplot as plt
from br_config import (SEED, TRAINING_SIZE, VALIDATION_SIZE, TESTING_SIZE)

"""
preprocessing.py

Performs data processing. Splits dataset into training, validation, and test sets using fixed sizes. Converts
image data into torch tensors and visualizes input-output image pairs.
"""

class ML_Preprocessing:
    """
    Preprocessing class for preparing grayscale border-detection datasets.

    Attributes:
    seed (int): Random seed for reproducibility.
    training_size (float): size of data for training.
    validation_size (float): size of data for validation.
    testing_size (float): size of data for testing.
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
        Visualizes the first 5 (input, target) image pairs using matplotlib.

        Args:
        x_data (np array): Input grayscale images with shape (N, H, W, 1)
        y_data (np array): Target (bordered) images, same shape as x_data
        """
        print("[INFO] Visualizing data")
        for i in range(5):
            x_sample = (x_data[i] * 255).astype(np.uint8)
            y_sample = (y_data[i] * 255).astype(np.uint8)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(x_sample.squeeze(), cmap='gray', vmin=0, vmax=255)
            axes[0].set_title("Input (No Border)")
            axes[1].imshow(y_sample.squeeze(), cmap='gray', vmin=0, vmax=255)
            axes[1].set_title("Target Output (Border)")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.show()
        
    def perform_dataset_split(self, x_data, y_data):
        """
        Splits the dataset into training, validation, and testing sets.

        Args:
        x_data (np array): Input images, shape (N, H, W, 1)
        y_data (np array): Target images, shape (N, H, W, 1)

        Returns:
        (x_train, y_train, x_val, y_val, x_test, y_test), all tensors of shape (B, H, W, 1)
        """
        print("[INFO] Creating training, validation, testing datasets")
        
        np.random.seed(self.seed)
        dataset = np.arange(len(x_data))
        np.random.shuffle(dataset)
        
        plain_train = self.training_size
        plain_val = self.validation_size
        plain_test = self.testing_size
        
        #calculate sizes
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