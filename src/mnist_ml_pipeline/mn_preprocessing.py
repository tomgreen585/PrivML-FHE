import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from mn_config import (
    SEED, AUGMENT_DATA,
    TRAINING_SIZE, VALIDATION_SIZE,
    TESTING_SIZE
)

"""
# preprocessing.py

Applies image data augmentation techniques. Performs dataset split to generate training, 
validation and testing datasets. Preprocesseng stacked to shape (N, 96, 96, 3) -> (N, Height, width, channel).
Converted to tensor version -> still (N, Height, width, channel)
"""

class ML_Preprocessing:
    """
    Preprocessing pipeline for image-based datasets. Handles visualization, 
    augmentation, and splitting of the input dataset into train, validation, 
    and test sets, and converts them into PyTorch tensors.

    Attributes:
    seed (int): random seed for dataset shuffling
    augment_data (bool): flag to apply augmentation techniques
    training_size (float): size of dataset for training
    validation_size (float): size of dataset for validation
    testing_size (float): size of dataset for testing
    processing_completed (bool): flag indicating completion of preprocessing.
    generated_test_data (bool): placeholder for future feature.
    data_augmentation_completed (bool): placeholder for augmentation logic.
    model_datasets_created (bool): flag indicating that split datasets were created.
    """
    
    def __init__(self):
        self.seed = SEED
        self.augment_data = AUGMENT_DATA
        self.training_size = TRAINING_SIZE
        self.validation_size = VALIDATION_SIZE
        self.testing_size = TESTING_SIZE
        
        self.completed_visualizing_data = False
        self.processing_completed = False
        self.data_augmentation_completed = False
        self.model_datasets_created = False
    
    def visualize_data(self, images, labels):
        """
        Displays a small sample of images with their corresponding labels.

        Args:
        images (np array): array of shape (N, H, W, C)
        labels (np array): corresponding labels
        """
        print("[INFO] Performing visualization from normal images")
        for i in range(5):
            x_sample = images[i]
            plt.imshow(x_sample, cmap='gray')
            plt.title(f"Sample x_data {labels[i]}")
            plt.axis('off')
            plt.show()   
        self.completed_visualizing_data = True
    
    def perform_data_augmentation(self, image):
        """
        Applies data augmentation to a single image -> horizontal flip, 90-degree rotation, central crop + resize

        Args:
        image (np.ndarray/tf.Tensor): input image in shape (H, W, C)

        Returns:
        flipped, rotated, and zoomed images as NumPy arrays
        """
        flipped_image = tf.image.flip_left_right(image)
        rotated_image = tf.image.rot90(image)
        zoom_factor = np.random.uniform(0.6, 0.8)
        zoomed_image = tf.image.central_crop(image, zoom_factor)
        zoomed = tf.image.resize(zoomed_image, (28, 28))
        f_image = flipped_image.numpy()
        r_image = rotated_image.numpy()
        z_image = zoomed.numpy()
        self.data_augmentation_completed = True
        return f_image, r_image, z_image
        
    def perform_dataset_split(self, x_data, y_data):
        """
        Splits the dataset into training, validation, and testing sets.

        Args:
        x_data (np array): input images
        y_data (np array): target labels

        Returns:
        (x_train, y_train, x_val, y_val, x_test, y_test)
        """
        
        print("[INFO] Creating training, validation, testing and encrypted datasets")
        
        np.random.seed(self.seed)
        dataset = np.arange(len(x_data))
        np.random.shuffle(dataset)
        
        plain_train = self.training_size
        plain_val = self.validation_size
        plain_test = self.testing_size
        
        #calculates sizes
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
        y_plain_train = torch.from_numpy(y_data[train_plain_set]).long()
        x_plain_val = torch.from_numpy(x_data[val_plain_set]).float()
        y_plain_val = torch.from_numpy(y_data[val_plain_set]).long()
        x_plain_test = torch.from_numpy(x_data[test_plain_set]).float()
        y_plain_test = torch.from_numpy(y_data[test_plain_set]).long()
        
        self.model_datasets_created = True
        return x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test
        
    def preprocessing_steps(self, images, labels, run_type):
        """
        Method that is called that runs the preprocessing pipeline. Visualizes image pairs 
        if `run_type` is "Testing". Performs dataset augmentation. Splits data into train/val/test sets

        Args:
        x_data (np array): Input grayscale images
        y_data (np array): Target grayscale images
        run_type (str): Either "Testing" or "Final_Model"

        Returns:
        training, validation, and test sets
        """
        print("[INFO] Performing Preprocessing Steps")
        
        if run_type == "Testing":
            self.visualize_data(images, labels)
        
        if self.augment_data:
            print("[INFO] Performing data augmentation techniques")
            augmented_images = []
            augmented_labels = []
            for img, lbl in zip(images, labels):
                flipped, rotated, zoomed = self.perform_data_augmentation(img)
                augmented_images.extend([flipped, rotated, zoomed])
                augmented_labels.extend([lbl] * 3)
            all_images = np.concatenate([images, np.array(augmented_images)], axis=0)
            all_labels = np.concatenate([labels, np.array(augmented_labels)], axis=0)
            
            x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test = self.perform_dataset_split(all_images, all_labels)
        else:
            x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test = self.perform_dataset_split(images, labels)
        
        print("[INFO] Finished Preprocessing Steps")
        self.processingcompleted = True
        
        return x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test