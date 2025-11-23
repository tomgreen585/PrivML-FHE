import numpy as np
import struct
from fhe_mn_config import (
    DATASET_PATH, DATASET_SIZE, 
    TRAINING_IMAGE_FILE, TRAINING_LABEL_FILE, 
    TESTING_IMAGE_FILE, TESTING_LABEL_FILE
)

"""
mn_data_loader.py

Loads and prepares the MNIST dataset for training and evaluation. Normalizes images and reshapes to (H, W, C).
Supports configurable dataset truncation for faster prototyping. Returns arrays for direct conversion to PyTorch tensors or FHE encoding.
Expected image output shape: (N, 96, 96, 3)
"""

class ML_Data_Loader:
    """
    Class responsible for loading MNIST dataset from binary files.

    Attributes:
    images (np.ndarray): loaded image data.
    labels (np.ndarray): corresponding labels.
    dataset_path (str): path to the MNIST dataset directory.
    dataset_size (int): number of samples to load.
    """
    
    def __init__(self):
        self.images = []
        self.labels = []
        self.dataset_path = DATASET_PATH
        self.dataset_size = DATASET_SIZE
        self.training_image_file = TRAINING_IMAGE_FILE
        self.training_label_file = TRAINING_LABEL_FILE
        self.testing_image_file = TESTING_IMAGE_FILE
        self.testing_label_file = TESTING_LABEL_FILE
    
    def load_mnist_images(self, file):
        """
        Loads image data from a given MNIST image file.

        Args:
        file (str): filename of the MNIST image file.

        Returns:
        images (np array): image array of shape (N, H, W, 1).
        """
        print(f'[INFO] Loading images from: {self.dataset_path+file}')
        with open(self.dataset_path + file, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            images = images / 255.0
            images = np.expand_dims(images, axis=-1)
        return images
        
    def load_mnist_labels(self, file):
        """
        Loads image label from a given MNIST image file.

        Args:
        file (str): filename of the MNIST image file.

        Returns:
        images (np array): image array of shape (N,).
        """
        print(f'[INFO] Loading labels from: {self.dataset_path+file}')
        with open(self.dataset_path + file, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
    def loading_dataset(self):
        """
        Loads and merges the training and testing datasets.

        Returns:
        x_data (np array): images
        y_data (np array): labels
        """
        print(f'[INFO] Loading dataset: {self.dataset_path}')
        tr_img_file = self.training_image_file
        tr_lbl_file = self.training_label_file
        ts_img_file = self.testing_image_file
        ts_lbl_file = self.testing_label_file
            
        train_images = self.load_mnist_images(tr_img_file)   
        test_images = self.load_mnist_images(ts_img_file)
        train_labels = self.load_mnist_labels(tr_lbl_file)
        test_labels = self.load_mnist_labels(ts_lbl_file)
        
        all_images = np.concatenate((train_images, test_images), axis=0)
        all_labels = np.concatenate((train_labels, test_labels), axis=0)
        
        subset_size = min(self.dataset_size, len(all_images))
        all_images = all_images[:subset_size]
        all_labels = all_labels[:subset_size]
        
        self.images = all_images
        self.labels = all_labels
        
        return self.images, self.labels
        
    def display_dataset_metrics(self):
        """
        Displays stats about loaded/annotated dataset.
        """
        print(f'[INFO] Total images loaded: {len(self.images)}')
        if self.images is not None and self.labels is not None:
            print(f'[INFO] Image shape: {self.images.shape}')
            print(f'[INFO] Label shape: {self.labels.shape}')
        else:
            print(f'[ERR] Failed to display metrics')