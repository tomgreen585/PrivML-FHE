import numpy as np
import os
import struct
import cv2
from mn_config import (DATASET_SIZE)

# data_loader.py
# - Loads mnist datasets (plaintext format).
# - Performs train/test split and returns data in usable format (e.g. NumPy arrays, tensors, encrytped vectors).
# - Metrics display numpy arrays of shape (96, 96, 3) -> (Height, width, channel)

class ML_Data_Loader:
    def __init__(self, dataset_path):
        self.images = []
        self.labels = []
        self.dataset_path = dataset_path
    
    def load_mnist_images(self, file):
        print(f'[INFO] Loading images from: {self.dataset_path+file}')
        with open(self.dataset_path + file, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return images
        
    def load_mnist_labels(self, file):
        print(f'[INFO] Loading labels from: {self.dataset_path+file}')
        with open(self.dataset_path + file, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
    def loading_dataset(self):
        print(f'[INFO] Loading dataset: {self.dataset_path}')
        tr_img_file = "train-images.idx3-ubyte"
        tr_lbl_file = "train-labels.idx1-ubyte"
        ts_img_file = "t10k-images.idx3-ubyte"
        ts_lbl_file = "t10k-labels.idx1-ubyte"
            
        train_images = self.load_mnist_images(tr_img_file)   
        test_images = self.load_mnist_images(ts_img_file)
        train_labels = self.load_mnist_labels(tr_lbl_file)
        test_labels = self.load_mnist_labels(ts_lbl_file)
        
        all_images = np.concatenate((train_images, test_images), axis=0)
        all_labels = np.concatenate((train_labels, test_labels), axis=0)
        
        subset_size = min(DATASET_SIZE, len(all_images))  # just in case DATASET_SIZE > 70000
        all_images = all_images[:subset_size]
        all_labels = all_labels[:subset_size]
        
        self.images = all_images
        self.labels = all_labels
        
        return self.images, self.labels
        
    def display_dataset_metrics(self):
        if self.images is not None and self.labels is not None:
            print(f'[INFO] Image shape: {self.images.shape}')
            print(f'[INFO] Label shape: {self.labels.shape}')
        else:
            print(f'[ERR] Failed to display metrics')