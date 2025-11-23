import torch
import torch.nn as nn
from mn_config import (
    HIDDEN, OUTPUT, CONV_KERNEL_SIZE, 
    CONV_STRIDE_SIZE, CONV_PADDING_SIZE, DROPOUT, 
    POOL_KERNEL_SIZE, POOL_STRIDE_SIZE
)

"""
model.py

Defines the machine learning model architecture (e.g. logistic regression, 
decision tree, neural net). Keeps it modular so you can plug the model into both training.py and evaluation.py. 
Keep standard TenSEAL structure for ease of integration of FHE model.
"""

class ML_Model(nn.Module):
    def __init__(self, hidden=HIDDEN, output=OUTPUT):
        super(ML_Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE_SIZE, padding=CONV_PADDING_SIZE),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE_SIZE),  # 28x28 → 14x14
            nn.Dropout(DROPOUT)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE_SIZE, padding=CONV_PADDING_SIZE),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE_SIZE),  # 14x14 → 7x7
            nn.Dropout(DROPOUT)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE_SIZE, padding=CONV_PADDING_SIZE),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE_SIZE),  # 7x7 → 3x3
            nn.Dropout(DROPOUT)
        )

        self.fc1 = nn.Linear(32 * 3 * 3, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)  # Flatten (B, C, H, W) → (B, C*H*W)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
