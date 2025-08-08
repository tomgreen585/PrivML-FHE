import sys
import torch
import torch.nn as nn
from mn_config import (EN_KERNEL_SIZE, EN_STRIDE, EN_PADDING, DE_KERNEL_SIZE, DE_STRIDE, DE_PADDING)

# model.py
# - Defines the machine learning model architecture (e.g. logistic regression, decision tree, neural net).
# - Keeps it modular so you can plug the model into both training.py and evaluation.py
# - Keep standard TenSEAL structure for ease of integration of FHE model

class ML_Model(nn.Module):
    
    ######################################## INITIAL MODEL ####################################
    
    import torch
import torch.nn as nn

class ML_Model(nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ML_Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 → 14x14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 → 7x7
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7 → 3x3
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
