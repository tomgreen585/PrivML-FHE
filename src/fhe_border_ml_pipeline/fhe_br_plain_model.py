import torch.nn as nn
from fhe_br_config import (PLAIN_CONV1_KERNEL_SIZE, PLAIN_CONV2_KERNEL_SIZE, PLAIN_CONV1_PADDING)

"""
model.py

Defines the machine learning model architecture (e.g. logistic regression, 
decision tree, neural net). Keeps it modular so you can plug the model into both training.py and evaluation.py. 
Keep standard TenSEAL structure for ease of integration of FHE model.
"""

class ML_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=PLAIN_CONV1_KERNEL_SIZE, padding=PLAIN_CONV1_PADDING)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=PLAIN_CONV2_KERNEL_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x
        x = self.conv2(x)
        return x
