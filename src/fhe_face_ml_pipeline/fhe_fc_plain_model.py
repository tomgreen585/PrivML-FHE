import torch.nn as nn
from fhe_fc_config import (
    PLAIN_HIDDEN, PLAIN_OUTPUT, 
    PLAIN_KERNEL_SIZE, PLAIN_STRIDE, 
    PLAIN_PADDING, PLAIN_DROPOUT
)

"""
model.py

Defines the machine learning model architecture (e.g. logistic regression, decision tree, neural net).Keeps it modular 
so you can plug the model into both training.py and evaluation.py. Keep standard TenSEAL structure for ease of integration of FHE model
"""

class ML_Model(nn.Module):
    def __init__(self, hidden=PLAIN_HIDDEN, output=PLAIN_OUTPUT, dropout=PLAIN_DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=PLAIN_KERNEL_SIZE, stride=PLAIN_STRIDE, padding=PLAIN_PADDING)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64, hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
