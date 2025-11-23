import torch.nn as nn
from fhe_mn_config import (
    PLAIN_HIDDEN, PLAIN_OUTPUT,
    PLAIN_KERNEL_SIZE, PLAIN_STRIDE,
    PLAIN_PADDING, PLAIN_DROPOUT
)

"""
plain_model.py

Defines the machine learning model architecture (e.g. logistic regression, decision tree, neural net). Keeps it modular 
so you can plug the model into both training.py and evaluation.py. Keep standard TenSEAL structure for ease of integration of FHE model.
x*x is the square activation functions that is indicated in FHE model. conv1 output: 12x8x8
"""

class ML_Model(nn.Module):
    def __init__(self, hidden=PLAIN_HIDDEN, output=PLAIN_OUTPUT):
        super(ML_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=PLAIN_KERNEL_SIZE, stride=PLAIN_STRIDE, padding=PLAIN_PADDING)
        self.dropout = nn.Dropout(PLAIN_DROPOUT)
        self.fc1 = nn.Linear(768, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x 
        x = self.dropout(x)
        x = x.view(-1, 768)
        x = self.fc1(x)
        x = x * x
        x = self.dropout(x)
        x = self.fc2(x)
        return x
