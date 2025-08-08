import sys
import torch
import torch.nn as nn
from fc_config import (EN_KERNEL_SIZE, EN_PADDING, EN_ACT)

# model.py
# - Defines the machine learning model architecture (e.g. logistic regression, decision tree, neural net).
# - Keeps it modular so you can plug the model into both training.py and evaluation.py
# - Keep standard TenSEAL structure for ease of integration of FHE model

class ML_Model(nn.Module):
    
    ######################################## INITIAL MODEL ####################################
    
    def __init__(self):
        super(ML_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=EN_KERNEL_SIZE, padding=EN_PADDING), #(B, 32, 192, 192)
            EN_ACT,
            nn.MaxPool2d(2), #(B, 32, 96, 96)

            nn.Conv2d(32, 64, kernel_size=EN_KERNEL_SIZE, padding=EN_PADDING), #(B, 64, 96, 96)
            EN_ACT,
            nn.MaxPool2d(2), #(B, 64, 48, 48)

            nn.Conv2d(64, 128, kernel_size=EN_KERNEL_SIZE, padding=EN_PADDING), #(B, 128, 48, 48)
            EN_ACT,
            nn.MaxPool2d(2), #(B, 128, 24, 24)

            nn.Flatten(), #(B, 128 * 24 * 24)
            nn.Linear(128 * 24 * 24, 256),
            EN_ACT,
            nn.Linear(256, 4) #Predict bbox: [cx, cy, w, h]
        )
        
    def forward(self, x):
        return self.model(x)