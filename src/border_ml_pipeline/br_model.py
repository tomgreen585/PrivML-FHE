import torch.nn as nn
from br_config import (EN_KERNEL_SIZE, EN_STRIDE, EN_PADDING, DE_KERNEL_SIZE, DE_STRIDE, DE_PADDING)

"""
model.py

Defines the machine learning model architecture (e.g. logistic regression, 
decision tree, neural net). Keeps it modular so you can plug the model into both training.py and evaluation.py. 
Keep standard TenSEAL structure for ease of integration of FHE model.
"""

class ML_Model(nn.Module):
    
    ######################################## INITIAL MODEL ####################################
    
    def __init__(self):
        super(ML_Model, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 8, kernel_size=EN_KERNEL_SIZE, stride=EN_STRIDE, padding=EN_PADDING) #encoding: 192x192->96x96
        self.conv2 = nn.Conv2d(8, 16, kernel_size=EN_KERNEL_SIZE, stride=EN_STRIDE, padding=EN_PADDING) #encoding: 96x96->48x48
        
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=DE_KERNEL_SIZE, stride=DE_STRIDE, padding=DE_PADDING) #decoding: 48x48->96x96
        self.deconv2 = nn.ConvTranspose2d(8, 1, kernel_size=DE_KERNEL_SIZE, stride=DE_STRIDE, padding=DE_PADDING) #decoding: 96x96->192x192
        
    def square_activation(self, x):
        return x * x
    
    def forward(self, x):
        x = self.square_activation(self.conv1(x))
        x = self.square_activation(self.conv2(x))
        x = self.square_activation(self.deconv1(x))
        x = self.square_activation(self.deconv2(x))
        return x