import sys
import os
import torch
from datetime import datetime

"""
save.py

Saves the model created when Final_Model is specified during runtime. 
Uses timestamped filename in the /models directory. Uses torch.save() to save state_dict.
"""

class ML_Saving_Model:
    def __init__(self):
        self.completed_saving_model = False
    
    def save_ml_model(self, model):
        """
        Saves the model's state dictionary to a timestamped `.pth` file.

        Args:
        model (torch.nn.Module): The trained PyTorch model to be saved.

        Output:
        Saves model file to ./models/YYYYMMDD_HHMMSS.pth
        """
        print("[INFO] Saving ML Model")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{timestamp}.pth"
        
        torch.save(model.state_dict(), model_path)
        
        print(f'[INFO] Saved model to {model_path}')
        
        self.completed_saving_model = True