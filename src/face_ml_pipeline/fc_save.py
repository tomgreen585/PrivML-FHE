import sys
import os
import torch
from datetime import datetime

# save.py
# - Saves the model created when Final_Model is specified during runtime
# - Outputs the saved model.pth to directory

class ML_Saving_Model:
    def __init__(self):
        self.completed_saving_model = False
    
    def save_ml_model(self, model):
        print("[INFO] Saving ML Model")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{timestamp}.pth"
        
        torch.save(model.state_dict(), model_path)
        
        print(f'[INFO] Saved model to {model_path}')
        
        self.completed_saving_model = True