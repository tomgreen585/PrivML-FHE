import sys
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mn_evaluation import Ml_Metrics_Evaluation
from mn_config import (OPTIMIZER, LOSS_FUNCTION)

# training.py
# - Handles model training on plaintext data (or encrypted data if supported by your FHE framework).
# - Defines training loop, optimizer, loss functions, metrics, etc.
# - Logs progress and saves trained models (serialized weights or encrypted parameters).
# - Applies permute (0, 3, 1, 2) -> becomes (N, 3, 96, 96)
# - Sends performance metrics to evaluation.py to visualize/track

class ML_Training_Class:
    
    def __init__(self):
        self.optimizer = OPTIMIZER
        self.loss_function = LOSS_FUNCTION
        self.training_completed = False
        
    def train_model(self, model, x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate):
        print("[INFO] Starting Training Loop")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'[INFO] Using device: {device}')
        
        optimizer = self.optimizer(model.parameters(), lr=learning_rate)
        
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        total_start_time = time.time()
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            batch_count = 0.0
            training_correct = 0.0
            training_total = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
                if batch_x.dim() == 3:
                    batch_x = batch_x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                batch_count += 1
                _, predicted = torch.max(outputs, 1)
                training_total += batch_y.size(0)
                training_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = epoch_train_loss / batch_count
            train_losses.append(avg_train_loss)
            
            train_accuracy = 100 * training_correct / training_total
            train_accuracies.append(train_accuracy)
            
            model.eval()
            val_preds = []
            val_labels = []
            epoch_val_loss = 0.0
            validation_correct = 0.0
            validation_total = 0.0
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x, val_y = val_x.to(device), val_y.to(device).long()
                    val_x = val_x.unsqueeze(1)
                    val_outputs = model(val_x)
                    epoch_val_loss += self.loss_function(val_outputs, val_y).item()
                    _, predicted = torch.max(val_outputs, 1)
                    validation_total += val_y.size(0)
                    validation_correct += (predicted == val_y).sum().item()
                    val_labels.append(val_y.cpu())
                    val_preds.append(torch.argmax(val_outputs, dim=1).cpu())
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            val_accuracy = 100 * validation_correct / validation_total
            val_accuracies.append(val_accuracy)
            
            print("###########################################################################")       
            print(f'[Epoch {epoch+1}/{epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Train Accuracies: {train_accuracy:.4f}%')
            print(f'Val Loss: {avg_val_loss:.4f}')
            print(f'Val Accuracy: {val_accuracy:.4f}%')
        
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        self.ml_validation_metrics_output(val_labels, val_preds, total_elapsed_time, train_losses, val_losses)
        
        self.training_completed = True
        return model
    
    def ml_validation_metrics_output(self, test, pred, time, train_losses, val_losses):
        print("###########################################################################") 
        print("[INFO] Calculating Validation Metrics")
        model_type = "Validation"
        Ml_Metrics_Evaluation.clas_accuracy_score(model_type, test, pred)
        Ml_Metrics_Evaluation.clas_precision_score(model_type, test, pred)
        Ml_Metrics_Evaluation.clas_recall_score(model_type, test, pred)
        Ml_Metrics_Evaluation.clas_f1_score(model_type, test, pred)
        Ml_Metrics_Evaluation.clas_confusion_matrix(model_type, test, pred)
        Ml_Metrics_Evaluation.training_evaluation(model_type, train_losses)
        Ml_Metrics_Evaluation.train_val_loss_plot(model_type, train_losses, val_losses)
        Ml_Metrics_Evaluation.function_time(model_type, time)
        
        print("[INFO] Finished Calculating Validation Metrics")