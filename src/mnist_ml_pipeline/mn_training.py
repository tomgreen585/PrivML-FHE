import torch
import time
from mn_evaluation import Ml_Metrics_Evaluation
from mn_config import (
    PLAIN_OPTIMIZER, PLAIN_EPOCHS, 
    PLAIN_LEARNING_RATE, PLAIN_BATCH_SIZE, 
    PLAIN_LOSS_FUNCTION
)

"""
training.py

Handles training of the ML model using plaintext data. Defines the full training loop with
optimizer, loss computation, logging, and metric evalaution.
"""

class ML_Training_Class:
    """
    Manages training of ML model on plaintext data and evaluating its performance.

    Attributes:
    optimizer (torch.optim.Optimizer): optimizer function
    epochs (int): number of training epochs.
    learning_rate (float): learning rate for optimizer.
    batch_size (int): batch size for training.
    loss_function (nn.Module): loss function
    training_completed (bool): flag indicating training completion.
    """
    
    def __init__(self):
        self.optimizer = PLAIN_OPTIMIZER
        self.epochs = PLAIN_EPOCHS
        self.learning_rate = PLAIN_LEARNING_RATE
        self.batch_size = PLAIN_BATCH_SIZE
        self.loss_function = PLAIN_LOSS_FUNCTION
        self.training_completed = False
        
    def train_model(self, model, x_train, y_train, x_val, y_val):
        """
        Executes the training loop for the provided model.

        Args:
        model (torch.nn.Module): model to train.
        x_train (torch.Tensor): training input tensor.
        y_train (torch.Tensor): training ground truth tensor.
        x_val (torch.Tensor): validation input tensor.
        y_val (torch.Tensor): validation ground truth tensor.

        Returns:
        model (torch.nn.Module): trained model.
        """
        print("[INFO] Starting Training Loop")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'[INFO] Using device: {device}')
        
        optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
        
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        total_start_time = time.time()
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0.0
            batch_count = 0.0
            training_correct = 0.0
            training_total = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
                if batch_x.dim() == 4 and batch_x.shape[-1] == 1:
                    batch_x = batch_x.permute(0, 3, 1, 2)
                if batch_x.dim() == 3:
                    batch_x = batch_x.unsqueeze(1)
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
                    if val_x.dim() == 4 and val_x.shape[-1] == 1:
                        val_x = val_x.permute(0, 3, 1, 2)
                    if val_x.dim() == 3:
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
            print(f'[Epoch {epoch+1}/{self.epochs}]')
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
        """
        Sends training metrics to the evaluation module.

        Args:
        test (np array): flattened ground truth tensors.
        pred (np array): flattened predicted outputs.
        time (float): total time elapsed during training.
        train_losses (List): training loss per epoch.
        val_losses (List): validation loss per epoch.
        """
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