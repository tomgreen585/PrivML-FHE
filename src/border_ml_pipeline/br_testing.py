import sys
import torch
import time
from io import BytesIO
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from br_evaluation import Ml_Metrics_Evaluation, predicted_images

# testing.py
# - For running predictions on new encrypted inputs after training
# - Useful for demonstraing practical use of the encrypted model in deployment
# - Sends performance metrics to evaluation.py to visualize/track

class ML_Testing_Class:
    
    def __init__(self):
        self.testing_completed = False
        
    def test_model(self, model, x_test, y_test, sample_count, run_type):
        print("[INFO] Starting Testing")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'[INFO] Using device: {device}')
        
        total_start_time = time.time()
        
        model = model.to(device)
        model.eval()
        
        x_test = x_test[:sample_count].to(device)
        y_test = y_test[:sample_count].to(device)
        
        x_test_nchw = x_test.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            test_predictions = model(x_test_nchw)
         
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
           
        test_preds = test_predictions.permute(0, 3, 1, 2).cpu()
        test_labels = y_test.permute(0, 3, 1, 2).cpu()
        test_preds = test_preds.numpy()
        test_labels = test_labels.numpy()
        flatten_preds = test_preds.reshape((test_preds.shape[0], -1))
        flatten_labels = test_labels.reshape((test_labels.shape[0], -1))
        self.ml_test_metrics_output(flatten_preds, flatten_labels, total_elapsed_time)
        
        x_np = x_test.cpu().numpy()
        y_np = y_test.cpu().numpy()
        preds_np = test_predictions.permute(0, 2, 3, 1).cpu().numpy()
        
        x_np = (x_np * 255).astype("uint8")
        y_np = (y_np * 255).astype("uint8")
        preds_np = (preds_np * 255).clip(0, 255).astype("uint8")
        for i in range(sample_count):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(x_np[i])
            axes[0].set_title("Input (No Border)")
            axes[1].imshow(preds_np[i])
            axes[1].set_title("Predicted Output")
            axes[2].imshow(y_np[i])
            axes[2].set_title("Ground Truth (Bordered)")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            predicted_images.append(("Predicted Image", buf))
            if run_type == "Testing":
                plt.show()
            else:
                plt.close()
        
        self.testing_completed = True
        
    def ml_test_metrics_output(self, test, pred, time):
        print("[INFO] Calculating Test Metrics")
        model_type = "Test"
        Ml_Metrics_Evaluation.reg_mean_squared_error(model_type, test, pred)
        Ml_Metrics_Evaluation.reg_mean_absolute_error(model_type, test, pred)
        Ml_Metrics_Evaluation.reg_mean_absolute_percentage_error(model_type, test, pred)
        Ml_Metrics_Evaluation.reg_r2_score(model_type, test, pred)
        Ml_Metrics_Evaluation.reg_explained_variance_score(model_type, test, pred)
        Ml_Metrics_Evaluation.reg_prediction_vs_actual_plot(model_type, test, pred)
        Ml_Metrics_Evaluation.reg_residual_plot(model_type, test, pred)
        Ml_Metrics_Evaluation.reg_residual_histogram(model_type, test, pred)
        Ml_Metrics_Evaluation.function_time(model_type, time)
        print("[INFO] Finished Calculating Testing Metrics")