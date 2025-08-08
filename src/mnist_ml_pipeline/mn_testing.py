import sys
import torch
import time
from io import BytesIO
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mn_evaluation import Ml_Metrics_Evaluation, predicted_images

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

        x_test = x_test.to(device)
        y_test = y_test.to(device).long()

        if x_test.dim() == 3:
            x_test = x_test.unsqueeze(1)

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(x_test)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_test).sum().item()
            total = y_test.size(0)

        test_accuracy = 100 * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time

        predicted = predicted.cpu().numpy()
        y_test = y_test.cpu().numpy()
        self.ml_test_metrics_output(y_test, predicted, total_elapsed_time)

        x_np = x_test.cpu().numpy().squeeze(1)
        y_np = y_test
        preds_np = predicted

        for i in range(sample_count):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(x_np[i], cmap='gray')
            axes[0].set_title(f"Ground Truth: {y_np[i]}")
            axes[1].imshow(x_np[i], cmap='gray')
            axes[1].set_title(f"Predicted: {preds_np[i]}")
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
        Ml_Metrics_Evaluation.clas_accuracy_score(model_type, test, pred)
        Ml_Metrics_Evaluation.clas_precision_score(model_type, test, pred)
        Ml_Metrics_Evaluation.clas_recall_score(model_type, test, pred)
        Ml_Metrics_Evaluation.clas_f1_score(model_type, test, pred)
        Ml_Metrics_Evaluation.clas_confusion_matrix(model_type, test, pred)
        Ml_Metrics_Evaluation.function_time(model_type, time)
        print("[INFO] Finished Calculating Testing Metrics")