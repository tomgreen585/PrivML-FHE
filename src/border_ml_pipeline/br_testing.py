import torch
import time
import matplotlib.pyplot as plt
from io import BytesIO
from br_evaluation import Ml_Metrics_Evaluation, plaintext_predicted_images
from br_config import (SAMPLE_OUTPUT_COUNT)

# testing.py
# - For running predictions on new encrypted inputs after training
# - Useful for demonstraing practical use of the encrypted model in deployment
# - Sends performance metrics to evaluation.py to visualize/track

class ML_Testing_Class:
    """
    Class to manage the testing phase of the ML pipeline.

    Attributes:
    sample_count (int): Number of samples to visualize and evaluate.
    testing_completed (bool): Flag set after successful test execution.
    """
    
    def __init__(self):
        self.sample_count = SAMPLE_OUTPUT_COUNT
        self.testing_completed = False
        
    def test_model(self, model, x_test, y_test, run_type):
        """
        Runs the model on test data and visualizes predictions.

        Args:
        model (torch.nn.Module): The trained model to evaluate.
        x_test (torch.Tensor): Test inputs, shape (N, H, W, 1).
        y_test (torch.Tensor): Ground truth targets, shape (N, H, W, 1).
        run_type (str): "Testing" or "Final_Model" — controls visualization.
        """
        print("[INFO] Starting Testing")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'[INFO] Using device: {device}')

        total_start_time = time.time()

        x_test = x_test[:self.sample_count].to(device)
        y_test = y_test[:self.sample_count].to(device)

        x_test = x_test.permute(0, 3, 1, 2).contiguous()

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            test_predictions = model(x_test)

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time

        test_preds = test_predictions.squeeze(1).cpu()
        test_labels = y_test.permute(0, 3, 1, 2).squeeze(1).cpu()
        flatten_preds = test_preds.numpy().reshape((test_preds.shape[0], -1))
        flatten_labels = test_labels.numpy().reshape((test_labels.shape[0], -1))
        
        self.ml_test_metrics_output(flatten_preds, flatten_labels, total_elapsed_time)

        x_np = x_test.cpu().numpy()
        x_np = (x_np * 255).astype("uint8")
        y_np = y_test.cpu().numpy()
        y_np = (y_np * 255).astype("uint8")
        preds_np = test_predictions.permute(0, 2, 3, 1).cpu().numpy()
        preds_np = (preds_np * 255).clip(0, 255).astype("uint8")

        for i in range(self.sample_count):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(x_np[i].squeeze(), cmap='gray')
            axes[0].set_title("Input (No Border)")
            axes[1].imshow(preds_np[i].squeeze(), cmap='gray')
            axes[1].set_title("Predicted Output")
            axes[2].imshow(y_np[i].squeeze(), cmap='gray')
            axes[2].set_title("Ground Truth (Bordered)")

            title = "Plaintext Predictions"
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plaintext_predicted_images.append(("Plaintext Predicted Image", buf))
            if run_type == "Testing":
                plt.show()
            else:
                plt.close()

        self.testing_completed = True

    def ml_test_metrics_output(self, test, pred, time):
        """
        Calculation and plotting of test-time regression metrics using evaluation.py.

        Args:
        test (np array): Flattened predicted outputs.
        pred (np array): Flattened true outputs.
        time (float): Time taken for the full inference pass.
        """
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