import torch
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from fhe_fc_evaluation import Ml_Metrics_Evaluation, actual_prediction_plaintext, actual_vs_real_prediction_plaintext
from fhe_fc_config import (SAMPLE_OUTPUT_COUNT)

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
        Runs inference on test data and visualizes predictions.

        Args:
        model (torch.nn.Module): trained bounding box model
        x_test (torch.Tensor): input images (N, H, W, C)
        y_test (torch.Tensor): ground truth boxes (N, 4)
        run_type (str): "Testing" or "Final_Model"
        """
        print("[INFO] Starting Testing")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'[INFO] Using device: {device}')
        
        total_start_time = time.time()
        
        x_test = x_test[:self.sample_count].to(device)
        y_test = y_test[:self.sample_count].to(device)

        x_test = x_test.permute(0, 3, 1, 2)

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(x_test)
        
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time

        x_np = x_test.cpu().numpy()
        y_np = y_test.cpu().numpy()
        preds_np = outputs.cpu().numpy()
        
        self.ml_test_metrics_output(y_np, preds_np, total_elapsed_time)
        
        for i in range(self.sample_count):
            if run_type == "Testing":
                print("###########################################################################")
                print(f"Sample {i+1}:")
                print("Predicted BBox:", preds_np[i])
                print("Ground Truth BBox:", y_np[i])
                
            print(f"[INFO] Generating predictions")
            pred_type = "JPred"
            self.visualize_predictions(image=x_np[i], pred_box=preds_np[i], gt_box=None, run_type=run_type, p_type=pred_type)
            self.visualize_predictions(x_np[i], preds_np[i], y_np[i], run_type, pred_type)
        
        self.testing_completed = True

    def ml_test_metrics_output(self, true, pred, time):
        """
        Calls evaluation module to compute performance metrics.

        Args:
        true (np array): ground truth bounding boxes (N, 4)
        pred (np array): predicted bounding boxes (N, 4)
        time (float): time taken for full inference run
        """
        print("[INFO] Calculating Test Metrics")
        model_type = "Test"
        Ml_Metrics_Evaluation.reg_mean_squared_error(model_type, true, pred)
        Ml_Metrics_Evaluation.reg_mean_absolute_error(model_type, true, pred)
        Ml_Metrics_Evaluation.reg_mean_absolute_percentage_error(model_type, true, pred)
        Ml_Metrics_Evaluation.reg_r2_score(model_type, true, pred)
        Ml_Metrics_Evaluation.reg_explained_variance_score(model_type, true, pred)
        Ml_Metrics_Evaluation.function_time(model_type, time)
        print("[INFO] Finished Calculating Testing Metrics")
        
    def visualize_predictions(self, image, pred_box, gt_box, run_type, p_type):
        """
        Draws and saves (or displays) predicted and optionally true bounding boxes.

        Args:
        image (np array): image in CHW or HWC format.
        pred_box (np array): predicted [cx, cy, bw, bh].
        gt_box (np array or None): ground truth [cx, cy, bw, bh], or None.
        run_type (str): "Testing" or "Final_Model"
        p_type (str): type of image ("JPred" = pred only, else = pred vs ground truth).
        """
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        h, w = image.shape[:2]
        px, py, pw, ph = pred_box
        px, py, pw, ph = px * w, py * h, pw * w, ph * h
        top_left_pred = (px - pw / 2, py - ph / 2)
        pred_rect = patches.Rectangle(top_left_pred, pw, ph, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(pred_rect)

        if gt_box is not None:
            gx, gy, gw, gh = gt_box
            gx, gy, gw, gh = gx * w, gy * h, gw * w, gh * h
            top_left_gt = (gx - gw / 2, gy - gh / 2)
            gt_rect = patches.Rectangle(top_left_gt, gw, gh, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(gt_rect)

        title = "Green: Predicted"
        if gt_box is not None:
            title += " | Red: Ground Truth"
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        if p_type == "JPred":
            actual_prediction_plaintext.append(("Predicted Image", buf))
        else:
            actual_vs_real_prediction_plaintext.append(("Actual (Red) vs Predicted (Green)", buf))
        if run_type == "Testing":
            plt.show()
        else:
            plt.close()

