import sys
import torch
import time
from io import BytesIO
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fc_evaluation import Ml_Metrics_Evaluation, actual_prediction, actual_vs_real_prediction

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
        model = model.to(device).eval()

        x_test = x_test[:sample_count].to(device)
        y_test = y_test[:sample_count].to(device)

        x_test_input = x_test.permute(0, 3, 1, 2)  # (N, 3, 96, 96)

        total_start_time = time.time()
        with torch.no_grad():
            predictions = model(x_test_input)
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time

        preds_np = predictions.cpu().numpy()
        labels_np = y_test.cpu().numpy()
        images_np = x_test.cpu().numpy()

        self.ml_test_metrics_output(labels_np, preds_np, total_elapsed_time)

        for i in range(sample_count):
            if run_type == "Testing":
                print(f"\nSample {i+1}:")
                print("Predicted BBox:", preds_np[i])
                print("Ground Truth BBox:", labels_np[i])

            print("[INFO] Generating Predictions")
            pred_type = "JPred"
            self.draw_bounding_boxes(image=images_np[i], pred_box=preds_np[i], gt_box=None, run_type=run_type, p_type=pred_type)
            self.draw_bounding_boxes(images_np[i], preds_np[i], labels_np[i], run_type, pred_type)

        self.testing_completed = True

    def ml_test_metrics_output(self, true, pred, time):
        print("[INFO] Calculating Test Metrics")
        model_type = "Test"
        Ml_Metrics_Evaluation.reg_mean_squared_error(model_type, true, pred)
        Ml_Metrics_Evaluation.reg_mean_absolute_error(model_type, true, pred)
        Ml_Metrics_Evaluation.reg_mean_absolute_percentage_error(model_type, true, pred)
        Ml_Metrics_Evaluation.reg_r2_score(model_type, true, pred)
        Ml_Metrics_Evaluation.reg_explained_variance_score(model_type, true, pred)
        Ml_Metrics_Evaluation.function_time(model_type, time)
        print("[INFO] Finished Calculating Testing Metrics")
        
    def draw_bounding_boxes(self, image, pred_box, gt_box, run_type, p_type):
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
            actual_prediction.append(("Predicted Image", buf))
        else:
            actual_vs_real_prediction.append(("Actual (Red) vs Predicted (Green)", buf))
        if run_type == "Testing":
            plt.show()
        else:
            plt.close()