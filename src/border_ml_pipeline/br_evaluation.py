import os
import csv
from io import BytesIO
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image

from br_config import (
    DATASET_PATH, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO, SEED,
    SAMPLE_OUTPUT_COUNT, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    OPTIMIZER, LOSS_FUNCTION, DATASET_SIZE
)

time_outputs = []
training_evaluation = []
train_val_evaluation = []
regression_string_evalaution = []
prediction_vs_plot = []
residual_plot = []
residual_histogram = []
predicted_images = []

# evaluation.py
# - Performs basic regression evaluation such as MSE, MAE
# - Generates plots to visually evaluate model
# - Append metrics to a continuously updated .csv -> continuously track performance
# - Generates a new .pdf for each model run to visualize plots -> continuously track performance

class Ml_Metrics_Evaluation:
    
    ################### EVALUATION METHODS #############################
    
    @staticmethod
    def reg_mean_squared_error(modeltype, test, pred):
        mse_sc = mean_squared_error(test, pred)
        regression_string_evalaution.append((modeltype, "MSE", mse_sc))
    
    @staticmethod
    def reg_mean_absolute_error(modeltype, test, pred):
        mae_sc = mean_absolute_error(test, pred)
        regression_string_evalaution.append((modeltype, "MAE", mae_sc))
    
    @staticmethod
    def reg_mean_absolute_percentage_error(modeltype, test, pred):
        mape_sc = mean_absolute_percentage_error(test, pred)
        regression_string_evalaution.append((modeltype, "MAPE", mape_sc))
    
    @staticmethod
    def reg_r2_score(modeltype, test, pred):
        r2_sc = r2_score(test, pred)
        regression_string_evalaution.append((modeltype, "R2", r2_sc))
    
    @staticmethod
    def reg_explained_variance_score(modeltype, test, pred):
        evs_sc = explained_variance_score(test, pred)
        regression_string_evalaution.append((modeltype, "EVS", evs_sc))
    
    @staticmethod
    def reg_prediction_vs_actual_plot(modeltype, test, pred):
        plt.figure(figsize=(6, 6))
        plt.scatter(test, pred, color='blue', alpha=0.6)
        plt.plot([test.min(), test.max()],
                [test.min(), test.max()],
                'r--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f'Prediction vs Actual - {modeltype}')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        prediction_vs_plot.append((modeltype, buf))
        plt.close()
    
    @staticmethod
    def reg_residual_plot(modeltype, test, pred):
        residuals = test - pred
        plt.figure(figsize=(6, 4))
        plt.scatter(pred, residuals, color='purple', alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f'Residual Plot - {modeltype}')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        residual_plot.append((modeltype, buf))
        plt.close()
        
    @staticmethod
    def reg_residual_histogram(modeltype, test, pred):
        residuals = (test - pred).flatten()
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=20, color='orange', edgecolor='black')
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residuals - FHE")
        plt.grid(True)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        residual_histogram.append((modeltype, buf))
        plt.close()
    
    @staticmethod 
    def training_evaluation(modeltype, train_losses):
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        training_evaluation.append((modeltype, buf))
        plt.close()
    
    @staticmethod
    def train_val_loss_plot(modeltype, train_losses, val_losses):
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{modeltype} - Train vs Validation Loss')
        plt.legend()
        plt.grid(True)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        train_val_evaluation.append((modeltype, buf))
        plt.close()
        
    @staticmethod
    def function_time(timeType: str, time_it_took: int):
        time_outputs.append((timeType, "Time", time_it_took))
    
    ################### DISPLAY METHODS #################################
    
    @staticmethod
    def save_ml_metrics_csv():
        print("[INFO] Writing regression metrics to CSV")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_id = 1

        if os.path.isfile("br_outputs/metrics.csv"):
            with open("br_outputs/metrics.csv", 'r', newline='') as f:
                reader = csv.DictReader(f)
                model_numbers = []
                for row in reader:
                    model_str = row.get("Model", "")
                    if model_str.startswith("Model "):
                        try:
                            model_num = int(model_str.split(" ")[1])
                            model_numbers.append(model_num)
                        except ValueError:
                            continue
                if model_numbers:
                    model_id = max(model_numbers) + 1

        file_exists = os.path.isfile("br_outputs/metrics.csv")
        with open("br_outputs/metrics.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow(["Timestamp", "Model", "Process", "Metric", "Score"])

            for modeltype, metric_name, score in regression_string_evalaution:
                writer.writerow([timestamp, f"Model {model_id}", modeltype, metric_name, f"{score:.4f}"])
                
            for modeltype, metric_name, score in time_outputs:
                writer.writerow([timestamp, f"Model {model_id}", modeltype, metric_name, f"{score}"])
                
        print("[INFO] Finished Writing Metrics to CSV")
        return model_id
    
    @staticmethod        
    def prepare_pdf_report():
        elements = []
        styles = getSampleStyleSheet()
        
        elements.append(Paragraph("Privacy-Preserving ML: ML Metrics Report", styles['Heading2']))
        elements.append(Spacer(1, 5))
        
        for model_name, buf in prediction_vs_plot:
            elements.append(Paragraph(f"Prediction vs Plot - {model_name}", styles['Heading3']))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
            
        for model_name, buf in residual_plot:
            elements.append(Paragraph(f"Residual Plot - {model_name}", styles['Heading3']))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
            
        for model_name, buf in residual_histogram:
            elements.append(Paragraph(f"Residual Histogram - {model_name}", styles['Heading3']))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
            
        for model_name, buf in training_evaluation:
            elements.append(Paragraph(f"Training Loss - {model_name}", styles["Heading3"]))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
            
        for model_name, buf in train_val_evaluation:
            elements.append(Paragraph(f"Train vs Validation Loss - {model_name}", styles["Heading3"]))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
            
        for model_name, buf in predicted_images:
            elements.append(Paragraph(f"Predicted Images - {model_name}", styles['Heading3']))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
        
        return elements
    
    @staticmethod
    def create_ml_report(model_id):
        print("[INFO] Creating PDF report")

        output_filename = f"br_outputs/ml_metrics_report_model_{model_id}.pdf"
        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        elements.append(Paragraph("PrivML: Test Outputs", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Timestamp: {timestamp}", styles['Normal']))
        elements.append(Paragraph(f"Model ID: Model {model_id}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Training Configuration", styles['Heading2']))
        elements.append(Paragraph(f"Dataset Path: {DATASET_PATH}", styles['Normal']))
        elements.append(Paragraph(f"Image Size: {IMAGE_SIZE}", styles['Normal']))
        elements.append(Paragraph(f"Number of Images in Dataset: {DATASET_SIZE}", styles['Normal']))
        elements.append(Paragraph(f"Train/Val Split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}%", styles['Normal']))
        elements.append(Paragraph(f"Random Seed: {SEED}", styles['Normal']))
        elements.append(Paragraph(f"Sample Output Count: {SAMPLE_OUTPUT_COUNT}", styles['Normal']))
        elements.append(Paragraph(f"Epochs: {EPOCHS}", styles['Normal']))
        elements.append(Paragraph(f"Batch Size: {BATCH_SIZE}", styles['Normal']))
        elements.append(Paragraph(f"Learning Rate: {LEARNING_RATE}", styles['Normal']))
        elements.append(Paragraph(f"Optimizer: {OPTIMIZER.__name__}", styles['Normal']))
        elements.append(Paragraph(f"Loss Function: {LOSS_FUNCTION.__class__.__name__}", styles['Normal']))
        elements.append(Spacer(1, 12))

        mmt = Ml_Metrics_Evaluation.prepare_pdf_report()
        if mmt:
            elements.extend(mmt)

        doc.build(elements)
        print(f"[INFO] PDF report saved to: {output_filename}")