import os
import csv
from io import BytesIO
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn import metrics
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image

from mn_config import (
    DATASET_PATH, TRAIN_RATIO, VAL_RATIO, SEED,
    SAMPLE_OUTPUT_COUNT, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    OPTIMIZER, LOSS_FUNCTION, DATASET_SIZE
)

time_outputs = []
training_evaluation = []
train_val_evaluation = []
classification_string_evaluation = []
confusion_matrix_images = []
predicted_images = []

# evaluation.py
# - Performs basic regression evaluation such as MSE, MAE
# - Generates plots to visually evaluate model
# - Append metrics to a continuously updated .csv -> continuously track performance
# - Generates a new .pdf for each model run to visualize plots -> continuously track performance

class Ml_Metrics_Evaluation:
    
    ################### EVALUATION METHODS #############################
    
    @staticmethod
    def clas_accuracy_score(modeltype, test, pred):
        acc_sc = accuracy_score(test, pred)
        classification_string_evaluation.append((modeltype, "acc", acc_sc))
    
    @staticmethod
    def clas_precision_score(modeltype, test, pred):
        ps_sc = precision_score(test, pred, average='macro')
        classification_string_evaluation.append((modeltype, "ps", ps_sc))
    
    @staticmethod
    def clas_recall_score(modeltype, test, pred):
        rs_sc = recall_score(test, pred, average='macro')
        classification_string_evaluation.append((modeltype, "recall", rs_sc))
    
    @staticmethod
    def clas_f1_score(modeltype, test, pred):
        f1_sc = f1_score(test, pred, average='macro')
        classification_string_evaluation.append((modeltype, "f1", f1_sc))
    
    @staticmethod
    def clas_confusion_matrix(modeltype, test, pred):
        cm = confusion_matrix(test, pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix - {modeltype}')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        confusion_matrix_images.append((modeltype, buf))
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

        if os.path.isfile("mn_outputs/metrics.csv"):
            with open("mn_outputs/metrics.csv", 'r', newline='') as f:
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

        file_exists = os.path.isfile("mn_outputs/metrics.csv")
        with open("mn_outputs/metrics.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow(["Timestamp", "Model", "Process", "Metric", "Score"])

            for modeltype, metric_name, score in classification_string_evaluation:
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
            
        for model_name, buf in confusion_matrix_images:
            elements.append(Paragraph(f"Confusion Matrix - {model_name}", styles['Heading3']))
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

        output_filename = f"mn_outputs/ml_metrics_report_model_{model_id}.pdf"
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