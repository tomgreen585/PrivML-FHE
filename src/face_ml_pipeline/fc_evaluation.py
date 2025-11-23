import os
import csv
import psutil
import torch
import platform
from io import BytesIO
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image

from fc_config import (
    PLAIN_OPTIMIZER, PLAIN_EPOCHS, PLAIN_LEARNING_RATE, PLAIN_BATCH_SIZE, PLAIN_LOSS_FUNCTION,
    SAMPLE_OUTPUT_COUNT,
    DATASET_SIZE, DATASET_PATH,
    EN_KERNEL_SIZE, EN_PADDING, EN_ACT,
    SEED, TRAINING_SIZE, VALIDATION_SIZE, TESTING_SIZE
)

time_outputs = []
training_evaluation = []
train_val_evaluation = []
ckks_string_evaluation = []
regression_string_evalaution = []
prediction_vs_plot = []
residual_plot = []
residual_histogram = []
system_usage_metrics = []
actual_prediction_plaintext = []
actual_vs_real_prediction_plaintext = []
actual_prediction_encrypted = []
actual_vs_real_prediction_encrypted = []
system_info_logged = False

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
        
    @staticmethod  
    def log_resource_usage(model_type):
        global system_info_logged
        
        if not system_info_logged:
            system_usage_metrics.append((model_type, "OS", platform.platform()))
            system_usage_metrics.append((model_type, "Python Version", platform.python_version()))
            system_usage_metrics.append((model_type, "Processor", platform.processor()))
            system_info_logged = True
        
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_used = round(ram.used / (1024 ** 3), 2)
        ram_total = round(ram.total / (1024 ** 3), 2)

        system_usage_metrics.append((model_type, "CPU Usage (%)", cpu_percent))
        system_usage_metrics.append((model_type, "RAM Usage (Used GB)", ram_used))
        system_usage_metrics.append((model_type, "RAM Usage (Total GB)", ram_total))
        system_usage_metrics.append((model_type, "RAM Usage (%)", ram.percent))

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_alloc = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
            gpu_mem_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            gpu_util_percent = round((gpu_mem_alloc / gpu_mem_total) * 100, 2)

            system_usage_metrics.append((model_type, "GPU Name", gpu_name))
            system_usage_metrics.append((model_type, "GPU Mem Used (GB)", gpu_mem_alloc))
            system_usage_metrics.append((model_type, "GPU Mem Total (GB)", gpu_mem_total))
            system_usage_metrics.append((model_type, "GPU Util (%)", gpu_util_percent))
        else:
            system_usage_metrics.append((model_type, "GPU", "Not Available"))
            
    ################### ENCRYPTED EVALUATION METHODS ####################
    
    @staticmethod
    def CKKS_METRICS(modeltype, pmd, skg, bs, eds, ec):
        ckks_string_evaluation.append((modeltype, "EncryptionScheme", "CKKS"))
        ckks_string_evaluation.append((modeltype, "PolyModulusDegree", pmd))
        coeff_mod_bit_sizes = [skg] + [bs]*6 + [eds]
        ckks_string_evaluation.append((modeltype, "CoeffModBitSizes", str(coeff_mod_bit_sizes)))
        ckks_string_evaluation.append((modeltype, f"GlobalScale(2^{bs})", ec.global_scale))
        estimated_key_bits = pmd * bs
        ckks_string_evaluation.append((modeltype, "AppEncryptionKeySize", estimated_key_bits))
    
    ################### DISPLAY METHODS #################################
    
    @staticmethod
    def save_ml_metrics_csv():
        print("[INFO] Writing regression metrics to CSV")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_id = 1

        os.makedirs("fc_outputs", exist_ok=True)
        if os.path.isfile("fc_outputs/metrics.csv"):
            with open("fc_outputs/metrics.csv", 'r', newline='') as f:
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

        file_exists = os.path.isfile("fc_outputs/metrics.csv")
        with open("fc_outputs/metrics.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow(["Timestamp", "Model", "Process", "Metric", "Score"])

            for modeltype, metric_name, score in regression_string_evalaution:
                try:
                    score_str = f"{float(score):.4f}"
                except (ValueError, TypeError):
                    score_str = str(score)
                writer.writerow([timestamp, f"Model {model_id}", modeltype, metric_name, score_str])

            for modeltype, metric_name, score in time_outputs:
                try:
                    score_str = f"{float(score):.4f}"
                except (ValueError, TypeError):
                    score_str = str(score)
                writer.writerow([timestamp, f"Model {model_id}", modeltype, metric_name, score_str])

            for modeltype, metric_name, score in ckks_string_evaluation:
                try:
                    score_str = f"{float(score):.4f}"
                except (ValueError, TypeError):
                    score_str = str(score)
                writer.writerow([timestamp, f"Model {model_id}", modeltype, metric_name, score_str])
                
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
        
        for model_name, buf in actual_prediction_plaintext:
            elements.append(Paragraph(f"Plaintext: Model Prediction - {model_name}", styles['Heading3']))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
        
        for model_name, buf in actual_vs_real_prediction_plaintext:
            elements.append(Paragraph(f"Plaintext: Predicted vs Actual - {model_name}", styles['Heading3']))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
            
        for model_name, buf in actual_prediction_encrypted:
            elements.append(Paragraph(f"Encrypted: Model Prediction - {model_name}", styles['Heading3']))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
        
        for model_name, buf in actual_vs_real_prediction_encrypted:
            elements.append(Paragraph(f"Encrypted: Predicted vs Actual - {model_name}", styles['Heading3']))
            elements.append(Image(buf, width=400, height=400))
            elements.append(Spacer(1, 12))
        
        return elements
    
    @staticmethod
    def create_ml_report(model_id):
        print("[INFO] Creating PDF report")

        os.makedirs("fc_outputs", exist_ok=True)
        output_filename = f"fc_outputs/ml_metrics_report_model_{model_id}.pdf"
        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        elements.append(Paragraph("PrivML: Test Outputs", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Timestamp: {timestamp}", styles['Normal']))
        elements.append(Paragraph(f"Model ID: Model {model_id}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        elements.append(HRFlowable(width="100%", thickness=1, color='grey'))
        elements.append(Spacer(1, 12))
        
        #dataset Info
        elements.append(Paragraph(f"Dataset Path: {DATASET_PATH}", styles['Normal']))
        elements.append(Paragraph(f"Number of Images in Dataset: {DATASET_SIZE}", styles['Normal']))
        elements.append(Paragraph(f"Training Size: {int(TRAINING_SIZE*100)}%", styles['Normal']))
        elements.append(Paragraph(f"Validation Size: {int(VALIDATION_SIZE*100)}%", styles['Normal']))
        elements.append(Paragraph(f"Testing Size: {int(TESTING_SIZE*100)}%", styles['Normal']))
        elements.append(Paragraph(f"Random Seed: {SEED}", styles['Normal']))
        elements.append(Spacer(1, 12))

        #plaintext training info
        elements.append(Paragraph("Plain Model Training", styles['Heading3']))
        elements.append(Paragraph(f"Epochs: {PLAIN_EPOCHS}", styles['Normal']))
        elements.append(Paragraph(f"Batch Size: {PLAIN_BATCH_SIZE}", styles['Normal']))
        elements.append(Paragraph(f"Learning Rate: {PLAIN_LEARNING_RATE}", styles['Normal']))
        elements.append(Paragraph(f"Optimizer: {PLAIN_OPTIMIZER.__name__}", styles['Normal']))
        elements.append(Paragraph(f"Loss Function: {PLAIN_LOSS_FUNCTION.__class__.__name__}", styles['Normal']))
        elements.append(Paragraph(f"Sample Output Count: {SAMPLE_OUTPUT_COUNT}", styles['Normal']))
        elements.append(Spacer(1, 12))

        #plaintext model Architecture
        elements.append(Paragraph("Plain Model Architecture", styles['Heading3']))
        elements.append(Paragraph(f"Convolution Kernel Size: {EN_KERNEL_SIZE}", styles['Normal']))
        elements.append(Paragraph(f"Convolution Padding Size: {EN_PADDING}", styles['Normal']))
        elements.append(Paragraph(f"Model Activation Function: {EN_ACT.__class__.__name__}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(HRFlowable(width="100%", thickness=1, color='grey'))
        elements.append(Spacer(1, 12))

        mmt = Ml_Metrics_Evaluation.prepare_pdf_report()
        if mmt:
            elements.extend(mmt)
        
        elements.append(HRFlowable(width="100%", thickness=1, color='grey'))
        elements.append(Spacer(1, 12))
            
        if system_usage_metrics:
            elements.append(Paragraph("Runtime System Resource Usage", styles["Heading2"]))
            for model_name, metric, value in system_usage_metrics:
                elements.append(Paragraph(f"{model_name}: {metric}: {value}", styles["Normal"]))
            elements.append(Spacer(1, 12))

        doc.build(elements)
        print(f"[INFO] PDF report saved to: {output_filename}")