import numpy as np
import pandas as pd
import io

from io import BytesIO
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image


# Machine Learning Metrics

# Requirements:
# - Model Accuracy (Plaintext vs Encrypted) -> drop in accuracy between plaintext and encrypted models (%)
# - Model Depth -> number of layers supported under encryption (e.g. NN-10, NN-50)
# - Activation Functions Support -> how many/which activations can be securely approximated (e.g. ReLU)
# - Quantization Bits -> bit width used for encrypted computations (e.g. 6-bit, 8-bit)
# - Error Tolerance -> acceptable probability of small error during encrypted inference

classification_string_evaluation = []
regression_string_evalaution = []
confusion_matrix_images = []
prediction_vs_plot = []
residual_plot = []
residual_histogram = []

class Ml_Metrics_Test:
    
    ################### CLASSIFICATION EVALUATION #########################
    
    @staticmethod
    def clas_accuracy_score(modeltype, test, pred):
        acc_sc = accuracy_score(test, pred)
        classification_string_evaluation.append((modeltype, acc_sc))
    
    @staticmethod
    def clas_precision_score(modeltype, test, pred):
        ps_sc = precision_score(test, pred)
        classification_string_evaluation.append((modeltype, ps_sc))
    
    @staticmethod
    def clas_recall_score(modeltype, test, pred):
        rs_sc = recall_score(test, pred)
        classification_string_evaluation.append((modeltype, rs_sc))
    
    @staticmethod
    def clas_f1_score(modeltype, test, pred):
        f1_sc = f1_score(test, pred)
        classification_string_evaluation.append((modeltype, f1_sc))
    
    @staticmethod
    def clas_roc_auc_score(modeltype, test, pred):
        rocauc_sc = roc_auc_score(test, pred)
        classification_string_evaluation.append((modeltype, rocauc_sc))
    
    @staticmethod
    def clas_confusion_matrix(modeltype, test, pred):
        cm = metrics.confusion_matrix(test, pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix - {modeltype}')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        confusion_matrix_images.append((modeltype, buf))
        plt.close()
    
    ################### REGRESSION EVALUATION #############################
    
    @staticmethod
    def reg_mean_squared_error(modeltype, test, pred):
        mse_sc = mean_squared_error(test, pred)
        regression_string_evalaution.append((modeltype, mse_sc))
    
    @staticmethod
    def reg_mean_absolute_error(modeltype, test, pred):
        mae_sc = mean_absolute_error(test, pred)
        regression_string_evalaution.append((modeltype, mae_sc))
    
    @staticmethod
    def reg_mean_absolute_percentage_error(modeltype, test, pred):
        mape_sc = mean_absolute_percentage_error(test, pred)
        regression_string_evalaution.append((modeltype, mape_sc))
    
    @staticmethod
    def reg_r2_score(modeltype, test, pred):
        r2_sc = r2_score(test, pred)
        regression_string_evalaution.append((modeltype, r2_sc))
    
    @staticmethod
    def reg_explained_variance_score(modeltype, test, pred):
        evs_sc = explained_variance_score(test, pred)
        regression_string_evalaution.append((modeltype, evs_sc))
    
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
        plt.scatter(pred, residuals, color='purple', alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f'Residual Plot - {modeltype}')
        plt.figure(figsize=(6, 4))
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        residual_plot.append((modeltype, buf))
        plt.close()
        
    @staticmethod
    def reg_residual_histogram(modeltype, test, pred):
        residuals = test - pred
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
    
    ################### DISPLAY METHODS #################################
    
    def display_ml_classification_metrics():
        for type, metric in classification_string_evaluation:
            print("\n Type:", type, "- Score:", metric)
            
    def generate_pdf_report(modeltype):
        elements = []
        styles = getSampleStyleSheet()
        
        elements.append(Paragraph("Privacy-Preserving ML: ML Metrics Report", styles['Heading2']))
        elements.append(Spacer(1, 5))
        
        if modeltype == "Classification":
            if classification_string_evaluation:
                data = [["Metric", "Score"]]
                for metric, score in classification_string_evaluation:
                    data.append([metric, f"{score:.4f}"])
                    
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))
                
            for model_name, buf in confusion_matrix_images:
                elements.append(Paragraph(f"Confusion Matrix - {model_name}", styles['Heading3']))
                elements.append(Image(buf, width=400, height=400))
                elements.append(Spacer(1, 12))
        
        elif(modeltype == "Regression"):
            if regression_string_evalaution:
                data = [["Metric", "Score"]]
                for metric, score in regression_string_evalaution:
                    data.append([metric, f"{score:.4f}"])
                    
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))
                
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
              
        else:
            return
        
        return elements
    