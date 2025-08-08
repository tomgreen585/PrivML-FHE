from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image

# - Bootstrapping -> does the program avoid bootstrapping or how often is bootstrapping being performed?
# - Approximate vs Exact Computations -> how much error is introduced due to approximation (CKKS or Concrete ML)
# - Parameter tuning -> how aggressive is optimization on modulus sizes, scaling factors, etc.

fhe_scheme_metrics = []

class FHE_Metrics:
    
    @staticmethod
    def bootstrapping_info(info):
        fhe_scheme_metrics.append(("Bootstrapping", info))

    @staticmethod
    def computation_type_error(mse_error):
        fhe_scheme_metrics.append(("Approximation Error (MSE)", f"{mse_error:.6f}"))

    @staticmethod
    def parameter_tuning_summary(summary):
        fhe_scheme_metrics.append(("Parameter Tuning", summary))

    @staticmethod
    def generate_pdf_report():
        elements = []
        styles = getSampleStyleSheet()
        
        elements.append(Paragraph("Privacy-Preserving ML: FHE Metrics Report", styles['Heading2']))
        data = [["Metric", "Value"]] + fhe_scheme_metrics
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
            
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        return elements