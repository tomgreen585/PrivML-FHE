from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Performance Metrics

# Requirements:
# - Execution time -> time taken for encryption, inference, and decryption (total runtime)
# - Latency -> time per prediction or per operation (for ML inference)
# - Throughput -> number of encrypted operations per second
# - Memory usage -> RAM needed for encrypted operations (during training or inference)
# - Bootstrapping time -> time taken per bootstrapping oepration schemes

# What this code does:
# - Stores all times taken during model training/testing/inference
# - Stores times from encrypting/decrypting using FHE
# - Prints times

time_outputs = []

class Time_Metrics_Test:

    @staticmethod
    def function_time(timeType: str, time_it_took: int):
        time_outputs.append((timeType, time_it_took))

    @staticmethod
    def display_security_metrics():
        for type, time_taken in time_outputs:
            print("\n Type:", type, "- Time Taken:", time_taken)
    
    @staticmethod
    def generate_pdf_report():
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Privacy-Preserving ML: FHE Performance Report", styles['Heading2']))
        elements.append(Spacer(1, 5))

        data = [["Metric", "Time (seconds)"]]  # Header row

        for metric, time in time_outputs:
            data.append([metric, f"{time}s"])

        # Create table
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
        return elements
    