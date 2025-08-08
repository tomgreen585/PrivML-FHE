from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from engr489_project.src.testing.time_metrics_test import Time_Metrics_Test
from engr489_project.src.testing.ml_metrics_test import Ml_Metrics_Test
from engr489_project.src.testing.security_test import Security_Metrics
from engr489_project.src.testing.fhe_metrics_test import FHE_Metrics

elements = []
testing_output_filename = "../outputs/fhe_metrics_report.pdf"
testing_model_type_one = "classification"
testing_model_type_two = "regression"

class Testing:
    
    @staticmethod
    def create_security_report(output_filename, model_type):
        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        styles = getSampleStyleSheet()
        
        elements.append(Paragraph("PrivML: Test Outputs", styles['Title']))
        smt = Time_Metrics_Test.generate_pdf_report()
        mmt = Ml_Metrics_Test.generate_pdf_report(model_type)
        st = Security_Metrics.generate_pdf_report()
        fhet = FHE_Metrics.generate_pdf_report()
        
        elements.extend(smt)
        elements.extend(mmt)
        elements.extend(st)
        elements.extend(fhet)
        
        doc.build(elements)
        print(f"[INFO] PDF report successfully saved to: {output_filename}")  

#run if test script is run manually. Use case is for actual implementation
if __name__ == "__main__":  
    Time_Metrics_Test.function_time("Encryption Time", 50)
    Time_Metrics_Test.function_time("ML Training Time", 120)
    Time_Metrics_Test.display_security_metrics()
    Security_Metrics.function_time("Encryption Time", 50)
    Security_Metrics.key_size_bits(4096, 4096)
    Security_Metrics.noise_budget_remaining(180)
    Security_Metrics.threshold_parameters(2, 3)
    Security_Metrics.quantum_resistance_notes("CKKS")

    Testing.create_security_report(testing_output_filename, testing_model_type_one)
    # Testing.create_security_report(testing_output_filename, testing_model_type_two)