from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ### Security Metrics

# - Encryption key size -> size (in bits) of public and private keys (e.g. 128, 256)
# - Noise Budget Management -> remaining noise after operations, important to ensure correct decryption
# - Threshold Parameters (if multiple people decrypting) -> minimum number of participants needed to decrypt (e.g. (t, n) - threshold)
# - Quantum resistance -> lattice problems / implementation resistant

security_parameters = []

class Security_Metrics:
    
    @staticmethod
    def key_size_bits(public_key_bits, private_key_bits):
        security_parameters.append(("Public Key Size (bits)", public_key_bits))
        security_parameters.append(("Private Key Size (bits)", private_key_bits))

    @staticmethod
    def noise_budget_remaining(remaining_bits):
        security_parameters.append(("Remaining Noise Budget (bits)", remaining_bits))

    @staticmethod
    def threshold_parameters(t, n):
        security_parameters.append((f"Threshold Decryption (t,n)", f"({t}, {n})"))

    @staticmethod
    def quantum_resistance_notes(scheme_type):
        note = f"{scheme_type} is lattice-based and offers quantum resistance."
        security_parameters.append(("Quantum Resistance", note))
        
    @staticmethod
    def generate_pdf_report():
        elements = []
        styles = getSampleStyleSheet()
        
        elements.append(Paragraph("Privacy-Preserving ML: Security Metrics Report", styles['Heading2']))
        data = [["Metric", "Value"]] + security_parameters
        
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