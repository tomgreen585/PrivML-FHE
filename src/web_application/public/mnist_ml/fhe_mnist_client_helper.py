import sys
import json
import tenseal as ts
import torch
from src.fhe_mnist_ml_pipeline import fhe_mn_config as config

"""
Client-side helper for Fully Homomorphic Encryption of user digit data
using TenSEAL CKKS scheme. 
Receives content from client -> trusted server -> helper.py.
"""

class FHEClientHelper: 
    """
    Helper class for mnist model.
    
    Initialize the helper with an empty TenSEAL context.
    """
    
    def __init__(self):
        self.context = None
    
    def create_context_from_password(self, password):
        """
        Create a TenSEAL CKKS context using parameters from config.

        Args:
        password (str): User-supplied password. 
            (Currently unused, but can be extended to 
            create CKKS context like this -> e.g. PySEAL.)

        Returns:
        context (ts.Context): The initialized TenSEAL context.

        Raises:
        ValueError: if context creation fails.
        """
        try:
            context = ts.context(
                config.SCHEME_TYPE,
                poly_modulus_degree=config.POLY_MODULUS_DEGREE,
                coeff_mod_bit_sizes=config.COEFF_MOD_BIT_SIZES
            )
            context.global_scale = 2 ** config.BITS_SCALE
            context.generate_galois_keys()
            self.context = context
            return context
        except Exception as e:
            print(f"Context creation failed: {e}", file=sys.stderr)
            raise ValueError(f"Failed to create TenSEAL context: {e}")
    
    def encrypt_image_data(self, image_data):
        """
        Encrypt a flattened input image using im2col encoding.

        Args:
        image_data (list): flattened list of 784 float values (28x28 image).

        Returns:
        x_enc, windows_nb: encrypted encoded input and number of im2col windows.

        Raises:
        ValueError: If context is not set or encryption fails.
        """
        if not self.context:
            raise ValueError("No context created")
        try:
            kernel_shape = [config.PLAIN_KERNEL_SIZE, config.PLAIN_KERNEL_SIZE]
            stride = config.PLAIN_STRIDE
            image_tensor = torch.tensor(image_data).reshape(28, 28)
            sample = image_tensor.tolist()
            assert isinstance(sample, list) and isinstance(sample[0], list), "Sample must be 2D"

            x_enc, windows_nb = ts.im2col_encoding(
                self.context,
                sample,
                kernel_shape[0],
                kernel_shape[1],
                stride
            )
            return x_enc, windows_nb
        except Exception as e:
            print(f"Encryption failed: {e}", file=sys.stderr)
            raise ValueError(f"Encryption failed: {e}")

    def serialize_for_server(self, encrypted_vector, windows_nb):
        """
        Serialize the encrypted vector and context for server upload.

        Args:
        encrypted_vector (ts.CKKSVector): Encrypted input vector.
        windows_nb (int): Number of im2col windows in the encoding.

        Returns:
        result (dict): JSON-serializable result for server input.

        Raises:
        ValueError: If serialization fails or context is missing.
        """
        if not self.context:
            raise ValueError("No context available")
        try:
            kernel_shape = [config.PLAIN_KERNEL_SIZE, config.PLAIN_KERNEL_SIZE]
            stride = config.PLAIN_STRIDE
            result = {
                'encrypted_vector': encrypted_vector.serialize().hex(),
                'context': self.context.serialize(save_secret_key=True).hex(),
                'kernel_shape': kernel_shape,
                'stride': stride,
                'windows_nb': windows_nb
            }
            return result
        except Exception as e:
            raise ValueError(f"Failed to serialize: {e}")
    
    def decrypt_result(self, encrypted_output_data):
        """
        Decrypt the server's prediction output and extract top prediction.

        Args:
        encrypted_output_data (str): hex-encoded encrypted result vector.

        Returns:
        result (dict): dictionary with predicted digit and partial decrypted vector.

        Raises:
        ValueError: if decryption fails or context is missing.
        """
        if not self.context:
            raise ValueError("No context available")
        try:
            if isinstance(encrypted_output_data, str):
                encrypted_output_data = bytes.fromhex(encrypted_output_data)
            encrypted_output = ts.ckks_vector_from(self.context, encrypted_output_data)
            output = encrypted_output.decrypt()
            output_tensor = torch.tensor(output).view(1, -1)
            _, pred = torch.max(output_tensor, 1)
            prediction = pred.item()
            
            result = {
                'prediction': int(prediction),
                'output_vector': output[:10] if len(output) >= 10 else output
            }
            return result
        except Exception as e:
            raise ValueError(f"Failed to decrypt: {e}")

def main():
    """
    Entry point for the script.

    Uses:
    1. Encrypts user and outputs encrypted result
    2. Decrypts model prediction and returns result as JSON
    """
    if len(sys.argv) < 2:
        print("Usage: python working_fhe_client_helper.py <operation> [args...]", file=sys.stderr)
        return
    
    operation = sys.argv[1]
    helper = FHEClientHelper()
    
    try:
        if operation == "encrypt":
            if len(sys.argv) != 7:
                raise ValueError("encrypt requires: password image_data_json kernel_h kernel_w stride")
            password = sys.argv[2]
            image_data = json.loads(sys.argv[3])
            helper.create_context_from_password(password)
            encrypted_vector, windows_nb = helper.encrypt_image_data(image_data)
            result = helper.serialize_for_server(encrypted_vector, windows_nb)
            print(json.dumps(result))
        elif operation == "decrypt":
            if len(sys.argv) != 4:
                raise ValueError("decrypt requires: password encrypted_output_data_path")
            password = sys.argv[2]
            json_path = sys.argv[3]
            with open(json_path, 'r') as f:
                encrypted_output_json = json.load(f)
            encrypted_output_hex = encrypted_output_json.get("encrypted_output")
            context_hex = encrypted_output_json.get("context")
            if not encrypted_output_hex or not context_hex:
                raise ValueError("Missing encrypted_output or context")
            helper.context = ts.context_from(bytes.fromhex(context_hex))
            result = helper.decrypt_result(encrypted_output_hex)
            print(json.dumps(result))
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()