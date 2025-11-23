import sys
import json
import base64
import os
import numpy as np
import tenseal as ts
from PIL import Image
from io import BytesIO
from src.fhe_face_ml_pipeline import fhe_fc_config as config

"""
Helper class for encrypting and decrypting face image data using 
Fully Homomorphic Encryption (FHE) with the TenSEAL CKKS scheme. 
Receives content from client -> trusted server -> helper.py.
"""

class FHEFaceClientHelper:
    """
    Helper class for face detection model.
    
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

    def load_image_and_normalize(self, source):
        """
        Load and normalize a 128x128 RGB image from a file path or base64 string.

        Args:
        source (str): Path to image file or base64-encoded string.

        Returns:
        image (np array): Normalized image tensor of shape (3, 128, 128), channel-first.

        Raises:
            Exception: If image decoding or processing fails.
        """
        try:
            if os.path.isfile(source):
                image = Image.open(source).convert('RGB').resize((128, 128))
            else:
                base64_data = source.split(',')[1] if ',' in source else source
                image_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_data)).convert('RGB').resize((128, 128))
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            return image
        except Exception as e:
            raise

    def encrypt_image(self, image_tensor):
        """
        Encrypt an image tensor using im2col encoding per channel.

        Args:
        image_tensor (np array): Image array with shape (3, 128, 128).

        Returns:
        result (dict): contains encrypted channel data, TenSEAL context, and encoding metadata.

        Raises:
        ValueError: If context is not initialized.
        Exception: On encryption failure.
        """
        if not self.context:
            raise ValueError("No context available")
        try:
            kernel_shape = [config.PLAIN_KERNEL_SIZE, config.PLAIN_KERNEL_SIZE]
            stride = config.PLAIN_STRIDE
            channels = []
            windows_nb = None

            for i, ch in enumerate(image_tensor):
                ch_data = ch.tolist()
                x_enc, windows_nb = ts.im2col_encoding(
                    self.context, 
                    ch_data, 
                    kernel_shape[0], 
                    kernel_shape[1], 
                    stride
                )
                channels.append(x_enc)
            result = {
                "channels": [ch.serialize().hex() for ch in channels],
                "context": self.context.serialize(save_secret_key=True).hex(),
                "kernel_shape": kernel_shape,
                "stride": stride,
                "windows_nb": windows_nb,
            }
            return result
        except Exception as e:
            raise

    def decrypt_prediction(self, encrypted_output_hex):
        """
        Decrypt the model's encrypted prediction (bounding box) from the server.

        Args:
        encrypted_output_hex (str): hex string of encrypted CKKS vector.

        Returns:
        result (List): list of 4 float values representing the bounding box (x, y, w, h).

        Raises:
        ValueError: If context is not initialized or decryption fails.
        """
        if not self.context:
            raise ValueError("No context available")
        try:
            if isinstance(encrypted_output_hex, str):
                encrypted_output_data = bytes.fromhex(encrypted_output_hex)
            encrypted_output = ts.ckks_vector_from(self.context, encrypted_output_data)
            prediction = encrypted_output.decrypt()
            result = prediction[:4]
            return result  
        except Exception as e:
            raise ValueError(f"Failed to decrypt: {e}")

def main():
    """
    Entry point for the script.

    Uses:
    1. Encrypts the image and outputs encrypted result
    2. Decrypts prediction and returns bounding box as JSON
    """
    if len(sys.argv) < 3:
        print("Usage: python fhe_face_client_helper.py <encrypt|decrypt> <args>", file=sys.stderr)
        return
    
    operation = sys.argv[1]
    helper = FHEFaceClientHelper()
    
    try:
        if operation == "encrypt":
            password = sys.argv[2]
            image_path_or_b64 = sys.argv[3]
            helper.create_context_from_password(password)
            image = helper.load_image_and_normalize(image_path_or_b64)
            result = helper.encrypt_image(image)
            sys.stderr.flush()
            json_output = json.dumps(result)
            sys.stdout.write(json_output)
            sys.stdout.flush()
        elif operation == "decrypt":
            password = sys.argv[2]
            json_path = sys.argv[3]
            with open(json_path, "r") as f:
                data = json.load(f)
            context_hex = data["context"]
            encrypted_output = data["encrypted_output"]
            helper.context = ts.context_from(bytes.fromhex(context_hex))
            pred = helper.decrypt_prediction(encrypted_output)
            sys.stderr.flush()
            json_output = json.dumps({"prediction": pred})
            sys.stdout.write(json_output)
            sys.stdout.flush()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()