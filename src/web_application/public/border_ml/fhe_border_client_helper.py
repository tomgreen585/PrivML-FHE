import sys
import json
import base64
import os
import logging
import numpy as np
import tenseal as ts
from PIL import Image
from io import BytesIO
from src.fhe_border_ml_pipeline import fhe_br_config as config

"""
Helper class for client to manage the encryption and decryption of images 
for the border creation task using the TenSEAL CKKS scheme. Receives content
from client -> trusted server -> helper.py.
"""

class FHEBorderClientHelper:
    """
    Helper class for border model.
    
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
        Load an image from a file path or base64 string, convert to grayscale,
        resize to 32x32, normalize pixel values, and reshape for model input.

        Args:
        source (str): Path to image file or base64-encoded string.

        Returns:
        image (np array): Normalized grayscale image tensor of shape (1, 32, 32).

        Raises:
        Exception: if the image cant be loaded or processed.
        """
        try:
            if os.path.isfile(source):
                image = Image.open(source).convert('RGB').resize((32, 32))
            else:
                base64_data = source.split(',')[1] if ',' in source else source
                image_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_data)).convert('RGB').resize((32, 32))
            image_gray = image.convert('L')
            image = np.asarray(image_gray, dtype=np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as e:
            raise

    def encrypt_image(self, image_tensor):
        """
        Encrypt a grayscale image using im2col encoding with the current TenSEAL context.

        Args:
        image_tensor (np array): Grayscale image of shape (1, 32, 32).

        Returns:
        result (dict): contains encrypted data, context, metadata.

        Raises:
        ValueError: If no context is set.
        Exception: If encryption fails.
        """
        if not self.context:
            raise ValueError("No context available")
        try:
            kernel_shape = [config.PLAIN_CONV1_KERNEL_SIZE, config.PLAIN_CONV1_KERNEL_SIZE]
            stride = 1 #manually set
            channels = []
            windows_nb = None
            
            for i, ch in enumerate(image_tensor):
                padded_ch = np.pad(ch, ((1, 1), (1, 1)), mode='constant', constant_values=0)
                ch_data = padded_ch.tolist()
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
                "image_size": 32,
                "task_type": "border_creation"
            }
            return result
        except Exception as e:
            raise

    def decrypt_prediction(self, encrypted_output_hex):
        """
        Decrypt the encrypted result from the model that client sent using the current TenSEAL context.

        Args:
        encrypted_output_hex (str): Hex-encoded serialized CKKS vector.

        Returns:
        list or np.ndarray: image (32x32) of decrypted pixel values

        Raises:
        ValueError: If no context is set.
        Exception: If decryption fails.
        """
        if not self.context:
            raise ValueError("No context available")
        try:
            if isinstance(encrypted_output_hex, str):
                encrypted_output_data = bytes.fromhex(encrypted_output_hex)
            encrypted_output = ts.ckks_vector_from(self.context, encrypted_output_data)
            prediction = encrypted_output.decrypt()
            image_size = 32
            expected_size = image_size * image_size
            if len(prediction) >= expected_size:
                image_data = prediction[:expected_size]
                cleaned_image = np.array(image_data).reshape(image_size, image_size)
                cleaned_image = np.clip(cleaned_image * 255.0, 0, 255).astype(np.uint8)
                return cleaned_image.tolist()
            else:
                return prediction 
        except Exception as e:
            raise

def main():
    """
    Entry point for the script.

    Uses:
    1. Encrypts the image and outputs encrypted JSON
    2. Decrypts prediction and returns cleaned image pixels as JSON.
    """
    if len(sys.argv) < 3:
        print("Usage: python fhe_border_client_helper.py <encrypt|decrypt> <args>", file=sys.stderr)
        return

    operation = sys.argv[1]
    helper = FHEBorderClientHelper()

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
            json_output = json.dumps({"cleaned_image": pred})
            sys.stdout.write(json_output)
            sys.stdout.flush()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        print(f"[PYTHON ERR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()