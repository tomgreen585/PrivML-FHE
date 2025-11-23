import sys
import json
import time
import torch
import tenseal as ts
from pathlib import Path
import traceback

"""
fhe_mnist.py

Loads a model from the specified path. Converts a given image to a tensor. Performs
inference and outputs the predicted label to a JSON file.
"""
def log(msg):
    """Log output to stdout"""
    print(f"[INFO] {msg}")
    sys.stdout.flush()

def err(msg):
    """Log output for standardized error output to stderr."""
    sys.stderr.write(f"[ERR] {msg}\n")
    sys.stderr.flush()

def load_mnist_model():
    """
    Attempts to load the mnist model from specified path.

    Returns:
    model (torch.nn.Module): The loaded PyTorch model.
    device (str): Device used for inference (e.g. "cpu").
    """
    log("Loading number detection model...")
    try:
        from fhe_mn_plain_model import ML_Model
        from fhe_mn_fhe_model import FHE_Model
        from fhe_mn_config import (CURRENT_MODEL)
        
        model = ML_Model()
        REPO_ROOT = Path(__file__).parent.parent.parent
        
        #path to trained model during ml_pipeline
        candidate_paths = [
            Path(__file__).parent / "models" / CURRENT_MODEL,
            REPO_ROOT / "src" / "fhe_mnist_ml_pipeline" / "models" / CURRENT_MODEL,
        ]
        
        model_loaded = False
        device = 'cpu'
        for model_path in candidate_paths:
            if model_path.is_file():
                try:
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    model_loaded = True
                    break
                except Exception as e:
                    log(f"Failed to load number model.")
        
        if not model_loaded:
            log("Model failed to load...exiting")
            sys.exit(1)
            
        fhe_model = FHE_Model(model.eval())
        log("Encrypted model instance created.")
        return fhe_model
    except ImportError as e:
        log(f"Could not load FHE model: {e}")
        traceback.print_exc(file=sys.stderr)
        raise

def main():
    """
    Script main that is run by server.js
    Loads the input image, runs the mnist model, and sends back the output.
    """
    if len(sys.argv) != 3:
        print("[ERR] Usage: python3 fhe_mnist.py <input_data_path> <output_path>")
        sys.exit(1)
    
    input_data_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        with open(input_data_path, 'r') as f:
            fhe_data = json.load(f)
        
        encrypted_vector_data = fhe_data['encrypted_vector']
        context_data = fhe_data['context']
        kernel_shape = fhe_data['kernel_shape']
        stride = fhe_data['stride']
        windows_nb = fhe_data['windows_nb']
        
        log("Reconstructing TenSEAL Data...")
        context = ts.context_from(bytes.fromhex(context_data))
        x_enc = ts.ckks_vector_from(context, bytes.fromhex(encrypted_vector_data))
        
        fhe_model = load_mnist_model()
        
        log("Running encrypted inference...")
        inference_start = time.time()
        enc_output = fhe_model(x_enc, windows_nb)
        end_time = time.time()
        inference_time = end_time - inference_start
        log(f"Inference completed in {inference_time:.3f}s")
        
        encrypted_output_data = enc_output.serialize().hex()
        
        result = {
            "encrypted_output": encrypted_output_data,
            "inference_time": inference_time,
            "windows_nb": windows_nb,
            "kernel_shape": kernel_shape,
            "stride": stride,
            "server_processing_complete": True,
        }
        
        log("Writing encrypted output.")
        with open(output_path, 'w') as f:
            json.dump(result, f)
    except Exception as e:
        print(f"FHE mnist server pipeline failed: {e}")
        traceback.print_exc(file=sys.stderr)
        try:
            with open(output_path, 'w') as f:
                json.dump({
                "error": str(e),
                "server_processing_complete": False,
                "timestamp": time.time()
            }, f)
        except Exception as io_err:
            err(f"Could not write fallback error JSON: {io_err}")
        sys.exit(1)

if __name__ == "__main__":
    main()