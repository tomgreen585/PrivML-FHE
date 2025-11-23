import sys
import json
import time
import torch
import tenseal as ts
from pathlib import Path
import traceback

"""
fhe_border.py

This script performs inference using a pre-trained border detection model with fhe wrapped model to add visual borders in input images.
It loads a PyTorch model, wraps with FHE model, processes an input image, runs the model, and saves the output.  
"""

def log(msg):
    """Prints an info message to stdout to display on web application log outputs."""
    print(f"[INFO] {msg}")
    sys.stdout.flush()

def err(msg):
    """Prints an error message to stderr for debugging."""
    sys.stderr.write(f"[ERR] {msg}\n")
    sys.stderr.flush()

def load_border_model():
    """
    Loads the ML border model from specified path.

    Returns:
    model (torch.nn.Module): loaded model
    device (str): device set for inference
    """
    log("Loading border model...")
    try:
        from fhe_br_fhe_model import FHE_Model
        from fhe_br_plain_model import ML_Model
        from fhe_br_config import (CURRENT_MODEL)
        
        model = ML_Model()
        REPO_ROOT = Path(__file__).parent.parent.parent
        candidate_paths = [
            Path(__file__).parent / "models" / CURRENT_MODEL,
            REPO_ROOT / "src" / "fhe_border_ml_pipeline" / "models" / CURRENT_MODEL,
        ]
        
        model_loaded = False
        device = "cpu"
        for path in candidate_paths:
            if path.is_file():
                try:
                    state_dict = torch.load(path, map_location=device)
                    model.load_state_dict(state_dict)
                    model_loaded = True
                    break
                except Exception as e:
                    err(f"Failed to load border model.")
        
        if not model_loaded:
            log("Model failed to load...exiting")
            sys.exit(1)
        
        model.eval()
        fhe_model = FHE_Model(model)
        log("Encrypted border model instance created.")
        return fhe_model
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise

def main():
    """
    Convers a PIL image to a PyTorch tensor for model inference.
    
    Args:
    img (PIL.Image): Input image
    device (str): device set for inference

    Returns:
    torch.Tensor: the image tensor shape (1, C, H, W)
    """
    if len(sys.argv) != 3:
        err("Usage: python3 fhe_border.py <input_data.json> <output_result.json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        with open(input_path, 'r') as f:
            input_data = json.load(f)
        
        channels_hex = input_data['channels']
        context_hex = input_data['context']
        kernel_shape = input_data['kernel_shape']
        stride = input_data['stride']
        windows_nb = input_data['windows_nb']
        image_size = input_data.get('image_size', 32)
        task_type = input_data.get('task_type', 'border_creation')
        
        log("Reconstructing TenSEAL Data...")
        context = ts.context_from(bytes.fromhex(context_hex))
        channels = []
        for i, ch_hex in enumerate(channels_hex):
            try:
                ch = ts.ckks_vector_from(context, bytes.fromhex(ch_hex))
                channels.append(ch)
            except Exception as e:
                raise
        
        model = load_border_model()
        
        log("Running encrypted inference...")
        start_time = time.time()
        try:
            enc_output = model(channels, windows_nb)
            end_time = time.time()
            inference_time = end_time - start_time
            log(f"Inference completed in {inference_time:.3f}s")
        except Exception as e:
            log("Inference pipeline failed")
        
        encrypted_output = enc_output.serialize().hex()
        
        result = {
            "encrypted_output": encrypted_output,
            "inference_time": inference_time,
            "kernel_shape": kernel_shape,
            "stride": stride,
            "windows_nb": windows_nb,
            "image_size": image_size,
            "task_type": task_type,
            "server_processing_complete": True
        }
        
        log("Writing encrypted output.")
        with open(output_path, 'w') as f:
            json.dump(result, f)
    except Exception as e:
        err(f"FHE border server pipeline failed")
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