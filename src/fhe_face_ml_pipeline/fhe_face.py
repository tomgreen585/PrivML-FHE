import sys
import json
import time
import torch
import tenseal as ts
from pathlib import Path
import traceback

"""
fhe_face.py

Performs bounding box regression on an input image using a trained face model wrapped
by an encrypted FHE model.

Loads a PyTorch model for bounding box prediction and wraps it with a FHE model using its
architecture. Converts the input image to a normalised tensor. Runs inference to predict 
normalised bounding box coordinates and saves the coordinates output image.
"""

def log(msg):
    """Log output to stdout"""
    print(f"[INFO] {msg}")
    sys.stdout.flush()

def err(msg):
    """Log output for standardized error output to stderr."""
    sys.stderr.write(f"[ERR] {msg}\n")
    sys.stderr.flush()

def load_face_model():
    """
    Attempts to load the trained face bounding box model and FHE model from specified path.

    Returns:
    fhe_model (torch.nn.Module): loaded FHE_Model
    device (str): Device used for inference (e.g. "cpu").
    """
    log("Loading face detection model...")
    try:
        from fhe_fc_fhe_model import FHE_Model
        from fhe_fc_plain_model import ML_Model
        from fhe_fc_config import (CURRENT_MODEL)

        model = ML_Model()
        REPO_ROOT = Path(__file__).parent.parent.parent

        #path to trained model during ml_pipeline
        candidate_paths = [
            Path(__file__).parent / "models" / CURRENT_MODEL,
            REPO_ROOT / "src" / "fhe_face_ml_pipeline" / "models" / CURRENT_MODEL
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
                    err(f"Failed to load face model.")

        if not model_loaded:
            log("Model failed to load...exiting")
            sys.exit(1)

        fhe_model = FHE_Model(model.eval())
        log("Encrypted model instance created.")
        return fhe_model
    except Exception as e:
        err(f"Could not load FHE model: {e}")
        traceback.print_exc(file=sys.stderr)
        raise

def main():
    """
    Script main that is run by server.js
    Loads the input image, runs the face model, and sends back bounding box coordinates.
    """
    if len(sys.argv) != 3:
        err("Usage: python3 fhe_face.py <input_data.json> <output_result.json>")
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

        log("Reconstructing TenSEAL Data...")
        context = ts.context_from(bytes.fromhex(context_hex))
        channels = [ts.ckks_vector_from(context, bytes.fromhex(ch)) for ch in channels_hex]

        model = load_face_model()

        log("Running encrypted inference...")
        start_time = time.time()
        enc_output = model(channels, windows_nb)
        end_time = time.time()
        inference_time = end_time - start_time
        log(f"Inference complete in {inference_time:.3f}s")

        encrypted_output = enc_output.serialize().hex()

        result = {
            "encrypted_output": encrypted_output,
            "inference_time": inference_time,
            "kernel_shape": kernel_shape,
            "stride": stride,
            "windows_nb": windows_nb,
            "server_processing_complete": True
        }

        log(f"Writing encrypted output.")
        with open(output_path, 'w') as f:
            json.dump(result, f)
    except Exception as e:
        err(f"FHE face server pipeline failed: {e}")
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
