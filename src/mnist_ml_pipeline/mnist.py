import sys
import json
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch

"""
mnist.py

Loads a model from the specified path. Converts a given image to a tensor. Performs
inference and outputs the predicted label to a JSON file.
"""

def log(msg):
    """Log output to stdout"""
    print(f"[INFO] {msg}")
    
def err(msg):
    """Log output for standardized error output to stderr."""
    sys.stderr.write(f"[ERR] {msg}\n")
    sys.stderr.flush()

def load_model():
    """
    Attempts to load the mnist model from specified path.

    Returns:
    model (torch.nn.Module): The loaded PyTorch model.
    device (str): Device used for inference (e.g. "cpu").
    """
    try:
        from mn_model import ML_Model
        from mn_config import (CURRENT_MODEL)
        
        model = ML_Model()
        REPO_ROOT = Path(__file__).parent.parent.parent
        
        #path to trained model during ml_pipeline
        candidate_paths = [
            Path(__file__).parent / "models" / CURRENT_MODEL,
            REPO_ROOT / "src" / "border_ml_pipeline" / "models" / CURRENT_MODEL,
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
        log("Model instance created.")
        return model, device
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise

def img_to_tensor(img, device):
    """
    Converts a PIL RGB image into a normalised PyTorch tensor for model input.
    
    Args:
    pil_img (PIL.Image): input image
    size (int): resize target size
    device (str): device set for inference

    Returns:
    t (torch.Tensor): image in tensor format for model input
    """
    g = img.convert("L")
    g = g.resize((28, 28), Image.BILINEAR)
    x = np.asarray(g, dtype=np.float32) / 255.0
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    return t

def main():
    """
    Script main that is run by server.js
    Loads the input image, runs the mnist model, and sends back the output.
    """
    if len(sys.argv) < 3:
        print("Usage: mnist.py <input_path> <output_json_path>", flush=True)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    input_path  = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    try:
        log("Reconstructing data...")
        img = Image.open(input_path).convert("RGB")

        log("Loading mnist model...")
        model, device = load_model()

        x = img_to_tensor(img, device)
        
        log("Running inference…")
        start_time = time.time()
        with torch.no_grad():
            y = model(x)
            pred = torch.argmax(y, dim=1).item()
        end_time = time.time()
        elapsed = end_time - start_time
        log(f"Inference completed in {elapsed:.3f}s")

        log("Writing output…")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"label": int(pred)}, f)
    except Exception as e:
        err(f"Border server pipeline failed")
        traceback.print_exc(file=sys.stderr)
        log(traceback.format_exc())

if __name__ == "__main__":
    main()
