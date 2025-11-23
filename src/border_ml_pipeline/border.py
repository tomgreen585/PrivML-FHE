import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch

"""
border.py

This script performs inference using a pre-trained border detection model to add visual borders in input images.
It loads a PyTroch model, processes an input image, runs the model, and saves the output.
"""

def log(msg):
    """Prints an info message to stdout to display on web application log outputs."""
    print(f"[INFO] {msg}")
    
def err(msg):
    """Prints an error message to stderr for debugging."""
    sys.stderr.write(f"[ERR] {msg}\n")
    sys.stderr.flush()

def load_model():
    """
    Loads the ML border model from specified path.

    Returns:
    model (torch.nn.Module): loaded model
    device (str): device set for inference
    """
    try:
        from br_model import ML_Model
        from br_config import (CURRENT_MODEL)
        
        model = ML_Model()
        REPO_ROOT = Path(__file__).parent.parent.parent
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
    Convers a PIL image to a PyTorch tensor for model inference.
    
    Args:
    img (PIL.Image): Input image
    device (str): device set for inference

    Returns:
    torch.Tensor: the image tensor shape (1, C, H, W)
    """
    if img.mode == 'RGB':
        img = img.convert('L') #convert to greyscale
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if len(arr.shape) == 2:
        arr = arr[np.newaxis, ...] #add channel dimension
    t = torch.from_numpy(arr).unsqueeze(0).to(device) #add batch dimension
    return t

def tensor_to_img(t):
    """
    Converts a PyTroch tensor back to a PIL image.
    Args:
        t (torch.Tensor): output to tensor

    Returns:
        PIL.Image: The reconstructed output image.
    """
    t = t.detach().clamp(0, 1).cpu().squeeze(0)
    if t.shape[0] == 1: #greyscale output
        t = t.squeeze(0)
        arr = (t.numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    else:
        #rgb output
        arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

def main():
    """
    Script main that is run by server.js
    Loads the input image, runs the border model, saves the output image.
    """
    if len(sys.argv) < 3:
        print("Usage: border.py <input_path> <output_path>", flush=True)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    input_path  = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    try:
        log("Reconstructing data...")
        img = Image.open(input_path).convert("RGB")
        
        log("Loading border model...")
        model, device = load_model()
        
        x = img_to_tensor(img, device)
        
        log("Running inference...")
        start_time = time.time()
        with torch.no_grad():
            y = model(x)
        out_img = tensor_to_img(y)
        end_time = time.time()
        elapsed = end_time - start_time
        log(f"Inference completed in {elapsed:.3f}s")

        log("Writing output...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(output_path, "PNG")
    except Exception as e:
        err(f"Border server pipeline failed")
        traceback.print_exc(file=sys.stderr)
        log(traceback.format_exc())

if __name__ == "__main__":
    main()
