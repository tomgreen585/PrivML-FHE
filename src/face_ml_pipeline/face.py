import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from fc_config import IMAGE_SIZE

"""
face.py

Performs bounding box regression on an input image using a trained face model.

Loads a PyTorch model for bounding box prediction. Converts the input image to a 
normalised tensor. Runs inference to predict normalised bounding box coordinates. 
Draw a green rectance over the predicted region and saves teh output image.
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
    Attempts to load the trained face bounding box model from specified path.

    Returns:
    model (torch.nn.Module): The loaded PyTorch model.
    device (str): Device used for inference (e.g. "cpu").
    """
    
    try:
        from fc_model import ML_Model
        from fc_config import (CURRENT_MODEL)
        
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

def img_to_tensor(pil_img, size, device):
    """
    Converts a PIL RGB image into a normalised PyTorch tensor for model input.
    
    Args:
    pil_img (PIL.Image): input image
    size (int): resize target size
    device (str): device set for inference

    Returns:
    t (torch.Tensor): image in tensor format for model input
    """
    img_resized = pil_img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t

def main():
    """
    Script main that is run by server.js
    Loads the input image, runs the face model, and sends back bounding box coordinates.
    """
    if len(sys.argv) < 3:
        print("Usage: face.py <input_path> <output_path>", flush=True)
        sys.exit(2)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    input_path  = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    try:
        log("Reconstructing data...")
        img = Image.open(input_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        log("Loading face model...")
        model, device = load_model()

        x = img_to_tensor(img, IMAGE_SIZE, device)
        
        log("Running inference...")
        start_time = time.time()
        with torch.no_grad():
            pred = model(x)
        p = pred.detach().cpu().reshape(-1).numpy().astype(float)
        if p.shape[0] != 4:
            raise RuntimeError(f"Expected 4 values, got shape {p.shape}")
        p = np.clip(p, 0.0, 1.0)
        h, w = img.size[1], img.size[0]
        cx, cy, bw, bh = p
        px, py, pw, ph = cx * w, cy * h, bw * w, bh * h
        top_left_pred = (px - pw / 2, py - ph / 2)
        out_img = img.copy()
        draw = ImageDraw.Draw(out_img)
        draw.rectangle(
            [top_left_pred[0],
             top_left_pred[1],
             top_left_pred[0] + pw,
             top_left_pred[1] + ph],
            outline=(0, 255, 0),
            width=4
        )
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
