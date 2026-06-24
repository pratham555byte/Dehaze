import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
# Importing the specific variants as per your local file structure
from models import dehazeformer_t, dehazeformer_s, dehazeformer_m, dehazeformer_b

# --- Configurations ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIR = 'bali'
# Updated to your specific path
MODEL_ROOT = 'saved_models-20260507T102627Z-3-001/saved_models/outdoor'
OUTPUT_DIR = 'comparison_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map filenames/folders to the correct class
MODEL_MAP = {
    'dehazeformer_t': dehazeformer_t,
    'dehazeformer_s': dehazeformer_s,
    'dehazeformer_m': dehazeformer_m,
    'dehazeformer_b': dehazeformer_b
}

def get_model_type(filename):
    """Detects model type from string to initialize correct architecture."""
    name = filename.lower()
    if '_t' in name or 'tiny' in name: return 'dehazeformer_t'
    if '_s' in name or 'small' in name: return 'dehazeformer_s'
    if '_m' in name or 'medium' in name: return 'dehazeformer_m'
    if '_b' in name or 'big' in name: return 'dehazeformer_b'
    return 'dehazeformer_s' # Default fallback

def process_all_models():
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Walk through all subfolders
    for root, dirs, files in os.walk(MODEL_ROOT):
        for file in files:
            if file.endswith('.pth'):
                ckpt_path = os.path.join(root, file)
                
                # Determine model variant and initialize
                model_type = get_model_type(file)
                print(f"\n--- Loading {model_type} from {file} ---")
                
                try:
                    # Initialize the specific model function/class
                    model = MODEL_MAP[model_type]().to(DEVICE)
                    model.eval()

                    # Load weights
                    state_dict = torch.load(ckpt_path, map_location=DEVICE)
                    # Handle both full checkpoints and weight-only dicts
                    actual_weights = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
                    model.load_state_dict(actual_weights)
                    
                    # Create unique output subfolder based on the path to avoid overwrites
                    relative_path = os.path.relpath(root, MODEL_ROOT).replace(os.sep, '_')
                    ckpt_output_dir = os.path.join(OUTPUT_DIR, f"{relative_path}_{file.replace('.pth', '')}")
                    os.makedirs(ckpt_output_dir, exist_ok=True)

                    for img_name in images:
                        img_path = os.path.join(INPUT_DIR, img_name)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Pad image to be divisible by 8 (DehazeFormer requirement)
                        h, w, _ = img.shape
                        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
                        img_padded = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                        
                        input_tensor = to_tensor(img_padded).unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                        
                        # Post-process and crop back to original size
                        output_img = to_pil_image(output.squeeze(0).cpu())
                        output_img = output_img.crop((0, 0, w, h)) 
                        
                        save_path = os.path.join(ckpt_output_dir, img_name)
                        output_img.save(save_path)
                        print(f" Saved: {save_path}")
                        
                except Exception as e:
                    print(f" Error processing {file}: {e}")

if __name__ == "__main__":
    process_all_models()