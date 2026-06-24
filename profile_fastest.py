import os
import time
import torch
import cv2
import numpy as np
from collections import OrderedDict

# Set device to CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_path = "./image seq/00000.JPG"

def load_state_dict(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned[key] = value
    return cleaned

def main():
    from models import dehazeformer_t
    from utils import read_img, hwc_to_chw, chw_to_hwc
    from image_conditioning import preprocess_image, postprocess_image
    
    img = read_img(image_path)
    h, w, c = img.shape
    
    model_name = "dehazeformer-t"
    model_path = os.path.join('./save_models/outdoor', model_name + '.pth')
    
    # Initialize models
    net_fp32 = dehazeformer_t()
    net_fp32.load_state_dict(load_state_dict(model_path, device))
    net_fp32.to(device).eval()
    
    net_fp16 = dehazeformer_t()
    net_fp16.load_state_dict(load_state_dict(model_path, device))
    net_fp16.to(device).half().eval()
    
    resolutions = [
        ("Original", 0),
        ("Standard Resize (1280)", 1280),
        ("Fast Resize (640)", 640)
    ]
    
    print("\n==================== FASTEST MODEL (DEHAZEFORMER-T) BENCHMARKS ====================", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    
    for label, max_side in resolutions:
        # Resize
        if max_side > 0:
            scale = max_side / max(h, w)
            new_w = max(2, int(round(w * scale)))
            new_h = max(2, int(round(h * scale)))
            new_w -= new_w % 2
            new_h -= new_h % 2
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h, new_w = h, w
            resized_img = img.copy()
            
        tensor_fp32 = torch.from_numpy(hwc_to_chw(resized_img * 2 - 1)).unsqueeze(0).to(device).float()
        tensor_fp16 = tensor_fp32.half()
        
        # Benchmarking FP32
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = net_fp32(tensor_fp32)
            torch.cuda.synchronize()
        # Measure
        t_fp32_runs = []
        with torch.no_grad():
            for _ in range(5):
                t1 = time.perf_counter()
                out = net_fp32(tensor_fp32)
                torch.cuda.synchronize()
                t_fp32_runs.append((time.perf_counter() - t1) * 1000.0)
        avg_fp32 = np.mean(t_fp32_runs)
        
        # Benchmarking FP16
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = net_fp16(tensor_fp16)
            torch.cuda.synchronize()
        # Measure
        t_fp16_runs = []
        with torch.no_grad():
            for _ in range(5):
                t1 = time.perf_counter()
                out = net_fp16(tensor_fp16)
                torch.cuda.synchronize()
                t_fp16_runs.append((time.perf_counter() - t1) * 1000.0)
        avg_fp16 = np.mean(t_fp16_runs)
        
        print(f"\nResolution: {new_w}x{new_h} ({label})", flush=True)
        print(f"  FP32 Inference: {avg_fp32:.2f} ms ({1000/avg_fp32:.2f} FPS)", flush=True)
        print(f"  FP16 Inference: {avg_fp16:.2f} ms ({1000/avg_fp16:.2f} FPS)", flush=True)

if __name__ == '__main__':
    main()
