import os
import time
import torch
import cv2
import numpy as np
from collections import OrderedDict

# Set device to CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

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

def benchmark_run(network, tensor, fp16=False, runs=3):
    # Warmup
    with torch.inference_mode():
        for _ in range(2):
            _ = network(tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
    # Measure
    times = []
    with torch.inference_mode():
        for _ in range(runs):
            t1 = time.perf_counter()
            _ = network(tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            times.append((t2 - t1) * 1000.0)
    return np.mean(times)

def main():
    from models import dehazeformer_s
    from utils import read_img, hwc_to_chw
    
    img = read_img(image_path)
    h, w, c = img.shape
    
    model_name = "dehazeformer-s"
    model_path = os.path.join('./save_models/outdoor', model_name + '.pth')
    if not os.path.exists(model_path):
        print("Model file not found!")
        return

    print(f"\nEvaluating optimizations for {model_name} on {image_path}:", flush=True)
    
    # Configurations to test
    # We will test:
    # 1. Baseline FP32 (max_side=1280)
    # 2. FP16 (max_side=1280)
    # 3. FP32 + torch.compile (max_side=1280)
    # 4. FP16 + torch.compile (max_side=1280)
    # 5. FP16 (max_side=640)
    # 6. FP16 + torch.compile (max_side=640)
    
    for max_side in [1280, 640]:
        scale = max_side / max(h, w)
        new_w = max(2, int(round(w * scale)))
        new_h = max(2, int(round(h * scale)))
        new_w -= new_w % 2
        new_h -= new_h % 2
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # FP32 Tensor
        tensor_fp32 = torch.from_numpy(hwc_to_chw(resized_img * 2 - 1)).unsqueeze(0).to(device).float()
        # FP16 Tensor
        tensor_fp16 = tensor_fp32.half()
        
        print(f"\n--- Resolution: {new_w}x{new_h} ---", flush=True)
        
        # A. Baseline FP32 Model
        try:
            net_fp32 = dehazeformer_s()
            net_fp32.load_state_dict(load_state_dict(model_path, device))
            net_fp32.to(device).eval()
            
            t_fp32 = benchmark_run(net_fp32, tensor_fp32, fp16=False)
            print(f"FP32 Baseline: {t_fp32:.2f} ms ({1000/t_fp32:.2f} FPS)", flush=True)
            
            # Try torch.compile on FP32
            print("Compiling FP32 model (this may take a minute)...", flush=True)
            t_compile_start = time.perf_counter()
            net_fp32_compiled = torch.compile(net_fp32)
            # Run once to compile
            _ = net_fp32_compiled(tensor_fp32)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            compile_time = time.perf_counter() - t_compile_start
            print(f"FP32 Compiled in {compile_time:.1f}s", flush=True)
            
            t_fp32_compiled = benchmark_run(net_fp32_compiled, tensor_fp32, fp16=False)
            print(f"FP32 + torch.compile: {t_fp32_compiled:.2f} ms ({1000/t_fp32_compiled:.2f} FPS)", flush=True)
            
            del net_fp32, net_fp32_compiled
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"FP32 failed: {e}", flush=True)
            
        # B. FP16 Model
        try:
            net_fp16 = dehazeformer_s()
            net_fp16.load_state_dict(load_state_dict(model_path, device))
            net_fp16.to(device).half().eval()
            
            t_fp16 = benchmark_run(net_fp16, tensor_fp16, fp16=True)
            print(f"FP16: {t_fp16:.2f} ms ({1000/t_fp16:.2f} FPS)", flush=True)
            
            # Try torch.compile on FP16
            print("Compiling FP16 model (this may take a minute)...", flush=True)
            t_compile_start = time.perf_counter()
            net_fp16_compiled = torch.compile(net_fp16)
            _ = net_fp16_compiled(tensor_fp16)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            compile_time = time.perf_counter() - t_compile_start
            print(f"FP16 Compiled in {compile_time:.1f}s", flush=True)
            
            t_fp16_compiled = benchmark_run(net_fp16_compiled, tensor_fp16, fp16=True)
            print(f"FP16 + torch.compile: {t_fp16_compiled:.2f} ms ({1000/t_fp16_compiled:.2f} FPS)", flush=True)
            
            del net_fp16, net_fp16_compiled
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"FP16 failed: {e}", flush=True)

if __name__ == '__main__':
    main()
