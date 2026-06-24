import os
import time
import torch
import cv2
import numpy as np
from collections import OrderedDict

from image_conditioning import preprocess_image, postprocess_image

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

# Define the models we want to profile
models = ['dehazeformer-t', 'dehazeformer-s', 'dehazeformer-m', 'dehazeformer-b']

# Benchmark a single image to save time
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model_name, img_path):
    print(f"\n--- Benchmarking {model_name} on {img_path} ---", flush=True)
    model_path = os.path.join('./save_models/outdoor', model_name + '.pth')
    if not os.path.exists(model_path):
        print(f"Weights file not found at {model_path}!", flush=True)
        return None
    
    # Load model
    t0 = time.perf_counter()
    from models import dehazeformer_t, dehazeformer_s, dehazeformer_m, dehazeformer_b
    from utils import read_img, hwc_to_chw, chw_to_hwc
    
    network = eval(model_name.replace('-', '_'))()
    network.load_state_dict(load_state_dict(model_path, device))
    network.to(device)
    network.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    load_time = (time.perf_counter() - t0) * 1000.0
    
    num_params = count_parameters(network)
    
    # Read image
    img = read_img(img_path)
    h, w, c = img.shape
    print(f"Original resolution: {w}x{h}", flush=True)
    
    configs = [
        {"name": "Original", "max_side": 0},
        {"name": "Resized (max_side=1280)", "max_side": 1280}
    ]
    
    results = []
    
    for config in configs:
        max_side = config["max_side"]
        
        # Resize if needed
        t_resize_0 = time.perf_counter()
        if max_side > 0:
            scale = max_side / max(h, w)
            if scale < 1:
                new_w = max(2, int(round(w * scale)))
                new_h = max(2, int(round(h * scale)))
                new_w -= new_w % 2
                new_h -= new_h % 2
                test_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                test_img = img.copy()
        else:
            test_img = img.copy()
        resize_time = (time.perf_counter() - t_resize_0) * 1000.0
        
        cur_h, cur_w = test_img.shape[:2]
        
        # Warmup and timing
        try:
            # Warmup iterations
            with torch.no_grad():
                for _ in range(2):
                    warm_img = preprocess_image(test_img, 'none')
                    warm_tensor = torch.from_numpy(hwc_to_chw(warm_img * 2 - 1)).unsqueeze(0).to(device)
                    _ = network(warm_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            # Run benchmarks
            num_runs = 3
            preprocess_times = []
            model_times = []
            postprocess_times = []
            
            with torch.no_grad():
                for _ in range(num_runs):
                    # 1. Preprocess
                    t1 = time.perf_counter()
                    p_img = preprocess_image(test_img, 'none')
                    tensor = torch.from_numpy(hwc_to_chw(p_img * 2 - 1)).unsqueeze(0).to(device)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    preprocess_times.append((t2 - t1) * 1000.0)
                    
                    # 2. Inference
                    t3 = time.perf_counter()
                    output = network(tensor).clamp_(-1, 1)
                    output = output * 0.5 + 0.5
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t4 = time.perf_counter()
                    model_times.append((t4 - t3) * 1000.0)
                    
                    # 3. Postprocess
                    t5 = time.perf_counter()
                    out_img = postprocess_image(chw_to_hwc(output.detach().cpu().squeeze(0).numpy()), 'none')
                    t6 = time.perf_counter()
                    postprocess_times.append((t6 - t5) * 1000.0)
            
            avg_preprocess = np.mean(preprocess_times)
            avg_model = np.mean(model_times)
            avg_postprocess = np.mean(postprocess_times)
            avg_total = avg_preprocess + avg_model + avg_postprocess
            fps = 1000.0 / avg_total
            
            print(f"[{config['name']}] Resolution: {cur_w}x{cur_h}", flush=True)
            print(f"  Preprocess:  {avg_preprocess:.2f} ms", flush=True)
            print(f"  Inference:   {avg_model:.2f} ms", flush=True)
            print(f"  Postprocess: {avg_postprocess:.2f} ms", flush=True)
            print(f"  Total Frame: {avg_total:.2f} ms  ({fps:.2f} FPS)", flush=True)
            
            results.append({
                "config": config["name"],
                "resolution": f"{cur_w}x{cur_h}",
                "params": num_params,
                "preprocess_ms": avg_preprocess,
                "inference_ms": avg_model,
                "postprocess_ms": avg_postprocess,
                "total_ms": avg_total,
                "fps": fps,
                "oom": False
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[{config['name']}] Resolution: {cur_w}x{cur_h} - GPU OOM", flush=True)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                results.append({
                    "config": config["name"],
                    "resolution": f"{cur_w}x{cur_h}",
                    "params": num_params,
                    "preprocess_ms": 0.0,
                    "inference_ms": 0.0,
                    "postprocess_ms": 0.0,
                    "total_ms": 0.0,
                    "fps": 0.0,
                    "oom": True
                })
            else:
                raise e
                
    return {
        "model": model_name,
        "load_time_ms": load_time,
        "params": num_params,
        "results": results
    }

def main():
    from image_conditioning import preprocess_image, postprocess_image
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}!", flush=True)
        return
        
    all_results = []
    for model in models:
        res = benchmark_model(model, image_path)
        if res:
            all_results.append(res)
            
    # Print Markdown Table
    print("\n\n==================== BENCHMARK SUMMARY ====================", flush=True)
    print(f"Image: {image_path}", flush=True)
    print("| Model | Config | Resolution | Params | Preproc (ms) | Inference (ms) | Postproc (ms) | Total (ms) | FPS | Status |", flush=True)
    print("|---|---|---|---|---|---|---|---|---|---|", flush=True)
    for res in all_results:
        for item in res["results"]:
            if item["oom"]:
                print(f"| {res['model']} | {item['config']} | {item['resolution']} | {res['params']:,} | N/A | N/A | N/A | N/A | N/A | OOM |", flush=True)
            else:
                print(f"| {res['model']} | {item['config']} | {item['resolution']} | {res['params']:,} | {item['preprocess_ms']:.2f} | {item['inference_ms']:.2f} | {item['postprocess_ms']:.2f} | {item['total_ms']:.2f} | {item['fps']:.2f} | OK |", flush=True)

if __name__ == '__main__':
    main()
