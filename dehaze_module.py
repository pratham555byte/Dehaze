import os
import time
import threading
import cv2
import torch
import numpy as np
from collections import OrderedDict

from models import *
from image_conditioning import preprocess_image, postprocess_image
from utils import chw_to_hwc, hwc_to_chw

def load_state_dict(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned[key] = value
    return cleaned

class DehazeModule:
    def __init__(self, model_name='dehazeformer-t', resolution=320, device='auto'):
        self.model_name = model_name
        self.resolution = resolution
        self.camera_index = 0 # default to Front camera
        self.fp16 = True
        self.preprocess_mode = 'video'
        self.postprocess_mode = 'video'
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.lock = threading.Lock()
        self.latest_orig = None
        self.latest_dehazed = None
        
        self.running = False
        self.thread = None
        self.fps = 0.0
        self.latency = 0.0
        self.is_fallback = False
        
        # Adaptive Dehazing state
        self.adaptive_mode = True
        self.manual_override = False
        self.threshold = 0.20
        self.current_density = 0.0
        self.dehazing_active = False
        self.last_density_check = 0.0
        self.fog_estimator = None
        self._load_fog_estimator()
        
        # Thread-safe config update signals
        self.new_model_name = model_name
        self.new_camera_index = 0
        self.new_resolution = resolution
        self.new_fp16 = True
        self.new_preprocess_mode = 'video'
        self.new_postprocess_mode = 'video'
        self.reconfig_model_flag = False
        self.reconfig_cam_flag = False
        self.reconfig_resolution_flag = False
        self.reconfig_fp16_flag = False
        self.reconfig_preprocess_flag = False
        self.reconfig_postprocess_flag = False
        
        self.fallback_path = os.path.join(
            os.path.dirname(__file__), 
            "Cycling in Foggy weather. Riding To Jaam Gate. #cycling #nature #greenery #fog #mountains #clouds.mp4"
        )

    def _load_fog_estimator(self):
        try:
            from fog_estimator_architecture import FogEstimator
            self.fog_estimator = FogEstimator().to(self.device)
            weights_path = os.path.join(os.path.dirname(__file__), 'fog_estimator.pth')
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                self.fog_estimator.load_state_dict(state_dict)
                self.fog_estimator.eval()
                print("[DehazeModule] Fog density estimator loaded successfully.")
            else:
                print(f"[DehazeModule] Warning: fog_estimator.pth not found at {weights_path}")
        except Exception as e:
            print(f"[DehazeModule] Error loading fog density estimator: {e}")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

    def set_model(self, model_name):
        with self.lock:
            if self.model_name != model_name:
                self.new_model_name = model_name
                self.reconfig_model_flag = True
                print(f"[DehazeModule] Set model request: {model_name}")

    def set_camera(self, index):
        with self.lock:
            if self.camera_index != index:
                self.new_camera_index = index
                self.reconfig_cam_flag = True
                print(f"[DehazeModule] Set camera request: {index}")

    def set_resolution(self, resolution):
        with self.lock:
            if self.resolution != resolution:
                self.new_resolution = resolution
                self.reconfig_resolution_flag = True
                print(f"[DehazeModule] Set resolution request: {resolution}")

    def set_fp16(self, enabled):
        with self.lock:
            if self.fp16 != enabled:
                self.new_fp16 = enabled
                self.reconfig_fp16_flag = True
                print(f"[DehazeModule] Set FP16 request: {enabled}")

    def set_preprocess(self, mode):
        with self.lock:
            if self.preprocess_mode != mode:
                self.new_preprocess_mode = mode
                self.reconfig_preprocess_flag = True
                print(f"[DehazeModule] Set preprocess request: {mode}")

    def set_postprocess(self, mode):
        with self.lock:
            if self.postprocess_mode != mode:
                self.new_postprocess_mode = mode
                self.reconfig_postprocess_flag = True
                print(f"[DehazeModule] Set postprocess request: {mode}")

    def set_adaptive_mode(self, enabled):
        with self.lock:
            self.adaptive_mode = enabled
            print(f"[DehazeModule] Set adaptive_mode: {enabled}")

    def set_manual_override(self, enabled):
        with self.lock:
            self.manual_override = enabled
            print(f"[DehazeModule] Set manual_override: {enabled}")

    def set_threshold(self, threshold):
        with self.lock:
            self.threshold = threshold
            print(f"[DehazeModule] Set threshold: {threshold}")

    def get_fog_stats(self):
        with self.lock:
            return {
                "adaptive_mode": self.adaptive_mode,
                "manual_override": self.manual_override,
                "threshold": self.threshold,
                "current_density": self.current_density,
                "dehazing_active": self.dehazing_active
            }

    def _load_network(self, model_name):
        print(f"[DehazeModule] Loading model {model_name} on {self.device}...")
        model_path = os.path.join(os.path.dirname(__file__), 'save_models/outdoor', f"{model_name}.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), 'saved_models/outdoor', f"{model_name}.pth")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights not found for model: {model_name}")
            
        network = eval(model_name.replace('-', '_'))()
        network.load_state_dict(load_state_dict(model_path, self.device))
        network.to(self.device)
        network.eval()
        return network

    def _open_camera(self, index):
        print(f"[DehazeModule] Opening camera index {index}...")
        cap = cv2.VideoCapture(index)
        webcam_ok = False
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                webcam_ok = True
                self.is_fallback = False
                print(f"[DehazeModule] Webcam {index} opened successfully.")
            else:
                cap.release()
                
        if not webcam_ok:
            print(f"[DehazeModule] Webcam {index} failed. Using fallback video.")
            cap = cv2.VideoCapture(self.fallback_path)
            self.is_fallback = True
            
        return cap

    def _run_loop(self):
        # 1. Load initial model
        try:
            network = self._load_network(self.model_name)
            # Apply initial FP16 if CUDA
            if self.fp16 and self.device.type == 'cuda':
                network.half()
        except Exception as e:
            print(f"[DehazeModule] Error initializing model: {e}")
            self.running = False
            return

        # 2. Open initial camera
        cap = self._open_camera(self.camera_index)
        frame_timestamps = []
        use_fp16 = self.fp16 and self.device.type == 'cuda'
        
        while self.running:
            t_start = time.perf_counter()
            
            # --- CHECK DYNAMIC CONFIG RE-CONFIGS ---
            reconfig_model = False
            reconfig_cam = False
            reconfig_fp16 = False
            with self.lock:
                if self.reconfig_model_flag:
                    self.model_name = self.new_model_name
                    self.reconfig_model_flag = False
                    reconfig_model = True
                if self.reconfig_cam_flag:
                    self.camera_index = self.new_camera_index
                    self.reconfig_cam_flag = False
                    reconfig_cam = True
                if self.reconfig_resolution_flag:
                    self.resolution = self.new_resolution
                    self.reconfig_resolution_flag = False
                if self.reconfig_fp16_flag:
                    self.fp16 = self.new_fp16
                    self.reconfig_fp16_flag = False
                    reconfig_fp16 = True
                if self.reconfig_preprocess_flag:
                    self.preprocess_mode = self.new_preprocess_mode
                    self.reconfig_preprocess_flag = False
                if self.reconfig_postprocess_flag:
                    self.postprocess_mode = self.new_postprocess_mode
                    self.reconfig_postprocess_flag = False
                    
            if reconfig_model or reconfig_fp16:
                try:
                    network = self._load_network(self.model_name)
                    use_fp16 = self.fp16 and self.device.type == 'cuda'
                    if use_fp16:
                        network.half()
                except Exception as e:
                    print(f"[DehazeModule] Error reloading model: {e}")
                    
            if reconfig_cam:
                cap.release()
                cap = self._open_camera(self.camera_index)
                
            # Read frame
            ret, frame = cap.read()
            if not ret:
                if self.is_fallback:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[DehazeModule] Video stream read failure. Re-opening...")
                    cap.release()
                    cap = self._open_camera(self.camera_index)
                    continue

            # Resize frame
            h_orig, w_orig = frame.shape[:2]
            scale = self.resolution / max(h_orig, w_orig)
            if scale < 1.0:
                new_w = max(2, int(round(w_orig * scale)))
                new_h = max(2, int(round(h_orig * scale)))
                new_w -= new_w % 2
                new_h -= new_h % 2
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame.copy()

            frame_resized_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # --- ADAPTIVE DEHAZING: FOG DENSITY ESTIMATION CHECK ---
            now_time = time.perf_counter()
            with self.lock:
                adaptive = self.adaptive_mode
                manual = self.manual_override
                thresh = self.threshold
                
            if self.fog_estimator is not None and (now_time - self.last_density_check >= 60.0 or self.last_density_check == 0.0):
                self.last_density_check = now_time
                try:
                    # Prep image for ResNet-18 (resize to 224x224 and normalize using ImageNet stats)
                    img_224 = cv2.resize(frame_resized_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                    img_t = img_224.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img_t = (img_t - mean) / std
                    
                    tensor_input = torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        score = float(self.fog_estimator(tensor_input).item())
                    
                    with self.lock:
                        self.current_density = score
                    print(f"[DehazeModule] Fog density: {score:.4f} (threshold: {thresh:.2f})")
                except Exception as est_err:
                    print(f"[DehazeModule] Error in fog density estimation: {est_err}")
            
            # Decide if we run DehazeFormer
            active_dehaze = False
            with self.lock:
                if manual:
                    active_dehaze = True
                elif adaptive:
                    active_dehaze = (self.current_density > thresh)
                else:
                    active_dehaze = False
                self.dehazing_active = active_dehaze

            # Dehazing inference
            t_infer_start = time.perf_counter()
            if active_dehaze:
                try:
                    img_float = frame_resized_rgb.astype(np.float32) / 255.0
                    img_pre = preprocess_image(img_float, self.preprocess_mode)
                    
                    with torch.no_grad():
                        tensor = torch.from_numpy(hwc_to_chw(img_pre * 2 - 1)).unsqueeze(0).to(self.device)
                        if use_fp16:
                            tensor = tensor.half()
                        output = network(tensor).clamp_(-1, 1)
                        output = output * 0.5 + 0.5
                        
                    out_float = chw_to_hwc(output.detach().cpu().squeeze(0).float().numpy())
                    out_post = postprocess_image(out_float, self.postprocess_mode)
                    dehazed_uint8 = np.clip(out_post * 255.0, 0, 255).astype(np.uint8)
                except Exception as e:
                    # print(f"[DehazeModule] Inference error: {e}")
                    dehazed_uint8 = frame_resized_rgb.copy()
            else:
                # Bypass: use the original resized frame (convert back to uint8 BGR if needed, but here it's already RGB)
                dehazed_uint8 = frame_resized_rgb.copy()
                # Simulate small bypass delay or just use 0 latency
                
            t_end = time.perf_counter()
            latency_ms = (t_end - t_infer_start) * 1000.0 if active_dehaze else 0.0
            
            with self.lock:
                self.latest_orig = frame_resized_rgb
                self.latest_dehazed = dehazed_uint8
                self.latency = latency_ms

            # FPS
            frame_timestamps.append(t_end)
            now = time.perf_counter()
            frame_timestamps = [ft for ft in frame_timestamps if now - ft < 1.0]
            self.fps = len(frame_timestamps)
            
            if self.is_fallback:
                elapsed = t_end - t_start
                target_period = 1.0 / 24.0
                if elapsed < target_period:
                    time.sleep(target_period - elapsed)
            else:
                time.sleep(0.001)

        cap.release()
        print("[DehazeModule] Thread stopped.")

    def get_frames(self):
        with self.lock:
            return self.latest_orig, self.latest_dehazed, self.fps, self.latency
