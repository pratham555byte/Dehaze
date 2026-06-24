import os
import time
import threading
import subprocess
import cv2
import torch
import numpy as np
from collections import OrderedDict
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, send_file
from werkzeug.utils import secure_filename
import imageio_ffmpeg
from io import BytesIO

# Import local modules
from image_conditioning import preprocess_image, postprocess_image
from models import *
from utils import chw_to_hwc, hwc_to_chw

from dehaze_module import DehazeModule
from esp32_module import ESP32Communication
from sensor_module import SensorProcessing
from acc_module import AdaptiveCruiseControl
from digital_twin import DigitalTwin

# Initialize Flask
app = Flask(__name__)

# Folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Global state for pre-recorded video progress
progress_lock = threading.Lock()
progress_state = {
    "status": "idle",
    "current_frame": 0,
    "total_frames": 0,
    "fps": 0.0,
    "eta": 0.0,
    "error": None,
    "output_video": None,
    "comparison_video": None
}

# Live preview during video processing
latest_process_frame = None
latest_process_frame_lock = threading.Lock()

def update_progress(status=None, current_frame=None, total_frames=None, fps=None, eta=None, error=None, output_video=None, comparison_video=None):
    with progress_lock:
        if status is not None: progress_state["status"] = status
        if current_frame is not None: progress_state["current_frame"] = current_frame
        if total_frames is not None: progress_state["total_frames"] = total_frames
        if fps is not None: progress_state["fps"] = fps
        if eta is not None: progress_state["eta"] = eta
        if error is not None: progress_state["error"] = error
        if output_video is not None: progress_state["output_video"] = output_video
        if comparison_video is not None: progress_state["comparison_video"] = comparison_video

def load_state_dict(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned[key] = value
    return cleaned

# ==============================================================================
# ORIGINAL LIVE IP CAMERA MODE BACKEND SYSTEM
# ==============================================================================

class LiveFrameGrabber(threading.Thread):
    def __init__(self, stream_url):
        super().__init__()
        self.stream_url = stream_url
        self.capture = cv2.VideoCapture(stream_url)
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.connected = False
        self.daemon = True

    def run(self):
        while self.running:
            if not self.capture.isOpened():
                self.connected = False
                time.sleep(1.0)
                self.capture.open(self.stream_url)
                continue
            
            ok, frame = self.capture.read()
            if not ok:
                self.connected = False
                time.sleep(0.01)
                continue
                
            self.connected = True
            t = time.perf_counter()
            with self.lock:
                self.latest_frame = (frame.copy(), t)

    def get_latest(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        if self.capture.isOpened():
            self.capture.release()

class LiveStreamManager:
    def __init__(self):
        self.grabber = None
        self.processor_thread = None
        self.active = False
        
        self.model_name = 'dehazeformer-t'
        self.resolution = 640
        self.fp16 = True
        self.preprocess_mode = 'video'
        self.postprocess_mode = 'video'
        
        self.latest_mjpeg = None
        self.latest_dehazed = None
        self.frame_ready_event = threading.Event()
        
        self.stats_lock = threading.Lock()
        self.fps = 0.0
        self.latency = 0.0
        
        self.recording_lock = threading.Lock()
        self.recording_process = None
        self.recording_path = None
        self.recording_session = None
        self.recording_w = 0
        self.recording_h = 0

    def start(self, url, model_name, resolution, fp16, preprocess_mode, postprocess_mode):
        self.stop()
        
        self.model_name = model_name
        self.resolution = resolution
        self.fp16 = fp16
        self.preprocess_mode = preprocess_mode
        self.postprocess_mode = postprocess_mode
        
        self.grabber = LiveFrameGrabber(url)
        self.grabber.start()
        
        self.active = True
        self.processor_thread = threading.Thread(target=self._processing_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()

    def stop(self):
        self.active = False
        self.stop_recording()
        
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
            self.processor_thread = None
            
        if self.grabber:
            self.grabber.stop()
            self.grabber = None
            
        self.latest_mjpeg = None
        self.latest_dehazed = None
        self.frame_ready_event.clear()
        
        with self.stats_lock:
            self.fps = 0.0
            self.latency = 0.0

    def _processing_loop(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'save_models/outdoor', self.model_name + '.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), 'saved_models/outdoor', self.model_name + '.pth')
                
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at {model_path}")
                
            network = eval(self.model_name.replace('-', '_'))()
            network.load_state_dict(load_state_dict(model_path, device))
            network.to(device)
            if self.fp16 and device.type == 'cuda':
                network.half()
            network.eval()
        except Exception as e:
            print(f"Error loading model in live stream loop: {e}")
            self.active = False
            return

        last_timestamp = 0.0
        frame_timestamps = []
        
        while self.active:
            if not self.grabber or not self.grabber.connected:
                time.sleep(0.05)
                continue
                
            latest = self.grabber.get_latest()
            if not latest:
                time.sleep(0.01)
                continue
                
            frame, t_captured = latest
            if t_captured == last_timestamp:
                time.sleep(0.002)
                continue
                
            last_timestamp = t_captured
            t_proc_start = time.perf_counter()
            
            h_orig, w_orig = frame.shape[:2]
            scale = self.resolution / max(h_orig, w_orig)
            if scale < 1:
                new_w = max(2, int(round(w_orig * scale)))
                new_h = max(2, int(round(h_orig * scale)))
                new_w -= new_w % 2
                new_h -= new_h % 2
                img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                new_w, new_h = w_orig, h_orig
                img = frame.copy()
                
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # Apply preprocessing
                img_pre = preprocess_image(img_rgb, self.preprocess_mode)
                
                with torch.no_grad():
                    tensor = torch.from_numpy(hwc_to_chw(img_pre * 2 - 1)).unsqueeze(0).to(device)
                    if self.fp16 and device.type == 'cuda':
                        tensor = tensor.half()
                    output = network(tensor).clamp_(-1, 1)
                    output = output * 0.5 + 0.5
                    
                out_img = postprocess_image(chw_to_hwc(output.detach().cpu().squeeze(0).float().numpy()), self.postprocess_mode)
                out_uint8 = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
                
                self.latest_dehazed = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
                
                # Side-by-side compile
                orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                combined = np.concatenate([orig_rgb, out_uint8], axis=1)
                combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                
                ok_enc, jpeg_bytes = cv2.imencode('.jpg', combined_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_enc:
                    self.latest_mjpeg = jpeg_bytes.tobytes()
                    self.frame_ready_event.set()
                    self.frame_ready_event.clear()
                    
                with self.recording_lock:
                    if self.recording_process:
                        if new_w == self.recording_w and new_h == self.recording_h:
                            try:
                                self.recording_process.stdin.write(out_uint8.tobytes())
                            except Exception as rec_err:
                                print(f"Error piping frame to live recording: {rec_err}")
                                
            except Exception as proc_err:
                print(f"Error processing live stream frame: {proc_err}")
                
            t_proc_end = time.perf_counter()
            cur_latency = (t_proc_end - t_captured) * 1000.0
            
            frame_timestamps.append(t_proc_end)
            now = time.perf_counter()
            frame_timestamps = [ft for ft in frame_timestamps if now - ft < 1.0]
            
            with self.stats_lock:
                self.latency = cur_latency
                self.fps = len(frame_timestamps)

        if device.type == 'cuda':
            del network
            torch.cuda.empty_cache()

    def start_recording(self):
        with self.recording_lock:
            if not self.active or self.latest_dehazed is None:
                return False, "Stream must be active to record."
            if self.recording_process:
                return True, "Recording is already active."
                
            h, w = self.latest_dehazed.shape[:2]
            self.recording_w = w
            self.recording_h = h
            
            session_id = str(int(time.time()))
            self.recording_session = session_id
            
            output_dir = os.path.join(app.config['RESULTS_FOLDER'], f'live_record_{session_id}')
            os.makedirs(output_dir, exist_ok=True)
            self.recording_path = os.path.join(output_dir, 'recorded.mp4')
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{w}x{h}',
                '-r', '12.0',
                '-i', '-',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                self.recording_path
            ]
            
            try:
                self.recording_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True, "Recording successfully initialized."
            except Exception as e:
                return False, f"FFmpeg connection failed: {e}"

    def stop_recording(self):
        with self.recording_lock:
            if not self.recording_process:
                return None
                
            try:
                self.recording_process.stdin.close()
                self.recording_process.wait()
            except:
                pass
                
            self.recording_process = None
            url = f"/results/live_record_{self.recording_session}/recorded.mp4"
            self.recording_session = None
            return url

# Initialize IP camera live manager instance
live_manager = LiveStreamManager()

# ==============================================================================
# ORIGINAL PRE-RECORDED VIDEO DEHAZING SYSTEM
# ==============================================================================

def run_dehaze_thread(video_path, model_name, resolution, fp16, preprocess_mode, postprocess_mode, comparison):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    session_id = str(int(time.time()))
    output_session_dir = os.path.join(app.config['RESULTS_FOLDER'], f'web_dehaze_{session_id}')
    os.makedirs(output_session_dir, exist_ok=True)
    
    capture = None
    process_out = None
    process_comp = None
    
    try:
        global latest_process_frame
        update_progress(status="extracting", current_frame=0, total_frames=0, error=None)
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f'Could not open video: {video_path}')
            
        fps_in = capture.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        
        duration = frame_count / fps_in
        target_frame_count = int(round(duration * 12.0))
        if target_frame_count < 1:
            target_frame_count = 1
            
        keep_indices = set(int(round(k * (fps_in / 12.0))) for k in range(target_frame_count))
        total_targets = len(keep_indices)
        
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Could not read any frames from the input video.")
            
        h_orig, w_orig = frame.shape[:2]
        scale = resolution / max(h_orig, w_orig)
        if scale < 1:
            new_w = max(2, int(round(w_orig * scale)))
            new_h = max(2, int(round(h_orig * scale)))
            new_w -= new_w % 2
            new_h -= new_h % 2
        else:
            new_w, new_h = w_orig, h_orig
            
        update_progress(status="loading_model")
        model_path = os.path.join(os.path.dirname(__file__), 'save_models/outdoor', model_name + '.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), 'saved_models/outdoor', model_name + '.pth')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        network = eval(model_name.replace('-', '_'))()
        network.load_state_dict(load_state_dict(model_path, device))
        network.to(device)
        if fp16 and device.type == 'cuda':
            network.half()
        network.eval()
        
        out_video_name = 'dehazed.mp4'
        out_video_path = os.path.join(output_session_dir, out_video_name)
        
        cmd_out = [
            ffmpeg_exe,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{new_w}x{new_h}',
            '-r', '12.0',
            '-i', '-',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            out_video_path
        ]
        
        process_out = subprocess.Popen(cmd_out, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        process_comp = None
        comp_video_name = None
        if comparison:
            comp_video_name = 'comparison.mp4'
            comp_video_path = os.path.join(output_session_dir, comp_video_name)
            
            cmd_comp = [
                ffmpeg_exe,
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{new_w * 2}x{new_h}',
                '-r', '12.0',
                '-i', '-',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                comp_video_path
            ]
            process_comp = subprocess.Popen(cmd_comp, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        update_progress(status="dehazing", current_frame=0, total_frames=total_targets)
        
        current_idx = 0
        written = 0
        start_time = time.perf_counter()
        
        with torch.no_grad():
            while ok:
                if current_idx in keep_indices:
                    if scale < 1:
                        img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        img = frame.copy()
                        
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    
                    img_pre = preprocess_image(img_rgb, preprocess_mode)
                    
                    tensor = torch.from_numpy(hwc_to_chw(img_pre * 2 - 1)).unsqueeze(0).to(device)
                    if fp16 and device.type == 'cuda':
                        tensor = tensor.half()
                        
                    output = network(tensor).clamp_(-1, 1)
                    output = output * 0.5 + 0.5
                    
                    out_img = postprocess_image(chw_to_hwc(output.detach().cpu().squeeze(0).float().numpy()), postprocess_mode)
                    out_uint8 = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
                    
                    process_out.stdin.write(out_uint8.tobytes())
                    
                    if process_comp:
                        orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        comp_frame = np.concatenate([orig_rgb, out_uint8], axis=1)
                        process_comp.stdin.write(comp_frame.tobytes())
                        with latest_process_frame_lock:
                            latest_process_frame = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2BGR)
                    else:
                        with latest_process_frame_lock:
                            latest_process_frame = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
                        
                    written += 1
                    
                    elapsed = time.perf_counter() - start_time
                    cur_fps = written / max(elapsed, 0.001)
                    eta = (total_targets - written) / max(cur_fps, 0.001)
                    
                    update_progress(current_frame=written, total_frames=total_targets, fps=cur_fps, eta=eta)
                    
                current_idx += 1
                ok, frame = capture.read()
                
        update_progress(status="compiling")
        
        if process_out:
            process_out.stdin.close()
            process_out.wait()
            process_out = None
            
        if process_comp:
            process_comp.stdin.close()
            process_comp.wait()
            process_comp = None
            
        if device.type == 'cuda':
            del network
            torch.cuda.empty_cache()
            
        update_progress(
            status="completed",
            output_video=f"/results/web_dehaze_{session_id}/{out_video_name}",
            comparison_video=f"/results/web_dehaze_{session_id}/{comp_video_name}" if comparison else None
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if process_out:
            try: process_out.kill()
            except: pass
        if process_comp:
            try: process_comp.kill()
            except: pass
        if device.type == 'cuda':
            try: del network
            except: pass
            torch.cuda.empty_cache()
        update_progress(status="error", error=str(e))
        
    finally:
        with latest_process_frame_lock:
            latest_process_frame = None
        if capture:
            capture.release()

# ==============================================================================
# DIGITAL TWIN & REAL-TIME ACC SAFETY MODULES
# ==============================================================================

dehaze_mod = DehazeModule(model_name='dehazeformer-t', resolution=320, device='auto')
esp32_comm = ESP32Communication(esp32_ip="192.168.4.1", esp32_port=5005, local_port=5006)
sensor_proc = SensorProcessing(ema_alpha=0.18, max_valid_range=300.0)
acc_system = AdaptiveCruiseControl(target_distance=50.0, critical_distance=25.0, Kp=4.5, Kd=90.0)
twin_sim = DigitalTwin(width=580, height=480)

# Global Telemetry/Control states
state_lock = threading.Lock()
keyboard_state = {"w": False, "s": False, "a": False, "d": False, "space": False}
vehicle_controls = {"throttle": 0.0, "steering": 0.0}
acc_enabled = False
acc_status_text = "MANUAL"
relative_velocity = 0.0
latest_twin_jpeg = None

def twin_processing_loop():
    """
    Headless Pygame render loop running in the background at 30 FPS.
    """
    global latest_twin_jpeg, acc_status_text, relative_velocity, acc_enabled
    import pygame
    pygame.init()
    
    last_time = time.perf_counter()
    max_speed = 200.0
    max_steering = 45.0
    speed_step = 25.0 * 8
    
    while True:
        current_time = time.perf_counter()
        dt = current_time - last_time
        last_time = current_time
        if dt <= 0:
            dt = 0.001
            
        with state_lock:
            w = keyboard_state["w"]
            s = keyboard_state["s"]
            a = keyboard_state["a"]
            d = keyboard_state["d"]
            space = keyboard_state["space"]
            
        target_steer = 0.0
        if a:
            target_steer = -max_steering
        elif d:
            target_steer = max_steering
            
        cur_steer = vehicle_controls["steering"]
        steer_diff = target_steer - cur_steer
        if steer_diff != 0:
            step = 90.0 * dt * (1.0 if steer_diff > 0 else -1.0)
            if abs(step) > abs(steer_diff):
                cur_steer = target_steer
            else:
                cur_steer += step
        else:
            cur_steer = target_steer
            
        cur_throttle = vehicle_controls["throttle"]
        if space:
            cur_throttle = 0.0
        else:
            target_throttle = 0.0
            if w:
                target_throttle = max_speed
            elif s:
                target_throttle = -max_speed
                
            if target_throttle > cur_throttle:
                cur_throttle = min(cur_throttle + speed_step * dt, target_throttle)
            elif target_throttle < cur_throttle:
                cur_throttle = max(cur_throttle - speed_step * 1.5 * dt, target_throttle)

        with state_lock:
            vehicle_controls["throttle"] = cur_throttle
            vehicle_controls["steering"] = cur_steer
            
        connected, l_dist, c_dist, r_dist, real_speed = esp32_comm.get_telemetry()
        
        filtered_distances, v_rel = sensor_proc.process(l_dist, c_dist, r_dist)
        relative_velocity = v_rel
        
        final_throttle, status_text = acc_system.compute_control(
            acc_enabled, cur_throttle, filtered_distances, v_rel
        )
        acc_status_text = status_text
        
        esp32_comm.send_control(final_throttle, cur_steer)
        
        sim_readings = twin_sim.update(final_throttle, cur_steer, connected, real_speed, dt)
        
        if not connected:
            esp32_comm.update_simulated_telemetry(
                sim_readings[0], sim_readings[1], sim_readings[2], 
                float(abs(twin_sim.speed) / 10.0)
            )
            
        surf = twin_sim.render()
        
        raw_rgb = pygame.image.tostring(surf, 'RGB')
        numpy_img = np.frombuffer(raw_rgb, dtype=np.uint8).reshape((twin_sim.height, twin_sim.width, 3))
        bgr_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        
        ok_enc, jpeg_bytes = cv2.imencode('.jpg', bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok_enc:
            latest_twin_jpeg = jpeg_bytes.tobytes()
            
        time.sleep(0.03)

# Start background services
dehaze_mod.start()
esp32_comm.start()

twin_thread = threading.Thread(target=twin_processing_loop, daemon=True)
twin_thread.start()

# ==============================================================================
# COMBINED FLASK ENDPOINTS (RESTORING ALL ORIGINAL ROUTES)
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')

# --- Original Video Mode Routes ---

@app.route('/process', methods=['POST'])
def process():
    with progress_lock:
        if progress_state["status"] not in ["idle", "completed", "error"]:
            return jsonify({"success": False, "error": "A video is already being processed."}), 400
            
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided."}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename."}), 400
        
    model_name = request.form.get('model', 'dehazeformer-t')
    resolution = int(request.form.get('resolution', '1280'))
    fp16 = request.form.get('fp16', 'false').lower() == 'true'
    preprocess_mode = request.form.get('preprocess', 'video')
    postprocess_mode = request.form.get('postprocess', 'video')
    comparison = request.form.get('comparison', 'false').lower() == 'true'
    
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    # Reset and update progress status synchronously before starting background thread to avoid race conditions in frontend polling
    update_progress(status="extracting", current_frame=0, total_frames=0, error=None, output_video=None, comparison_video=None)
    
    thread = threading.Thread(
        target=run_dehaze_thread, 
        args=(video_path, model_name, resolution, fp16, preprocess_mode, postprocess_mode, comparison)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True})

@app.route('/progress')
def get_progress():
    with progress_lock:
        return jsonify(progress_state)

@app.route('/process/preview')
def process_preview():
    global latest_process_frame
    with latest_process_frame_lock:
        if latest_process_frame is None:
            return "No active processing preview available", 404
        ok, jpeg = cv2.imencode('.jpg', latest_process_frame)
        if not ok:
            return "Failed to encode preview image", 500
        return send_file(BytesIO(jpeg.tobytes()), mimetype='image/jpeg')

# --- Original IP Camera Feed Routes ---

@app.route('/live/start', methods=['POST'])
def live_start():
    url = request.form.get('url')
    if not url:
        return jsonify({"success": False, "error": "IP Camera URL is required."}), 400
        
    model_name = request.form.get('model', 'dehazeformer-t')
    resolution = int(request.form.get('resolution', '640'))
    fp16 = request.form.get('fp16', 'false').lower() == 'true'
    preprocess_mode = request.form.get('preprocess', 'video')
    postprocess_mode = request.form.get('postprocess', 'video')
    
    try:
        live_manager.start(url, model_name, resolution, fp16, preprocess_mode, postprocess_mode)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/live/stop', methods=['POST'])
def live_stop():
    live_manager.stop()
    return jsonify({"success": True})

def live_stream_generator():
    while live_manager.active:
        if live_manager.frame_ready_event.wait(timeout=0.1):
            frame = live_manager.latest_mjpeg
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/live/stream')
def live_stream():
    return Response(live_stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live/stats')
def live_stats():
    with live_manager.stats_lock:
        connected = live_manager.grabber.connected if live_manager.grabber else False
        return jsonify({
            "active": live_manager.active,
            "connected": connected,
            "fps": round(live_manager.fps, 1),
            "latency": round(live_manager.latency, 1),
            "is_recording": live_manager.recording_process is not None
        })

@app.route('/live/snapshot')
def live_snapshot():
    frame = live_manager.latest_dehazed
    if frame is None:
        return "No active live frame available for snapshot.", 404
        
    ok, png_bytes = cv2.imencode('.png', frame)
    if not ok:
        return "Failed to encode snapshot image.", 500
        
    return send_file(
        BytesIO(png_bytes.tobytes()), 
        mimetype='image/png', 
        as_attachment=True, 
        download_name='dehazed_snapshot.png'
    )

@app.route('/live/record/start', methods=['POST'])
def live_record_start():
    success, msg = live_manager.start_recording()
    return jsonify({"success": success, "message": msg})

@app.route('/live/record/stop', methods=['POST'])
def live_record_stop():
    url = live_manager.stop_recording()
    if url:
        return jsonify({"success": True, "video_url": url})
    else:
        return jsonify({"success": False, "error": "No active live camera recording found."}), 400

# --- Web-Merged Digital Twin Streams ---

def get_mjpeg_stream(feed_type):
    while True:
        frame_bytes = None
        
        if feed_type == 'original':
            orig, _, _, _ = dehaze_mod.get_frames()
            if orig is not None:
                bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
                ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    frame_bytes = jpeg.tobytes()
        elif feed_type == 'dehazed':
            _, dehazed, _, _ = dehaze_mod.get_frames()
            if dehazed is not None:
                bgr = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
                ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    frame_bytes = jpeg.tobytes()
        elif feed_type == 'twin':
            frame_bytes = latest_twin_jpeg
            
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)

@app.route('/stream/original')
def stream_original():
    return Response(get_mjpeg_stream('original'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/dehazed')
def stream_dehazed():
    return Response(get_mjpeg_stream('dehazed'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/twin')
def stream_twin():
    return Response(get_mjpeg_stream('twin'), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Web-Merged Telemetry / Control APIs ---

@app.route('/api/control', methods=['POST'])
def api_control():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data received"}), 400
        
    with state_lock:
        keyboard_state["w"] = data.get("w", False)
        keyboard_state["s"] = data.get("s", False)
        keyboard_state["a"] = data.get("a", False)
        keyboard_state["d"] = data.get("d", False)
        keyboard_state["space"] = data.get("space", False)
        
    return jsonify({"success": True})

@app.route('/api/config', methods=['POST'])
def api_config():
    global acc_enabled
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data received"}), 400
        
    if "model" in data:
        dehaze_mod.set_model(data["model"])
    if "camera" in data:
        dehaze_mod.set_camera(int(data["camera"]))
    if "acc_enabled" in data:
        acc_enabled = bool(data["acc_enabled"])
    if "resolution" in data:
        dehaze_mod.set_resolution(int(data["resolution"]))
    if "fp16" in data:
        dehaze_mod.set_fp16(bool(data["fp16"]))
    if "preprocess" in data:
        dehaze_mod.set_preprocess(data["preprocess"])
    if "postprocess" in data:
        dehaze_mod.set_postprocess(data["postprocess"])
    if "adaptive_mode" in data:
        dehaze_mod.set_adaptive_mode(bool(data["adaptive_mode"]))
    if "manual_override" in data:
        dehaze_mod.set_manual_override(bool(data["manual_override"]))
    if "threshold" in data:
        dehaze_mod.set_threshold(float(data["threshold"]))
        
    return jsonify({"success": True})

@app.route('/api/mouse', methods=['POST'])
def api_mouse():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data received"}), 400
        
    mx = float(data.get("x", 0.0))
    my = float(data.get("y", 0.0))
    is_pressed = bool(data.get("is_pressed", False))
    
    twin_sim.handle_mouse((mx, my), is_pressed)
    return jsonify({"success": True})

@app.route('/api/telemetry', methods=['GET'])
def api_telemetry():
    connected, l_dist, c_dist, r_dist, real_speed = esp32_comm.get_telemetry()
    
    # Correct unpacking of DehazeModule stats to prevent TypeError
    orig_frame, dehaze_frame, fps, latency_ms = dehaze_mod.get_frames()
    fog_stats = dehaze_mod.get_fog_stats()
    
    ttc = -1.0
    if relative_velocity < -0.05:
        ttc = (c_dist / 100.0) / abs(relative_velocity)
        
    with state_lock:
        manual_throttle = vehicle_controls["throttle"]
        steering_angle = vehicle_controls["steering"]

    return jsonify({
        "connected": connected,
        "acc_enabled": acc_enabled,
        "acc_status": acc_status_text,
        "speed": round(real_speed, 2),
        "throttle": int(manual_throttle),
        "steering": int(steering_angle),
        "dist_l": round(l_dist, 1),
        "dist_c": round(c_dist, 1),
        "dist_r": round(r_dist, 1),
        "v_rel": round(relative_velocity, 2),
        "ttc": round(ttc, 2) if ttc >= 0 else None,
        "dehaze_fps": round(fps, 1),
        "dehaze_latency": round(latency_ms, 1),
        "fallback_active": dehaze_mod.is_fallback,
        "adaptive_mode": fog_stats["adaptive_mode"],
        "manual_override": fog_stats["manual_override"],
        "threshold": fog_stats["threshold"],
        "current_density": round(fog_stats["current_density"], 4),
        "dehazing_active": fog_stats["dehazing_active"]
    })

# --- Static results routes ---

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

