import os
import time
import threading
import cv2
import torch
import numpy as np

# Local ADAS modules
from object_detect import process_frame
from road_context import get_road_context
from voice_alert import speak_alert
from llm import get_llm_decision, DrivingContext
from risk_score import compute_risk
from temporal_fog_predictor import TemporalFogPredictor
from confidence_gated_alerts import ConfidenceGatedAlerts
from alert_hysteresis import AlertHysteresis
from sensor_fusion import SensorFusion
from motion_ttc import MotionAnalyzer
from improved_distance import EnhancedDistanceEstimator
from risk_alerts import RiskBasedAlertSystem

class ADASPipelineRunner:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.yolo_model = None
        self.midas_model = None
        self.midas_transforms = None
        self._midas_frame_counter = 0
        self._cached_depth_map = None
        
        # Configurable Parameters
        self.distance_mode = "pinhole"  # "midas" or "pinhole"
        self.cruising_speed = 50.0      # km/h
        
        # Alert systems
        self.temporal_fog = TemporalFogPredictor(window_size=10, alpha=0.3)
        self.confidence_gating = ConfidenceGatedAlerts(confidence_threshold=0.5, frames_required=3)
        self.alert_hysteresis = AlertHysteresis(warn_cooldown=5.0, critical_cooldown=10.0, escalation_threshold=3)
        self.sensor_fusion = SensorFusion()
        self.motion_analyzer = MotionAnalyzer(fps=30.0)
        self.distance_estimator = EnhancedDistanceEstimator(frame_height=480)
        self.risk_alerts = RiskBasedAlertSystem(
            ttc_warning_threshold=5.0,
            ttc_critical_threshold=2.0,
            min_distance_warning=20.0,
            min_distance_critical=10.0,
        )
        
        # State tracking
        self.prev_center_dist = None
        self.prev_dist_time = None
        self.relative_velocity = 0.0
        self.alpha_v = 0.3
        
        # Threading for Ollama LLM
        self.llm_interval_s = 5.0
        self.last_llm_time = 0.0
        self.cached_llm_response = ""
        self.last_llm_spoken = ""
        self._llm_in_progress = False
        self._llm_lock = threading.Lock()
        
        # GPS Context
        self.lat = 12.9716  # Bangalore lat
        self.lon = 77.5946  # Bangalore lon
        self.road_context = {"road_type": "highway", "blackspots": False, "road_name": "Unknown Road"}
        self.last_gps_time = 0.0
        self.gps_interval_s = 10.0
        
        # HUD overlays (fixed box boundary: middle 50% width, bottom 50% height of 640x480)
        self.left_x = 160
        self.right_x = 480
        self.top_y = 240
        self.bottom_y = 432
        
        # Active Threat Levels
        self.current_risk_score = 0.0
        self.current_risk_level = "LOW"
        self.current_threat_label = "SAFE"
        self.override_reason = ""
        self.nearest_obj_label = "none"
        self.nearest_obj_dist = 80.0
        self.traffic_light_status = "UNKNOWN"
        self.ttc = -1.0
        self.fps = 0.0

    def set_distance_mode(self, mode):
        if mode in ["midas", "pinhole"]:
            self.distance_mode = mode
            print(f"[ADASPipelineRunner] Distance mode set to: {mode}")

    def run_pipeline(self, frame, current_speed_kmh, is_live=False, lidar_distances=None):
        """
        Run object detection, distance estimation, sensor fusion, TTC, risk calculation, 
        and cognitive Ollama advice on a single frame.
        
        lidar_distances: [Left, Center, Right] in cm (if connected to ESP32 / sim)
        """
        t_start = time.perf_counter()
        now = time.time()
        
        # 1. Resize frame to 640x480 for standardized CV coordinate space
        h_orig, w_orig = frame.shape[:2]
        if (w_orig, h_orig) != (640, 480):
            frame_resized = cv2.resize(frame, (640, 480))
        else:
            frame_resized = frame.copy()
            
        annotated_frame = frame_resized.copy()
        
        # 2. Asynchronous GPS Context lookup
        if now - self.last_gps_time >= self.gps_interval_s:
            self.last_gps_time = now
            def run_gps_async():
                try:
                    ctx = get_road_context(self.lat, self.lon)
                    if ctx:
                        self.road_context = ctx
                except Exception:
                    pass
            threading.Thread(target=run_gps_async, daemon=True).start()

        # 3. Object Detection (runs YOLOv8n)
        # Note: process_frame returns annotated (with box if not is_video) and detections list
        # We pass is_video=True to manually draw customized ADAS bounding boxes here.
        _, detections = process_frame(frame_resized, is_video=True)
        
        # 4. Traffic Light State Tracking
        self.traffic_light_status = "UNKNOWN"
        for d in detections:
            if d["label"] == "traffic light":
                color = d.get("traffic_light_color", "UNKNOWN")
                if color != "UNKNOWN":
                    self.traffic_light_status = color

        # 5. Distance Estimation
        if self.distance_mode == "midas" and len(detections) > 0:
            # Lazy load MiDaSSmall model
            if self.midas_model is None:
                print("[ADASPipelineRunner] Loading MiDaS Small model...")
                self.midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.midas_model.to(self.device).eval()
                
            if self._cached_depth_map is None or self._midas_frame_counter % 3 == 0:
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                input_batch = self.midas_transforms.small_transform(img_rgb).to(self.device)
                with torch.no_grad():
                    prediction = self.midas_model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=frame_resized.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                self._cached_depth_map = prediction.cpu().numpy()
            
            self._midas_frame_counter += 1
            depth_map = self._cached_depth_map
            
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                roi_depth = depth_map[y1:y2, x1:x2]
                if roi_depth.size > 0:
                    median_depth = np.median(roi_depth)
                    estimated_dist = float(np.clip(3000.0 / (median_depth + 1e-5), 2.0, 80.0))
                    d["distance"] = round(estimated_dist, 1)
                else:
                    d["distance"] = 20.0
        else:
            # Pinhole focal height mapping
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                label = d.get("label", "_default").lower()
                d["distance"] = self.distance_estimator._estimate_from_bbox([x1, y1, x2, y2], label)

        # 6. Sensor Fusion
        # Fuses camera target distances with the hardware/simulated LiDAR sensors
        esp32_sensors_m = {"left": 80.0, "middle": 80.0, "right": 80.0}
        
        # Base mock sensor ranges on visual obstacle positions
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) // 2
            dist = d.get("distance", 80.0)
            if cx < 213:
                esp32_sensors_m["left"] = min(esp32_sensors_m["left"], dist)
            elif cx < 426:
                esp32_sensors_m["middle"] = min(esp32_sensors_m["middle"], dist)
            else:
                esp32_sensors_m["right"] = min(esp32_sensors_m["right"], dist)

        # Override simulated values if physical ESP32 or simulated distances are available
        if lidar_distances is not None:
            # lidar_distances contains [Left, Center, Right] in cm. Convert to meters.
            esp32_sensors_m["left"] = lidar_distances[0] / 100.0
            esp32_sensors_m["middle"] = lidar_distances[1] / 100.0
            esp32_sensors_m["right"] = lidar_distances[2] / 100.0
            
        esp32_sensors_m = {k: round(v, 2) for k, v in esp32_sensors_m.items()}

        # 7. Draw lane fixed box boundaries & sensor segments on annotated frame
        # Draw vertical separator lines
        cv2.line(annotated_frame, (213, 0), (213, 480), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.line(annotated_frame, (426, 0), (426, 480), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"L: {esp32_sensors_m['left']:.2f}m", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"M: {esp32_sensors_m['middle']:.2f}m", (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"R: {esp32_sensors_m['right']:.2f}m", (470, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw the target tracking boxes
        nearest_distance = 80.0
        nearest_label = "none"
        
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) // 2
            
            if cx < 213:
                section = "left"
            elif cx < 426:
                section = "middle"
            else:
                section = "right"
                
            # Fuse camera estimate and LiDAR range
            # If LiDAR is available, use it. If not, use camera.
            # (If connected, lidar_distances will not be None)
            if lidar_distances is not None:
                sensor_dist = esp32_sensors_m[section]
                d["distance"] = sensor_dist
                
            dist = d["distance"]
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_label = d["label"]
                
            # Color box by threat
            # Scale threshold limits down to physical car sizes (is_live)
            crit_zone = 0.25 if is_live else 10.0
            warn_zone = 0.50 if is_live else 20.0
            
            if dist < crit_zone:
                box_color = (0, 0, 255)  # Red (Critical Proximity)
            elif dist < warn_zone:
                box_color = (0, 255, 255) # Yellow (Warning Range)
            else:
                box_color = (0, 255, 0)  # Green (Safe)
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            lbl = f"{d['label']}: {dist:.1f}m"
            if d.get("traffic_light_color"):
                lbl += f" ({d['traffic_light_color']})"
            cv2.putText(annotated_frame, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        self.nearest_obj_label = nearest_label
        self.nearest_obj_dist = nearest_distance

        # 8. Time-To-Collision (TTC) & Approach Speed
        current_center_dist = esp32_sensors_m["middle"]
        if self.prev_center_dist is not None and self.prev_dist_time is not None:
            dt = current_time = time.time()
            dt = current_time - self.prev_dist_time
            if dt > 0.05:
                raw_v = (current_center_dist - self.prev_center_dist) / dt
                raw_v = max(-10.0, min(raw_v, 10.0))  # filter noise spikes
                self.relative_velocity = self.alpha_v * raw_v + (1.0 - self.alpha_v) * self.relative_velocity
                self.prev_center_dist = current_center_dist
                self.prev_dist_time = current_time
        else:
            self.relative_velocity = 0.0
            self.prev_center_dist = current_center_dist
            self.prev_dist_time = time.time()
            
        # Calculate TTC
        self.ttc = -1.0
        if self.relative_velocity < -0.05:  # obstacle approaching
            self.ttc = current_center_dist / abs(self.relative_velocity)

        # 9. Risk & Alert state machine
        # Use risk_score.py engine
        # Calculate fog score. We will map current_density from 0.0-1.0 to 0-100
        # If we have a fog density value from DehazeModule, use it. Let's assume we map 0.0-1.0 from ResNet-18 estimator
        resnet_density = getattr(self, "_latest_fog_density_resnet", 0.0) # range 0.0 - 1.0
        fog_score_100 = resnet_density * 100.0
        
        risk_result = compute_risk(
            fog_density=fog_score_100,
            distance_to_nearest=nearest_distance,
            actual_speed=current_speed_kmh,
            road_context=self.road_context,
            is_live=is_live
        )
        
        self.current_risk_score = risk_result["risk_score"]
        self.current_risk_level = risk_result["risk_level"]
        self.override_reason = risk_result["override_reason"]
        
        # Confidence Gating & Alert Hysteresis
        # (Pass risk parameters to stabilize alerts)
        stabilized_alert = self.alert_hysteresis.update(self.current_risk_level)
        self.current_threat_label = stabilized_alert

        # Automated emergency speech check
        crit_limit = 0.25 if is_live else 8.0
        if nearest_distance < crit_limit:
            speak_alert("automated_braking", "Collision warning! Stop immediately.")

        # 10. Cognitive local Ollama LLM Reasoning
        # Query Ollama on a time-gated interval (5s normal, 1.5s under CRITICAL warnings)
        interval = 1.5 if self.current_threat_label == "HIGH" or self.current_risk_level == "HIGH" else 5.0
        if now - self.last_llm_time >= interval and not self._llm_in_progress:
            self.last_llm_time = now
            self._llm_in_progress = True
            
            # Format trend direction
            trend = "stable"
            hist = self.temporal_fog.update(fog_score_100)
            if hist:
                trend = hist.get("drift_direction", "stable")
                
            extra_data = {
                "left_sensor_m": round(float(esp32_sensors_m["left"]), 2),
                "center_sensor_m": round(float(esp32_sensors_m["middle"]), 2),
                "right_sensor_m": round(float(esp32_sensors_m["right"]), 2),
                "time_to_collision_sec": round(float(self.ttc), 2),
                "relative_velocity_mps": round(float(self.relative_velocity), 2),
                "road_name": self.road_context.get("road", "Unknown Road"),
                "blackspots": self.road_context.get("blackspots", []),
                "traffic_light": self.traffic_light_status
            }
            
            ctx = DrivingContext(
                fog_density=fog_score_100,
                fog_trend=trend,
                nearest_object_m=nearest_distance,
                nearest_object_label=nearest_label,
                road_type=self.road_context.get("road_type", "highway"),
                current_speed_kmh=current_speed_kmh,
                extra=extra_data
            )
            
            def query_ollama_async(prompt_ctx):
                try:
                    response_json = get_llm_decision(prompt_ctx)
                    if response_json:
                        with self._llm_lock:
                            self.cached_llm_response = response_json
                            # Synthesize recommendation to vocal alert if new
                            voice = response_json.get("voice_alert", "")
                            if voice and voice != self.last_llm_spoken:
                                self.last_llm_spoken = voice
                                speak_alert("llm_recommendation", voice)
                except Exception as llm_err:
                    print(f"[Ollama ADAS Advisor] Query error: {llm_err}")
                finally:
                    self._llm_in_progress = False
                    
            threading.Thread(target=query_ollama_async, args=(ctx,), daemon=True).start()

        # Update FPS
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        self.fps = 1.0 / max(elapsed, 0.001)

        if (w_orig, h_orig) != (640, 480):
            return cv2.resize(annotated_frame, (w_orig, h_orig))
        return annotated_frame

    def update_latest_fog(self, density_0_1):
        """Update the runner with the latest ResNet-18 fog density score."""
        self._latest_fog_density_resnet = density_0_1
