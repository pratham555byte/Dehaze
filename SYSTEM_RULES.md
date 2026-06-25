# 📜 FogVision ADAS: Complete System Rules & Threshold Specifications

This document defines the operational rules, timing constraints, threshold limits, and sensor fusion decision heuristics implemented within the **FogVision ADAS** pipeline.

---

## 🕰️ Timing & Execution Gating Rules

To maintain real-time execution speeds, processing tasks are split across distinct time-gated loops:

1. **Frame Ingestion Rate:**
   - **Upload Video Mode:** Plays back at **10 FPS** (delivers frames to the pipeline after complete pre-dehazing caches).
   - **Live Webcam Mode:** Restricted to a locked **1 FPS** feed to save CPU/GPU overhead.
2. **Fog Density Estimation (Async):**
   - Runs asynchronously in a background thread once every **1.0 second (1 Hz)**. 
   - A synchronous run is only forced on the very first frame of the Live Feed to display correct initial metrics immediately.
3. **Cognitive Advisory (Ollama LLM):**
   - Runs asynchronously in a background thread.
   - Evaluated every **5.0 seconds** under nominal conditions.
   - Frequency increases to every **1.5 seconds** when active alerts (Warning or Critical) are triggered.
4. **Physical ESP32 Controls:**
   - Asynchronous serial/network speed commands (`POST /motor`) sent to the ESP32 vehicle with a rate limit of **0.08 seconds (12.5 Hz)** to prevent network congestion.
5. **GIS Road Context Lookup:**
   - Queries geocoding servers and blackspot databases every **10.0 seconds** to minimize network calls.

---

## 🚗 Safe Speed & Visibility Thresholds

Vehicular speed limits are dynamically calculated based on raw fog density percentages:

| Fog Density Range | Max Safe Speed (Video) | Max Safe Speed (Live Feed) | Visibility Level |
| :--- | :--- | :--- | :--- |
| **Fog $\ge 80\%$** | 30 km/h | 30 km/h | Very Low |
| **Fog $60\% - 79\%$** | 45 km/h | 50 km/h | Low |
| **Fog $40\% - 59\%$** | 60 km/h | 70 km/h | Moderate |
| **Fog $20\% - 39\%$** | 80 km/h | 100 km/h | Good |
| **Fog $< 20\%$** | 100 km/h | 100 km/h | Excellent |

*If the current vehicle speed exceeds the dynamically calculated `Max Safe Speed`, the system raises an `overspeeding` alert on the HUD safety panel.*

---

## 🌫️ Preprocessing & Dehazing Logic

1. **Activation Gating:**
   - Dehazing is completely bypassed in light fog (Fog Density $\le 35\%$).
   - If Fog Density exceeds **35%**, the Dark Channel Prior (DCP) restoration model is run on the frame before feeding it to the YOLO object detector.
2. **Video Dehaze Cache:**
   - Uploaded video files are scanned. If the first frame has a fog density $> 35\%$, the video is pre-dehazed completely and cached, saving live frame processing overhead.

---

## 👁️ Visual Perception Class Filtering

1. **Object Classification Constraints:**
   - Bounding boxes are only registered if the classification confidence is **$> 0.40$**.
   - Detections are filtered to only include road-relevant categories: `{"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light", "stop sign"}`.
2. **Visual Lane Boundaries (Video Mode):**
   - Targets are projected against a perspective trapezoid boundary overlay ($X_{\text{left}} = 160$, $X_{\text{right}} = 480$, $Y_{\text{top}} = 264$, $Y_{\text{bottom}} = 480$).
   - Objects whose centroids fall outside this trapezoid are classified as peripheral/out-of-lane and bypassed.

---

## 📡 Sensor Fusion & Plan A Section Splitting

### 1. Plan A Camera Section Splitting (Live webcam mode)
The $640 \times 480$ camera frame coordinate space is divided into 3 horizontal sectors:
- **Left Zone:** $X < 213$
- **Middle Zone:** $213 \le X \le 426$
- **Right Zone:** $X > 426$

### 2. Distance Fusion Priority (Camera vs. LiDAR)
When physical hardware sensor values are compared:
- **LiDAR Priority:** If LiDAR confidence is **$\ge 70\%$** and $|d_{\text{LiDAR}} - d_{\text{Camera}}| \le 5$ meters, the system utilizes the LiDAR distance.
- **Camera Fallback:** If LiDAR confidence is **$< 70\%$** (due to severe fog scattering), the system overrides and uses camera estimations.
- **Weighted Average:** If distance estimates diverge by $> 5$ meters, a weighted average is calculated:
  $$d_{\text{fused}} = 0.70 \times d_{\text{LiDAR}} + 0.30 \times d_{\text{Camera}}$$

---

## 🚨 Safety Alerts & Automated Braking Controls

1. **Time-to-Collision (TTC) Equations:**
   - Preceding vehicle velocity is tracked across frames.
   - For simulated video speed ($v_{\text{rel}} = \text{speed}_{\text{kmh}} / 3.6$ in m/s).
   - For live prototype vehicle speed ($v_{\text{rel}} = \text{speed}_{\text{motor}} / 100.0$ in m/s).
   - **TTC Thresholds:**
     - **Critical Warning:** TTC $< 0.8$ seconds (Live Mode) / $< 2.0$ seconds (Video Mode).
     - **Standard Warning:** TTC $< 1.5$ seconds (Live Mode) / $< 5.0$ seconds (Video Mode).
2. **Proximity Alerts Thresholds:**
   - **Critical Warning:** Distance $< 0.3$ meters (Live Mode) / $< 10.0$ meters (Video Mode).
   - **Standard Warning:** Distance $< 0.6$ meters (Live Mode) / $< 20.0$ meters (Video Mode).
   - **Caution Warning:** Distance $< 1.0$ meter (Live Mode) / $< 30.0$ meters (Video Mode).
3. **Automated Emergency Braking (ACC):**
   - In Live feed mode, if the Middle zone ESP32 distance sensor registers a barrier closer than **$0.25$ meters**, the system engages `automated_braking`, forcing the motor speed to **0** and issuing a warning voice announcement.
4. **Zero/Slow Speed Alarm Suppression:**
   - Collision and proximity warnings are completely disabled when current vehicle speed is **$\le 5.0$ km/h** (motor speed scale) to prevent false alarms while parked or reversing at slow speeds.
5. **Alert Hysteresis Cooldowns:**
   - Triggered alarms are held active to prevent display flicker and announcement stuttering:
     - **Warning alerts** are locked for a minimum of **5.0 seconds**.
     - **Critical alerts** are locked for a minimum of **10.0 seconds**.
