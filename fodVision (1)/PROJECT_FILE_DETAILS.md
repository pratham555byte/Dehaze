# 📁 FogVision ADAS: Project File Catalog & Architecture Reference

This document provides a comprehensive catalog of every file and directory in the **FogVision ADAS** project. It details the purpose, type, and specific role of each module within the overall Driver Assistance System architecture.

---

## 🗺️ Project Directory Tree

Below is the conceptual layout of the repository:

```yaml
fogVision/
├── DashBoard.py                      # Streamlit HUD and User Interface
├── pipeline.py                       # Core ADAS Pipeline Coordinator
├── video_engine.py                   # Playback & Capture Frame Ingestion
├── Adonet/                           # Deep learning dehazing models (FFA-Net/AOD-Net)
├── PyFADE/                           # Fog Aware Density Evaluator Python port
├── assets/                           # Image assets and output snapshots
├── config/                           # Blackspot database and ADAS scenario datasets
├── dehaze config/                    # Flask Dehaze service & speed profiling scripts
├── docs/                             # Developer manuals and architecture workflows
├── fog_impact_testing/               # Synthetic fog rendering and impact test suites
├── models/                           # Pre-trained YOLOv8n object detection weights
└── venv/                             # Python virtual environment (ignored)
```

---

## ⚙️ Core Application & ADAS Pipeline Modules

These files form the primary operational loop of the ADAS application, handling dashboard rendering, frame-by-frame scheduling, and input feeds.

| File Name | Size (Bytes) | Role & Key Responsibilities |
| :--- | :--- | :--- |
| [DashBoard.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/DashBoard.py) | ~65,922 | **Streamlit HUD Interface:** Coordinates the main frontend display. Configures dashboard layouts in Royal Midnight Blue, manages video uploads, controls speed limits/ego speed thresholds, configures Ollama LLM parameters, displays geocoded blackspot warning maps (via Folium), and renders real-time dashboard dials (gauges, speedometers) alongside visual overlays. |
| [pipeline.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/pipeline.py) | ~47,142 | **Orchestration Core:** Implements the `ADASPipeline` class. Runs the step-by-step frame processing loop: frame rescaling ($640 \times 480$), asynchronous fog calculation triggers, temporal trend updates, lane detection masking, object detection/color tracking, depth estimation, motion analysis (TTC), risk indexing, and LLM driving context generation. |
| [video_engine.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/video_engine.py) | ~10,878 | **Frame Acquisition Ingestion:** Manages pre-recorded video feeds and webcam streams. In pre-recorded video mode, it executes in a dual-phase manner: first dehazing high-density segments to cache a temporary video, and second reading frames at the target speed (default 10 FPS). In webcam mode, it pulls frames at a locked 1 FPS. |

---

## 👁️ Visual Processing & Perception Modules

These files extract visibility conditions (fog density and trend tracking), apply image restoration (dehazing), and classify visual targets.

| File Name | Size (Bytes) | Role & Key Responsibilities |
| :--- | :--- | :--- |
| [fog_density.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_density.py) | ~1,765 | **DCP Fog Estimator:** Implements Dark Channel Prior (DCP) morphology equations. Evaluates local $15\times15$ pixel structures to estimate the raw density percentage of atmospheric fog on input frames. |
| [dehaze.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze.py) | ~5,001 | **Atmospheric Haze Restorer:** Restores hazy frames by computing dark channel transmission mapping and solving atmospheric scattering equations. Improves visibility and target detection bounds in low-visibility environments. |
| [fog_aware.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_aware.py) | ~9,828 | **Asynchronous Preprocessor:** Wraps fog estimation in a threaded worker `FogAwarePreprocessor` to execute fog assessments in the background (at 1 Hz) to prevent main pipeline execution pauses. |
| [temporal_fog_predictor.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/temporal_fog_predictor.py) | ~3,797 | **Trend Forecaster:** Tracks historical fog density values using an Exponential Moving Average (EMA) and computes linear regression slopes over rolling windows to output predicted fog density levels and trends (`increasing`, `decreasing`, `stable`). |
| [object_detect.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/object_detect.py) | ~5,585 | **YOLO Bounding Box & Signal Classifier:** Feeds preprocessed frames to a YOLOv8n network to locate vehicles, pedestrians, and signals. Contains crop features that isolate traffic signals and maps them to HSV space to classify lights (`RED`, `YELLOW`, `GREEN`, `UNKNOWN`). |
| [lane_detection.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/lane_detection.py) | ~10,000+ | **Lane Boundaries Segmenter:** Generates trapezoidal overlays modeling the forward lane, segmenting detected targets into *In-Lane Targets* (high hazard) and *Out-of-Lane Targets* (low/peripheral hazard). |

---

## 📐 Distance Calibration & Sensor Fusion Modules

These files resolve physical target distances (using optical geometry, deep learning depth models, or physical sensor devices) and merge inputs.

| File Name | Size (Bytes) | Role & Key Responsibilities |
| :--- | :--- | :--- |
| [improved_distance.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/improved_distance.py) | ~6,072 | **Pinhole Geometry Estimator:** Implements `EnhancedDistanceEstimator`. Dynamically computes physical target distances using focal ratios and class-height averages (e.g. trucks at 3.8m, cars at 1.5m, pedestrians at 1.75m) in webcam/live feed modes. |
| [sensor_fusion.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/sensor_fusion.py) | ~3,705 | **Camera-LiDAR Fusion Engine:** Fuses distances from monocular cameras with hardware/simulated LiDAR sensors. Utilizes a confidence-based decision rule to prioritize LiDAR when signal strength is $\ge 70\%$ and fall back to camera estimates in high-attenuation dense fog. |
| [lidar_sensor.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/lidar_sensor.py) | ~6,376 | **LiDAR Hardware Interface:** Communicates with physical LiDAR devices to collect distance feeds. Employs simulated data or mathematical fallbacks when hardware interfaces are disconnected. |
| [esp32_module.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/esp32_module.py) | ~2,758 | **ESP32 Robot Vehicle Connector:** Connects to an ESP32 robot vehicle controller via HTTP APIs. Retrieves distance measurements from three onboard VL53L0X Time-of-Flight sensors (left, middle, right) and sends motor/steering control outputs (`/motor`). |

---

## 🚨 Threat Assessment & Alert Systems

These modules process relative motion statistics, evaluate collision risk, and filter warnings to prevent false-alarm fatigue.

| File Name | Size (Bytes) | Role & Key Responsibilities |
| :--- | :--- | :--- |
| [motion_ttc.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/motion_ttc.py) | ~9,065 | **TTC Analyzer:** Track target centroid bounding boxes across sequential frames. Calculates approach velocity and computes Time-to-Collision (TTC) bounds. Flags threats when TTC falls below critical parameters (e.g. 2.0s for Critical, 5.0s for Warning). |
| [risk_score.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/risk_score.py) | ~7,041 | **Hazard Score Evaluator:** Fuses pipeline parameters (fog density, nearest object proximity, vehicle speed, road hazards, humidity) into a normalized score ($0.0$ to $1.0$). Executes hard overrides to guarantee HIGH risk if targets are within 10m. |
| [risk_alerts.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/risk_alerts.py) | ~7,499 | **Dynamic Warnings Manager:** Evaluates safety zones and collision thresholds to dispatch alert classifications (`CRITICAL`, `WARNING`, `SAFE`). |
| [confidence_gated_alerts.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/confidence_gated_alerts.py) | ~4,580 | **Consistency Gatekeeper:** Gates target classifications. Requires detected objects to remain within confidence thresholds for $\ge 3$ consecutive frames before spawning alerts, filtering out brief transient frame errors. |
| [alert_hysteresis.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/alert_hysteresis.py) | ~5,919 | **Alert Cooldown Locks:** Implements hysteresis states that lock activated warning states for a cooldown period (Warning: 5s, Critical: 10s) to prevent dashboard alert flickering. |

---

## 🧠 Cognitive Advisories & Voice Alert Systems

These files utilize AI reasoning models and speech engines to deliver contextual guidance.

| File Name | Size (Bytes) | Role & Key Responsibilities |
| :--- | :--- | :--- |
| [llm.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/llm.py) | ~18,027 | **Ollama LLM Orchestrator:** Encapsulates the cognitive reasoning model calls. Receives pipeline states via a serialized `DrivingContext` object and queries local Ollama models (`llama3.1:latest` or `qwen3:1.7b`) to fetch structured driving advice. |
| [voice_alert.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/voice_alert.py) | ~723 | **Threaded Speech Synthesizer:** Spawns asynchronous worker threads using `pyttsx3` to execute safety voice alerts in the background, preventing speech pauses from bottlenecking the main frame loop. |
| [road_context.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/road_context.py) | ~3,710 | **Nominatim GIS Resolver:** Queries geocoding servers using current coordinate inputs to identify road names, and checks local spatial files to find nearby accident-prone zones. |
| [system_prompt.txt](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/system_prompt.txt) | ~8,199 | **Co-Driver Prompt System:** System configuration prompt guiding Ollama model outputs. Frames the model as a quick, decisive co-pilot that outputs JSON payloads prioritizing driver safety. |

---

## 📂 Subdirectories & Extension Repositories

### 1. `config/`
Stores geospatial resources and simulation templates.
* [blackspots.json](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/config/blackspots.json) *(~95,725 bytes)*: GeoJSON coordinate data mapping high-hazard accident blackspots in India (especially fog corridors).
* [adas_dataset_extended.json](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/config/adas_dataset_extended.json) *(~35,184 bytes)*: Contains structured test scripts defining diverse driving events for model evaluation.
* [.env.example](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/config/.env.example) *(~810 bytes)*: Reference template detailing environment variables for API integrations.

### 2. `PyFADE/`
A Python translation of the MATLAB *Fog Aware Density Evaluator* (FADE) logic.
* [setup.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/PyFADE/setup.py) & [pyproject.toml](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/PyFADE/pyproject.toml): Packaging and installer configurations.
* [src/pyfade/core.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/PyFADE/src/pyfade/core.py) *(~26,739 bytes)*: Evaluates statistical distributions of natural images without reference models to calculate fog values.
* [src/pyfade/_compat.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/PyFADE/src/pyfade/_compat.py) *(~13,765 bytes)*: Provides mathematical translations (e.g. convolutions, Gaussian filters) to replicate MATLAB functions in NumPy.
* [src/pyfade/cli.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/PyFADE/src/pyfade/cli.py): CLI interface to run FADE checks on image paths.
* `src/pyfade/models/`: MAT files storing statistical feature arrays of foggy and clear landscapes.

### 3. `Adonet/`
Implements deep learning dehazing networks, including AOD-Net and FFA-Net.
* [AOD_Net.caffemodel](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/Adonet/AOD_Net.caffemodel): Trained weight parameters for the AOD-Net model.
* [test_template.prototxt](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/Adonet/test_template.prototxt): Model graph representation file for importing AOD-Net into Caffe.
* [test.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/Adonet/test.py): Dehazes images using the Caffe runtime engine.
* [ffa_test.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/Adonet/ffa_test.py) *(~11,614 bytes)*: Main validation module loading FFA-Net weights.
* `its_train_ffa_3_19.pk` & `ots_train_ffa_3_19.pk`: Trained FFA-Net model weights for indoor/outdoor parameters.

### 4. `dehaze config/`
Contains Flask server scripts and profiling routines to evaluate model performance.
* [app.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze%20config/app.py) *(~26,887 bytes)*: Flask API backend exposing endpoints to upload images/videos, process dehazing tasks, and stream progress metrics.
* [dehaze_video.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze%20config/dehaze_video.py): Restores video files offline using the dehazing networks.
* [infer.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze%20config/infer.py): Runs network evaluations on standalone images.
* [profile_models.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze%20config/profile_models.py): Measures performance benchmarks (FPS and memory metrics) between FFA-Net and AOD-Net.
* [profile_optimizations.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze%20config/profile_optimizations.py): Tests the performance gains of half-precision (float16) and network quantization configurations.

### 5. `docs/`
Developer reference documentation.
* [project_context.txt](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/docs/project_context.txt) *(~65,463 bytes)*: Comprehensive context mapping variables, structures, configurations, and interfaces.
* [SYSTEM_WORKFLOW.md](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/docs/SYSTEM_WORKFLOW.md): Step-by-step logic workflows for video modes and camera integrations.
* [QUICKSTART.md](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/docs/QUICKSTART.md): Initial build instruction sets and testing checklists.
* [CODE_REVIEW.md](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/docs/CODE_REVIEW.md): Code review comments and optimizations checklist.

### 6. `fog_impact_testing/`
Contains test harnesses for evaluating synthetic fog degradation and model responses.
* [test_fog_impact.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_impact_testing/test_fog_impact.py) *(~30,889 bytes)*: Evaluates model robustness by running YOLO detectors over synthetic fog patterns and recording precision drops.
* [test_synthetic_fog.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_impact_testing/test_synthetic_fog.py): Simulates synthetic atmospheric scattering overlays onto test frames.
* `results/`, `results_car/`, `results_mp/`: Output data plots and CSV metrics tracking target retention vs fog density.

---

## 📈 Performance & Test Scripts

These files evaluate systems against benchmark metrics or test specialized modules.

* [evaluation_metrics.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/evaluation_metrics.py) *(~9,885 bytes)*: Analyzes predictions against ground truths. Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for fog density values, and logs alert metrics (Precision, Recall, and F1-Scores).
* [simulation.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/simulation.py) *(~16,853 bytes)*: Runs simulated driving runs. Simulates speed adjustments, changes coordinates, mimics LiDAR inputs, and tests how the ADAS pipeline triggers safety warnings in various situations.
* [test.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/test.py) & [test_esp32.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/test_esp32.py): Validation scripts verifying hardware setups by printing polling streams from active ESP32 sensor endpoints.

---

## 📦 General Configuration, Datasets, and Documentation

* [adas_dataset.json](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/adas_dataset.json) *(~4,082 bytes)*: Scenario mapping database referencing fog densities, distances, expected risk levels, recommended speed curves, and voice scripts.
* [Requirements.txt](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/Requirements.txt) *(~471 bytes)*: Package installation specifications listing core packages like Streamlit, PyTorch, Ultralytics, and PyTTsX3.
* [SYSTEM_DOCUMENTATION.md](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/SYSTEM_DOCUMENTATION.md) & [SYSTEM_DOCUMENTATION.txt](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/SYSTEM_DOCUMENTATION.txt): Main architecture specifications detailing steps of visual inference, mathematical logic equations, and setup checks.
* [SYSTEM_WORKFLOW.md](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/SYSTEM_WORKFLOW.md) *(~12,652 bytes)*: Contains structural flow specifications and sequencing patterns.
* [SYSTEM_ENHANCEMENT_REPORT.md](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/SYSTEM_ENHANCEMENT_REPORT.md) *(~3,996 bytes)*: Details camera-LiDAR fusion justification, latency benchmarks (goal under 100ms per frame), and sensor performance criteria.
* [summarize.md](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/summarize.md) *(~11,951 bytes)*: Highlights high-level components and workflow summaries.
* [README.md](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/README.md) *(~15,546 bytes)*: Project index guide, setup details, and run configurations.
* [sim_state.json](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/sim_state.json): Stores temporary parameter presets for the ADAS simulator.
* [.gitignore](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/.gitignore) & [.env](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/.env): Git environment configuration variables and repository exclusions.
