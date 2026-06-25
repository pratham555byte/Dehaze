"""
llm_advisor.py
Context-Aware Fog-Adaptive ADAS — LLM Reasoning Engine
=======================================================
Improvements over baseline:
  1. Strict JSON output with schema validation
  2. Priority-based hazard reasoning (collision > visibility > speed)
  3. Temporal fog memory — tracks worsening/improving visibility trend
  4. Filtered YOLO context — only nearest + braking-glow objects sent
  5. Structured DrivingContext dataclass for clean input contract
  6. Ollama inference with retry + schema fallback
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# pyrefly: ignore [missing-import]
from ollama import chat

# ---------------------------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------------------------
MODEL_NAME = "fodvision-adas:v2"
MAX_RETRIES = 2
FOG_HISTORY_WINDOW = 5             # rolling window for trend analysis

# ---------------------------------------------------------------------------
# SYSTEM PROMPT  (strict JSON, priority-ordered reasoning)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a real-time AI-powered ADAS (Advanced Driver Assistance System) safety advisor.
You will receive real-time driving context, including geocoded road name, active blackspots list, traffic signal state, vehicle speed, three proximity sensors (left_sensor_m, center_sensor_m, right_sensor_m) as well as Time-to-Collision (TTC) and relative approach speed.

## Priority Safety Rules:
1. Collision Avoidance (Highest Priority):
   - If time_to_collision_sec is positive and < 4.0, or center_sensor_m < 5.0, set risk_level to CRITICAL.
   - Recommended speed must be 0 km/h or a heavy reduction. Suggest "Brake now! TTC < X s".
2. Traffic Signal Compliance:
   - If traffic_light is "RED", recommended speed is "0 km/h". Highlight "RED signal ahead" in hazard_alert.
   - If traffic_light is "YELLOW", advise slowing down and preparing to stop.
3. Blackspot Corridor Awareness:
   - If nearby blackspots list is active, mention the name of the blackspot (e.g. "NH44 Fog Corridor") in hazard_alert or short_explanation. Limit speed to the blackspot speed limit or 40 km/h.
4. Steering and Maneuvering:
   - Use left_sensor_m, center_sensor_m, and right_sensor_m to guide steering:
     - If center is blocked (e.g. center_sensor_m < 8.0) but left_sensor_m is clear (> 15.0), recommend "Steer left to avoid obstacle".
     - If center is blocked but right_sensor_m is clear, recommend "Steer right to avoid obstacle".
     - If all three are blocked, recommend "Stop immediately - path fully blocked".
5. Fog adaptation:
   - fog_density >= 90 or worsening trend: reduce speed to < 20 km/h.

## Output Contract:
Return ONLY a single valid JSON object containing the exact keys listed below. Do NOT wrap in markdown code blocks. Do not add conversational text. Every value must reflect the specific names/numbers from the inputs.

{
  "risk_level":        "LOW | MODERATE | HIGH | CRITICAL",
  "hazard_alert":      "<primary hazard indicating specific road name or sensor obstacle, max 10 words>",
  "recommended_speed": "<N km/h>",
  "driving_suggestion":"<actionable steering/braking instruction using sensor/TTC, max 15 words>",
  "short_explanation": "<concise reason mentioning location or sensor state, max 20 words>",
  "priority_hazard":   "<collision | visibility | speed | none>",
  "voice_alert":       "<max 8 words speech-ready warning, e.g. 'Red light ahead, stop' or 'Collision warning, brake'>",
  "confidence":        <0.0-1.0>
}
"""

# ---------------------------------------------------------------------------
# DRIVING CONTEXT DATACLASS
# ---------------------------------------------------------------------------
@dataclass
class DrivingContext:
    """
    Structured input for the LLM reasoning engine.
    Populate from your sensor/vision pipeline before calling get_llm_decision().
    """
    fog_density: float                    # 0–100 DCP-derived fog score
    fog_trend: str = "stable"            # "worsening" | "stable" | "improving"
    nearest_object_m: float = 999.0      # distance to closest YOLO detection
    nearest_object_label: str = "none"   # e.g. "car", "truck", "pedestrian"
    road_type: str = "highway"           # "highway" | "curve" | "urban" | "wet"
    current_speed_kmh: float = 60.0      # ego vehicle speed
    timestamp: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)  # any additional sensor data

    def to_prompt_dict(self) -> dict:
        """Returns a clean dict for JSON serialisation into the prompt."""
        d = asdict(self)
        d.pop("extra", None)
        d.pop("timestamp", None)
        if self.extra:
            d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# TEMPORAL FOG MEMORY
# ---------------------------------------------------------------------------
class FogMemory:
    """
    Maintains a rolling window of fog density readings and derives trend.
    Inject into DrivingContext.fog_trend before every LLM call.
    """
    def __init__(self, window: int = FOG_HISTORY_WINDOW):
        self._history: deque[float] = deque(maxlen=window)

    def update(self, fog_density: float) -> str:
        """
        Push new reading and return trend string.
        Returns: "worsening" | "improving" | "stable"
        """
        self._history.append(fog_density)
        if len(self._history) < 2:
            return "stable"

        delta = self._history[-1] - self._history[0]
        if delta > 10:
            return "worsening"
        if delta < -10:
            return "improving"
        return "stable"

    @property
    def average(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def peak(self) -> float:
        return max(self._history, default=0.0)


# ---------------------------------------------------------------------------
# YOLO CONTEXT FILTER
# ---------------------------------------------------------------------------
def filter_yolo_detections(detections: list[dict]) -> dict:
    """
    From a raw YOLO detection list, extract only:
      - nearest object (by estimated distance)

    Each detection dict expected shape:
      {
        "label": str,
        "distance_m": float
      }

    Returns a minimal context dict for the LLM.
    """
    if not detections:
        return {"nearest_object_m": 999.0, "nearest_object_label": "none"}

    sorted_dets = sorted(detections, key=lambda d: d.get("distance_m", 999))
    nearest = sorted_dets[0]

    return {
        "nearest_object_m": nearest.get("distance_m", 999.0),
        "nearest_object_label": nearest.get("label", "unknown"),
    }


# ---------------------------------------------------------------------------
# SCHEMA VALIDATOR
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "risk_level", "hazard_alert", "recommended_speed",
    "driving_suggestion", "short_explanation",
    "priority_hazard", "voice_alert", "confidence",
}

def _validate_schema(parsed: dict) -> bool:
    return REQUIRED_KEYS.issubset(parsed.keys())


def _safe_parse_json(raw: str) -> Optional[dict]:
    """
    Attempt to parse JSON from LLM output.
    Strips markdown fences if present.
    """
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract first {...} block
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(clean[start:end])
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# FALLBACK RESPONSE (rule-based, no LLM)
# ---------------------------------------------------------------------------
def _rule_based_fallback(ctx: DrivingContext) -> dict:
    """
    Deterministic safety output when LLM fails.
    Ensures the system never returns nothing, and generates
    detailed context-aware safety advice dynamically.
    """
    # Extract values from context and extra dictionary safely
    left_m = ctx.extra.get("left_sensor_m", ctx.nearest_object_m)
    center_m = ctx.extra.get("center_sensor_m", ctx.nearest_object_m)
    right_m = ctx.extra.get("right_sensor_m", ctx.nearest_object_m)
    
    ttc = ctx.extra.get("time_to_collision_sec", -1.0)
    v_rel = ctx.extra.get("relative_velocity_mps", 0.0)
    road_name = ctx.extra.get("road_name", "Unknown Road")
    blackspots = ctx.extra.get("blackspots", [])
    traffic_light = ctx.extra.get("traffic_light", "UNKNOWN")
    
    # Base states
    risk = "LOW"
    rec_speed = max(20.0, ctx.current_speed_kmh)
    hazard = "Clear path ahead"
    suggestion = "Maintain safe cruising distance"
    priority = "none"
    voice = "Drive safely"
    explanation = f"Driving on {road_name} under normal conditions."
    
    # 1. Traffic Light Override
    if traffic_light == "RED":
        risk = "HIGH"
        rec_speed = 0.0
        hazard = "RED traffic signal detected"
        suggestion = "Bring vehicle to a complete stop"
        priority = "speed"
        voice = "Red light ahead, stop"
        explanation = "Traffic signal is RED, vehicle must hold position."
    elif traffic_light == "YELLOW":
        risk = "MODERATE"
        rec_speed = min(30.0, ctx.current_speed_kmh * 0.5)
        hazard = "YELLOW traffic signal detected"
        suggestion = "Prepare to stop at intersection"
        priority = "speed"
        voice = "Yellow light, slow down"
        explanation = "Traffic signal is YELLOW, exercise caution."

    # 2. Collision / Proximity warnings (LiDAR/Camera sensors)
    is_critical_proximity = (center_m < 3.0 or left_m < 2.0 or right_m < 2.0)
    is_warning_proximity = (center_m < 8.0 or left_m < 5.0 or right_m < 5.0)
    
    if (ttc > 0.0 and ttc < 4.0) or is_critical_proximity:
        risk = "CRITICAL"
        rec_speed = 0.0
        priority = "collision"
        if ttc > 0.0 and ttc < 4.0:
            hazard = f"Collision risk: TTC {ttc:.1f}s"
            suggestion = f"Brake now! TTC is {ttc:.1f}s"
            voice = "Collision warning, brake now"
            explanation = f"Critical collision danger. Closing speed {abs(v_rel):.1f} m/s, impact in {ttc:.1f}s."
        else:
            hazard = f"Path blocked center at {center_m:.1f}m"
            suggestion = "Stop immediately - path fully blocked"
            voice = "Path blocked, stop immediately"
            explanation = f"Critical proximity threshold breached: C: {center_m:.1f}m, L: {left_m:.1f}m, R: {right_m:.1f}m."
            
    elif is_warning_proximity:
        risk = "HIGH"
        rec_speed = min(30.0, ctx.current_speed_kmh)
        priority = "collision"
        
        # Steering advice based on sensor clearances
        if center_m < 8.0:
            if left_m > right_m and left_m > 8.0:
                hazard = f"Obstacle center at {center_m:.1f}m"
                suggestion = "Steer left to avoid obstacle"
                voice = "Obstacle ahead, steer left"
                explanation = f"Center path blocked at {center_m:.1f}m. Left lane clear ({left_m:.1f}m)."
            elif right_m > left_m and right_m > 8.0:
                hazard = f"Obstacle center at {center_m:.1f}m"
                suggestion = "Steer right to avoid obstacle"
                voice = "Obstacle ahead, steer right"
                explanation = f"Center path blocked at {center_m:.1f}m. Right lane clear ({right_m:.1f}m)."
            else:
                hazard = f"Path obstructed center at {center_m:.1f}m"
                suggestion = "Decelerate and prepare to stop"
                voice = "Path obstructed, slow down"
                explanation = f"All lanes restricted. C: {center_m:.1f}m, L: {left_m:.1f}m, R: {right_m:.1f}m."
        else:
            hazard = "Obstacle in adjacent lane"
            suggestion = "Keep center lane alignment"
            voice = "Obstacle nearby, stay in lane"
            explanation = f"Side restriction detected. Left: {left_m:.1f}m, Right: {right_m:.1f}m."

    # 3. Fog Adaptation (if not already overridden by collision)
    if risk not in ["HIGH", "CRITICAL"]:
        if ctx.fog_density >= 80:
            risk = "HIGH"
            rec_speed = min(25.0, rec_speed)
            hazard = "Dense fog — severely limited visibility"
            suggestion = "Reduce speed and use fog lights"
            priority = "visibility"
            voice = "Dense fog, reduce speed"
            explanation = f"Visibility is extremely low ({ctx.fog_density:.1f}% fog). Keeping speed under 25 km/h."
        elif ctx.fog_density >= 50:
            risk = "MODERATE"
            rec_speed = min(40.0, rec_speed)
            hazard = "Moderate fog detected"
            suggestion = "Slow down and use low beams"
            priority = "visibility"
            voice = "Fog ahead, slow down"
            explanation = f"Moderate fog ({ctx.fog_density:.1f}% fog) restricting long-range sight."

    # 4. Blackspot Corridor Warning
    if blackspots and risk not in ["CRITICAL"]:
        spot = blackspots[0]
        spot_name = spot.get("name", "Blackspot")
        risk = "HIGH" if risk == "LOW" else risk
        rec_speed = min(40.0, rec_speed)
        hazard = f"Accident Zone: {spot_name}"
        suggestion = "Caution in blackspot corridor"
        voice = "Accident zone ahead, drive carefully"
        explanation = f"Driving near {spot_name} ({spot.get('severity', 'high')} risk). Speed restricted to 40 km/h."

    return {
        "risk_level": risk,
        "hazard_alert": hazard,
        "recommended_speed": f"{int(rec_speed)} km/h" if rec_speed > 0 else "0 km/h",
        "driving_suggestion": suggestion,
        "short_explanation": explanation,
        "priority_hazard": priority,
        "voice_alert": voice,
        "confidence": 0.75,
        "_source": "fallback",
    }


# ---------------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# ---------------------------------------------------------------------------
def get_llm_decision(
    context: DrivingContext | dict,
    return_raw: bool = False,
) -> dict:
    """
    Run ADAS reasoning via Ollama LLM.

    Parameters
    ----------
    context    : DrivingContext (preferred) or raw dict
    return_raw : if True, also include raw LLM text in result under "_raw"

    Returns
    -------
    Validated JSON dict with all ADAS output fields.
    Falls back to rule-based response on LLM failure.
    """
    if isinstance(context, DrivingContext):
        ctx_dict = context.to_prompt_dict()
        ctx_obj = context
    else:
        ctx_dict = context
        ctx_obj = DrivingContext(**{k: context.get(k, v)
                                    for k, v in DrivingContext.__dataclass_fields__.items()
                                      if k in context})

    user_prompt = (
        "Analyze these driving conditions and return the JSON assessment:\n\n"
        + json.dumps(ctx_dict, indent=2)
    )

    last_error: str = ""
    models_to_try = [MODEL_NAME, "puneethkumar3619/fodvision-adas:v2"]
    
    for attempt in range(1, MAX_RETRIES + 1):
        model_name = models_to_try[(attempt - 1) % len(models_to_try)]
        try:
            response = chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": 0.1},   # low temp → deterministic safety output
            )
            raw_text: str = response['message']['content'] if isinstance(response, dict) else response.message.content

            parsed = _safe_parse_json(raw_text)

            if parsed and _validate_schema(parsed):
                if return_raw:
                    parsed["_raw"] = raw_text
                parsed["_source"] = "llm"
                parsed["_model_used"] = model_name
                return parsed

            last_error = f"Schema validation failed on attempt {attempt} using {model_name}"

        except Exception as exc:
            last_error = f"Ollama error on attempt {attempt} using {model_name}: {exc}"

    # All retries exhausted → rule-based fallback
    result = _rule_based_fallback(ctx_obj)
    result["_fallback_reason"] = last_error
    return result


# ---------------------------------------------------------------------------
# DATASET GENERATOR
# ---------------------------------------------------------------------------
DATASET_EXAMPLES: list[dict] = [
    # ── CRITICAL scenarios ──────────────────────────────────────────────
    {
        "input": {"fog_density": 95, "nearest_object_m": 4, "road_type": "curve",   "fog_trend": "worsening"},
        "output": {"risk_level": "CRITICAL", "hazard_alert": "Vehicle 4m ahead in critical fog", "recommended_speed": "0 km/h",  "driving_suggestion": "Emergency brake — stop immediately", "priority_hazard": "collision", "voice_alert": "Emergency brake now"},
    },
    {
        "input": {"fog_density": 92, "nearest_object_m": 3, "road_type": "highway", "fog_trend": "worsening"},
        "output": {"risk_level": "CRITICAL", "hazard_alert": "Object 3m ahead in critical fog",          "recommended_speed": "0 km/h",  "driving_suggestion": "Brake hard and stop safely",         "priority_hazard": "collision", "voice_alert": "Stop now, object very close"},
    },
    # ── HIGH scenarios ───────────────────────────────────────────────────
    {
        "input": {"fog_density": 85, "nearest_object_m": 9, "road_type": "highway", "fog_trend": "stable"},
        "output": {"risk_level": "HIGH",     "hazard_alert": "Vehicle 9m ahead in dense fog",   "recommended_speed": "25 km/h", "driving_suggestion": "Slow down and increase following distance", "priority_hazard": "collision",   "voice_alert": "Vehicle close, slow down"},
    },
    {
        "input": {"fog_density": 78, "nearest_object_m": 15, "road_type": "curve",  "fog_trend": "worsening"},
        "output": {"risk_level": "HIGH",     "hazard_alert": "Vehicle ahead on foggy curve",          "recommended_speed": "30 km/h", "driving_suggestion": "Reduce speed immediately on curve",   "priority_hazard": "collision",   "voice_alert": "Vehicle ahead, slow down"},
    },
    {
        "input": {"fog_density": 82, "nearest_object_m": 25, "road_type": "urban",  "fog_trend": "stable"},
        "output": {"risk_level": "HIGH",     "hazard_alert": "Dense fog in urban area",                 "recommended_speed": "35 km/h", "driving_suggestion": "Use fog lights and reduce speed",      "priority_hazard": "visibility","voice_alert": "Dense fog, use fog lights"},
    },
    # ── MODERATE scenarios ───────────────────────────────────────────────
    {
        "input": {"fog_density": 60, "nearest_object_m": 18, "road_type": "highway", "fog_trend": "stable"},
        "output": {"risk_level": "MODERATE", "hazard_alert": "Moderate fog reduces visibility",         "recommended_speed": "50 km/h", "driving_suggestion": "Maintain safe distance and stay alert","priority_hazard": "visibility","voice_alert": "Fog ahead, reduce speed"},
    },
    {
        "input": {"fog_density": 55, "nearest_object_m": 12, "road_type": "wet",    "fog_trend": "improving"},
        "output": {"risk_level": "MODERATE", "hazard_alert": "Wet road with moderate fog",              "recommended_speed": "45 km/h", "driving_suggestion": "Slow for wet road surface",           "priority_hazard": "speed",     "voice_alert": "Wet road, slow down"},
    },
    # ── LOW scenarios ────────────────────────────────────────────────────
    {
        "input": {"fog_density": 20, "nearest_object_m": 40, "road_type": "highway", "fog_trend": "improving"},
        "output": {"risk_level": "LOW",      "hazard_alert": "Clear highway conditions",                "recommended_speed": "80 km/h", "driving_suggestion": "Maintain speed and stay aware",       "priority_hazard": "none",      "voice_alert": "Conditions clear, drive safely"},
    },
    {
        "input": {"fog_density": 10, "nearest_object_m": 60, "road_type": "urban",  "fog_trend": "stable"},
        "output": {"risk_level": "LOW",      "hazard_alert": "No significant hazards detected",         "recommended_speed": "50 km/h", "driving_suggestion": "Observe speed limits in urban zone",  "priority_hazard": "none",      "voice_alert": "All clear, drive carefully"},
    },
]


def generate_dataset(path: str = "adas_dataset.json") -> None:
    """Write the curated driving dataset to a JSON file."""
    with open(path, "w") as f:
        json.dump(DATASET_EXAMPLES, f, indent=2)
    print(f"[Dataset] {len(DATASET_EXAMPLES)} examples written → {path}")


# ---------------------------------------------------------------------------
# DEMO / QUICK TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    fog_memory = FogMemory()

    # Simulate a worsening fog sequence
    sensor_frames = [
        {"fog_density": 45, "detections": [{"label": "car", "distance_m": 35}]},
        {"fog_density": 60, "detections": [{"label": "car", "distance_m": 20}]},
        {"fog_density": 78, "detections": [{"label": "truck","distance_m": 9}]},
        {"fog_density": 88, "detections": [{"label": "car", "distance_m": 6}]},
    ]

    print("=" * 60)
    print("Context-Aware Fog-Adaptive ADAS — LLM Reasoning Demo")
    print("=" * 60)

    for i, frame in enumerate(sensor_frames, 1):
        fog_density = frame["fog_density"]
        trend = fog_memory.update(fog_density)
        yolo_ctx = filter_yolo_detections(frame["detections"])

        ctx = DrivingContext(
            fog_density=fog_density,
            fog_trend=trend,
            nearest_object_m=yolo_ctx["nearest_object_m"],
            nearest_object_label=yolo_ctx["nearest_object_label"],
            road_type="highway",
            current_speed_kmh=70.0,
        )

        print(f"\n── Frame {i}  fog={fog_density}  trend={trend}  nearest={ctx.nearest_object_m}m")
        result = get_llm_decision(ctx)
        print(json.dumps(result, indent=2))

    # Write dataset
    generate_dataset()
    