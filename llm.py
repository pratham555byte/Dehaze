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
You are a real-time AI-powered ADAS (Advanced Driver Assistance System) safety engine.

## Priority Order (highest to lowest)
1. Collision avoidance — vehicle < 10 m ahead
2. Visibility hazards — fog_density > 70
3. Speed adaptation   — road_type curve/wet
4. General guidance

## Fog Reasoning Rules
- fog_density >= 90  → CRITICAL visibility; cap speed at 20 km/h
- fog_density 70–89  → HIGH visibility impairment; cap speed at 40 km/h
- fog_density 50–69  → MODERATE; reduce speed by 30%
- fog_density < 50   → LOW; normal speed permissible
- fog_trend = "worsening" → escalate risk one level
- fog_trend = "improving" → may reduce risk one level

## Object Proximity Rules
- nearest_object_m < 5   → CRITICAL risk, emergency brake alert
- nearest_object_m 5–10  → HIGH risk, hard slow-down
- nearest_object_m 10–20 → MODERATE risk, increase following distance
- nearest_object_m > 20  → LOW risk from proximity

## Output Contract
Return ONLY a single valid JSON object. No markdown. No extra text.

{
  "risk_level":        "LOW | MODERATE | HIGH | CRITICAL",
  "hazard_alert":      "<primary hazard, max 10 words>",
  "recommended_speed": "<N km/h>",
  "driving_suggestion":"<actionable instruction, max 15 words>",
  "short_explanation": "<why, max 20 words>",
  "priority_hazard":   "<collision | visibility | speed | none>",
  "voice_alert":       "<max 8 words, speech-ready>",
  "confidence":        <0.0–1.0>
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
    Ensures the system never returns nothing.
    """
    risk = "LOW"
    speed = 60
    hazard = "Normal driving conditions"
    suggestion = "Maintain current speed and distance"
    priority = "none"
    voice = "Drive safely"

    if ctx.fog_density >= 90 or ctx.nearest_object_m < 5:
        risk, speed, hazard = "CRITICAL", 20, "Critical hazard detected"
        suggestion = "Brake immediately and stop safely"
        priority = "collision" if ctx.nearest_object_m < 5 else "visibility"
        voice = "Stop now, critical hazard ahead"
    elif ctx.nearest_object_m < 10:
        risk, speed, hazard = "HIGH", 30, "Vehicle very close ahead"
        suggestion = "Slow down and increase following distance"
        priority = "collision"
        voice = "Slow down, vehicle ahead"
    elif ctx.fog_density >= 70:
        risk, speed, hazard = "HIGH", 40, "Dense fog — severely reduced visibility"
        suggestion = "Reduce speed and use fog lights"
        priority = "visibility"
        voice = "Dense fog, reduce speed now"
    elif ctx.fog_density >= 50:
        risk, speed, hazard = "MODERATE", 45, "Moderate fog"
        suggestion = "Reduce speed and stay alert"
        priority = "visibility"
        voice = "Fog ahead, slow down"

    return {
        "risk_level": risk,
        "hazard_alert": hazard,
        "recommended_speed": f"{speed} km/h",
        "driving_suggestion": suggestion,
        "short_explanation": "Rule-based fallback (LLM unavailable)",
        "priority_hazard": priority,
        "voice_alert": voice,
        "confidence": 0.7,
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
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = chat(
                model=MODEL_NAME,
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
                return parsed

            last_error = f"Schema validation failed on attempt {attempt}"

        except Exception as exc:
            last_error = f"Ollama error on attempt {attempt}: {exc}"

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
    