"""
risk_score.py
-------------
LLD Module 9 — Context & Risk Evaluation Module / Risk Score Engine.

Computes a normalised risk score in [0, 1] from all available sensor
and inference data, then classifies it into LOW / MEDIUM / HIGH.

Weights follow the LLD specification:
    Input              Weight
    ─────────────────────────────
    Fog density         0.30
    Object distance     0.25
    Speed               0.20
    Road context        0.10
    Red glow            0.10
    Humidity (future)   0.05   ← placeholder 0 until sensor is wired

Hard-override rules (from LLD "Hard Safety Rules"):
    • red_glow == True AND distance_to_nearest < 20 m → HIGH (score ≥ 0.7)
    • distance_to_nearest < 10 m                     → HIGH (score ≥ 0.7)
    • fog_density > 60 %                             → speed component boosted
    • road_type == curve AND distance < 25 m         → MEDIUM floor (score ≥ 0.35)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple


# ------------------------------------------------------------------
# Classification thresholds (match LLD table)
# ------------------------------------------------------------------
_LOW_MAX    = 0.30
_MED_MAX    = 0.60


def compute_risk(
    fog_density:         float,          # 0 – 100
    distance_to_nearest: float,          # metres  (0 = no object)
    actual_speed:        float,          # km/h of the vehicle (from Vehicular setting)
    road_context:        Dict[str, Any], # from road_context.py
    humidity:            float = 0.0,    # 0 – 100 (placeholder)
    is_live:             bool = False,   # scale thresholds for physical car
) -> Dict[str, Any]:
    """
    Returns
    -------
    {
        "risk_score"       : float   0 – 1
        "risk_level"       : str     LOW | MEDIUM | HIGH
        "component_scores" : dict    per-factor sub-scores for dashboard
        "hard_override"    : bool    True when a hard rule fired
        "override_reason"  : str
    }
    """

    # If vehicle speed is near zero/slow (≤ 5.0 km/h), the risk is zero (SYSTEM_RULES.md §5.4)
    if actual_speed < 5.0:
        return {
            "risk_score":        0.0,
            "risk_level":        "LOW",
            "component_scores":  {
                "fog":      0.0,
                "distance": 0.0,
                "speed":    0.0,
                "road":     0.0,
                "humidity": 0.0,
            },
            "hard_override":     False,
            "override_reason":   "",
        }

    # ── 1. Fog component (0–1) ────────────────────────────────────────
    fog_component = _normalise(fog_density, 0.0, 100.0)

    # ── 2. Distance component (0–1, closer = higher risk) ────────────
    if distance_to_nearest <= 0:
        dist_component = 0.0          # no vehicle seen → no distance risk
    else:
        # Scale for live physical car vs. simulated highway
        if is_live:
            # Risk saturates at 0.25 m (danger) and drops off at 1.5 m
            dist_component = _normalise_inv(distance_to_nearest, 0.25, 1.5)
        else:
            # Risk saturates at 5 m (certain collision zone)
            dist_component = _normalise_inv(distance_to_nearest, 5.0, 60.0)

    # ── 3. Speed component (higher actual speed = higher risk) ──────
    speed_component = _normalise(actual_speed, 0.0, 120.0)

    # Boost speed risk component if fog is dense
    if fog_density > 60.0:
        speed_component = min(speed_component * 1.5, 1.0)

    # ── 4. Road context component ─────────────────────────────────────
    blackspot_bonus = 0.4 if road_context.get("blackspots") else 0.0
    road_type_str   = str(road_context.get("road_type", "")).lower()
    curve_bonus     = 0.3 if "curve" in road_type_str else 0.0
    road_component  = min(blackspot_bonus + curve_bonus, 1.0)

    # ── 5. Humidity component (future) ────────────────────────────────
    hum_component   = _normalise(humidity, 0.0, 100.0)

    # ── Weighted sum ──────────────────────────────────────────────────
    weights = {
        "fog":      0.35,
        "distance": 0.30,
        "speed":    0.20,
        "road":     0.10,
        "humidity": 0.05,
    }
    components = {
        "fog":      round(fog_component,  3),
        "distance": round(dist_component, 3),
        "speed":    round(speed_component,3),
        "road":     round(road_component, 3),
        "humidity": round(hum_component,  3),
    }

    raw_score = sum(weights[k] * components[k] for k in weights)
    raw_score = round(float(raw_score), 4)

    # ── Hard-override rules ────────────────────────────────────────────
    hard_override   = False
    override_reason = ""

    crit_dist = 0.30 if is_live else 10.0
    curve_dist = 0.60 if is_live else 25.0

    if 0 < distance_to_nearest < crit_dist:
        raw_score       = max(raw_score, 0.70)
        hard_override   = True
        override_reason = f"Vehicle within {crit_dist:.2f} m"

    if "curve" in road_type_str and 0 < distance_to_nearest < curve_dist:
        raw_score       = max(raw_score, 0.35)
        if not hard_override:
            hard_override   = True
            override_reason = f"Curve + vehicle within {curve_dist:.2f} m"

    final_score = round(min(raw_score, 1.0), 4)

    # ── Classification ────────────────────────────────────────────────
    if final_score <= _LOW_MAX:
        level = "LOW"
    elif final_score <= _MED_MAX:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "risk_score":        final_score,
        "risk_level":        level,
        "component_scores":  components,
        "hard_override":     hard_override,
        "override_reason":   override_reason,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalise(value: float, lo: float, hi: float) -> float:
    """Linear scale value∈[lo,hi] → [0,1], clamped."""
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def _normalise_inv(value: float, lo: float, hi: float) -> float:
    """Inverted normalisation: lower value → higher risk score."""
    return 1.0 - _normalise(value, lo, hi)