"""
risk_alerts.py
--------------
Risk-based alert logic that generates alerts only when:
1. Object is in the current driving lane
2. Object is approaching the vehicle
3. Time-to-Collision is below threshold (low TTC)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AlertCondition:
    """Represents an alert condition for collision avoidance."""
    track_id: int
    label: str
    distance: float
    ttc: float
    is_in_lane: bool
    is_approaching: bool
    risk_level: str
    alert_type: str
    message: str


class RiskBasedAlertSystem:
    """
    Generates risk-based alerts based on object position, motion, and TTC.
    """

    def __init__(
        self,
        ttc_warning_threshold: float = 5.0,
        ttc_critical_threshold: float = 2.0,
        min_distance_warning: float = 20.0,
        min_distance_critical: float = 10.0,
        alert_cooldown_seconds: float = 3.0,
    ):
        """
        Parameters
        ----------
        ttc_warning_threshold : float
            TTC threshold for warning alerts (seconds)
        ttc_critical_threshold : float
            TTC threshold for critical alerts (seconds)
        min_distance_warning : float
            Minimum distance for warning (meters)
        min_distance_critical : float
            Minimum distance for critical (meters)
        alert_cooldown_seconds : float
            Cooldown between same-type alerts
        """
        self.ttc_warning = ttc_warning_threshold
        self.ttc_critical = ttc_critical_threshold
        self.min_dist_warning = min_distance_warning
        self.min_dist_critical = min_distance_critical
        self.cooldown = alert_cooldown_seconds

        self._last_alert_time: Dict[str, float] = {}
        self._alert_history: List[AlertCondition] = []

    def evaluate_alerts(
        self,
        in_lane_objects: List[Dict],
        motion_analysis: List[Dict],
        frame_timestamp: float,
    ) -> List[Dict]:
        """
        Evaluate and generate alerts based on current conditions.

        Parameters
        ----------
        in_lane_objects : List[Dict]
            Objects detected in the current driving lane
        motion_analysis : List[Dict]
            Motion analysis results for tracked objects
        frame_timestamp : float
            Current frame timestamp

        Returns
        -------
        List of alert dicts
        """
        alerts = []

        in_lane_ids = {obj.get("track_id") for obj in in_lane_objects if "track_id" in obj}

        for motion in motion_analysis:
            track_id = motion.get("track_id")
            if track_id not in in_lane_ids:
                continue

            if not motion.get("is_approaching", False):
                continue

            alert = self._create_alert(motion, frame_timestamp)
            if alert is not None:
                alerts.append(alert)

        alerts.sort(key=lambda x: (0 if x["severity"] == "critical" else 1, x["ttc"]))

        return alerts[:5]

    def _create_alert(self, motion: Dict, timestamp: float) -> Optional[Dict]:
        """Create an alert if conditions are met."""
        track_id = motion.get("track_id", 0)
        label = motion.get("label", "unknown")
        distance = motion.get("distance", 100.0)
        ttc = motion.get("ttc", float('inf'))
        risk_level = motion.get("risk_level", "safe")

        if ttc == float('inf') or ttc > self.ttc_warning:
            return None

        if distance < self.min_dist_warning and risk_level == "safe":
            return None

        alert_type, severity, message = self._determine_alert_type(
            ttc, distance, risk_level, label
        )

        alert_key = f"{alert_type}_{track_id}"
        if self._should_suppress_alert(alert_key, timestamp):
            return None

        self._last_alert_time[alert_key] = timestamp

        alert = {
            "type": alert_type,
            "severity": severity,
            "track_id": track_id,
            "label": label,
            "distance": round(distance, 1),
            "ttc": round(ttc, 2),
            "message": message,
            "timestamp": timestamp,
            "in_lane": True,
            "approaching": True,
        }

        self._alert_history.append(AlertCondition(
            track_id=track_id,
            label=label,
            distance=distance,
            ttc=ttc,
            is_in_lane=True,
            is_approaching=True,
            risk_level=risk_level,
            alert_type=alert_type,
            message=message,
        ))

        if len(self._alert_history) > 100:
            self._alert_history = self._alert_history[-100:]

        return alert

    def _determine_alert_type(
        self,
        ttc: float,
        distance: float,
        risk_level: str,
        label: str,
    ) -> Tuple[str, str, str]:
        """Determine the appropriate alert type based on conditions."""
        if ttc < self.ttc_critical or distance < self.min_dist_critical:
            alert_type = "collision_warning"
            severity = "critical"
            message = f"Critical: {label} close - brake immediately!"

        elif ttc < self.ttc_warning or risk_level == "high":
            alert_type = "approach_warning"
            severity = "warning"
            message = f"Warning: {label} approaching - {distance:.0f}m"

        elif distance < 30 and risk_level == "medium":
            alert_type = "caution"
            severity = "caution"
            message = f"Caution: {label} at {distance:.0f}m"

        else:
            return ("safe", "none", "")

        return alert_type, severity, message

    def _should_suppress_alert(self, alert_key: str, timestamp: float) -> bool:
        """Check if alert should be suppressed due to cooldown."""
        if alert_key not in self._last_alert_time:
            return False

        last_time = self._last_alert_time[alert_key]
        return (timestamp - last_time) < self.cooldown

    def get_alert_statistics(self) -> Dict:
        """Get statistics about generated alerts."""
        if not self._alert_history:
            return {
                "total_alerts": 0,
                "critical_count": 0,
                "warning_count": 0,
                "caution_count": 0,
            }

        critical = sum(1 for a in self._alert_history if a.alert_type == "collision_warning")
        warning = sum(1 for a in self._alert_history if a.alert_type == "approach_warning")
        caution = sum(1 for a in self._alert_history if a.alert_type == "caution")

        return {
            "total_alerts": len(self._alert_history),
            "critical_count": critical,
            "warning_count": warning,
            "caution_count": caution,
        }

    def reset(self):
        """Reset alert system state."""
        self._last_alert_time.clear()
        self._alert_history.clear()


def generate_risk_alerts(
    in_lane_objects: List[Dict],
    motion_analysis: List[Dict],
    timestamp: float,
) -> List[Dict]:
    """Simple functional interface for alert generation."""
    system = RiskBasedAlertSystem()
    return system.evaluate_alerts(in_lane_objects, motion_analysis, timestamp)