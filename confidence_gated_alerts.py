"""
confidence_gated_alerts.py
--------------------------
Confidence-gated alert system that only triggers alerts when
YOLO confidence meets threshold for N consecutive frames.
"""

from typing import List, Dict, Any, Optional
from collections import deque


class ConfidenceGatedAlerts:
    """
    Manages alert gating based on confidence thresholds and temporal consistency.
    """

    def __init__(self, confidence_threshold: float = 0.5, frames_required: int = 3):
        """
        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence score for detection (0-1)
        frames_required : int
            Number of consecutive frames detection must be present
        """
        self.confidence_threshold = confidence_threshold
        self.frames_required = frames_required

        # Track detections per class
        self.detection_buffers: Dict[str, deque] = {}
        self.active_alerts: Dict[str, bool] = {}

    def update_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update with new frame detections and determine which alerts should fire.

        Parameters
        ----------
        detections : list
            List of detection dictionaries from YOLO

        Returns
        -------
        dict with gated alerts and confidence states
        """
        # Group detections by label
        current_detections = {}
        for det in detections:
            label = det.get('label', 'unknown')
            conf = det.get('confidence', 0.0)

            if conf >= self.confidence_threshold:
                if label not in current_detections:
                    current_detections[label] = []
                current_detections[label].append(det)

        # Update buffers for each label
        alerts_to_fire = {}
        confidence_states = {}

        # Check all possible labels (including those not detected this frame)
        all_labels = set(self.detection_buffers.keys()) | set(current_detections.keys())

        for label in all_labels:
            if label not in self.detection_buffers:
                self.detection_buffers[label] = deque(maxlen=self.frames_required)

            # Add detection state for this frame
            has_detection = label in current_detections and len(current_detections[label]) > 0
            self.detection_buffers[label].append(has_detection)

            # Check if we have enough consecutive detections
            buffer_list = list(self.detection_buffers[label])
            consecutive_count = 0
            for state in reversed(buffer_list):
                if state:
                    consecutive_count += 1
                else:
                    break

            # Determine if alert should fire
            should_alert = consecutive_count >= self.frames_required
            alerts_to_fire[label] = should_alert

            # Update active alerts state
            was_active = self.active_alerts.get(label, False)
            self.active_alerts[label] = should_alert

            # Confidence state info
            confidence_states[label] = {
                "consecutive_frames": consecutive_count,
                "buffer_length": len(buffer_list),
                "should_alert": should_alert,
                "just_activated": should_alert and not was_active,
                "just_deactivated": not should_alert and was_active,
                "detections_this_frame": len(current_detections.get(label, []))
            }

        return {
            "alerts_to_fire": alerts_to_fire,
            "confidence_states": confidence_states,
            "gated_detections": current_detections
        }

    def reset_label(self, label: str):
        """
        Reset the confidence buffer for a specific label.
        Useful when an object leaves the frame.
        """
        if label in self.detection_buffers:
            self.detection_buffers[label].clear()
        if label in self.active_alerts:
            self.active_alerts[label] = False

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of current alert states.
        """
        return {
            "active_alerts": {k: v for k, v in self.active_alerts.items() if v},
            "monitored_labels": list(self.detection_buffers.keys()),
            "confidence_threshold": self.confidence_threshold,
            "frames_required": self.frames_required
        }