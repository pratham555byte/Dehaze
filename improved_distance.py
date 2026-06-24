"""
improved_distance.py
--------------------
Enhanced distance estimation using bounding-box scaling and
monocular depth heuristics for better risk assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


_REF_CLASSES = {
    "car":          {"height_m": 1.5,  "ref_px_at_10m": 80},
    "truck":        {"height_m": 3.8,  "ref_px_at_10m": 120},
    "bus":          {"height_m": 3.2,  "ref_px_at_10m": 130},
    "person":       {"height_m": 1.75, "ref_px_at_10m": 90},
    "bicycle":      {"height_m": 1.1,  "ref_px_at_10m": 70},
    "motorcycle":   {"height_m": 1.2,  "ref_px_at_10m": 65},
    "motorbike":    {"height_m": 1.2,  "ref_px_at_10m": 65},
    "traffic light":{"height_m": 0.8,  "ref_px_at_10m": 40},
    "stop sign":    {"height_m": 0.75, "ref_px_at_10m": 35},
    "chair":        {"height_m": 0.9,  "ref_px_at_10m": 50},
    "bench":        {"height_m": 0.9,  "ref_px_at_10m": 55},
    "_default":     {"height_m": 1.5,  "ref_px_at_10m": 70},
}

_REF_DISTANCE_M = 10.0
_MIN_DIST = 1.0
_MAX_DIST = 120.0
_CALIB_HEIGHT = 480


class EnhancedDistanceEstimator:
    """
    Enhanced distance estimator using multiple depth cues.
    """

    def __init__(
        self,
        frame_height: int = 480,
        focal_length: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        frame_height : int
            Frame height in pixels
        focal_length : float, optional
            Camera focal length (computed from frame height if not provided)
        """
        self.frame_height = frame_height
        self.focal_length = focal_length or (frame_height / 2)

    def estimate_distances(
        self,
        detections: List[Dict],
        frame_height: int = 480,
    ) -> List[Dict]:
        """
        Estimate distances for all detections.

        Parameters
        ----------
        detections : List[Dict]
            List of detection dicts with 'bbox' and 'label'
        frame_height : int
            Frame height in pixels

        Returns
        -------
        List of detections with added 'distance' field
        """
        if frame_height != self.frame_height:
            self.frame_height = frame_height
            self.focal_length = frame_height / 2

        results = []

        for det in detections:
            label = det.get("label", "_default").lower()
            bbox = det.get("bbox", [0, 0, 0, 0])

            if len(bbox) != 4:
                results.append({**det, "distance": 50.0})
                continue

            distance = self._estimate_from_bbox(bbox, label)
            results.append({**det, "distance": distance})

        return results

    def _estimate_from_bbox(self, bbox: List[int], label: str) -> float:
        """
        Estimate distance from bounding box using pinhole model.
        """
        ref_info = _REF_CLASSES.get(label, _REF_CLASSES["_default"])
        real_height_m = ref_info["height_m"]
        ref_px = ref_info["ref_px_at_10m"]

        box_height = max(bbox[3] - bbox[1], 1)

        scale = _CALIB_HEIGHT / self.frame_height
        scaled_ref_px = ref_px * scale

        dist = (scaled_ref_px * _REF_DISTANCE_M) / box_height

        dist = max(_MIN_DIST, min(dist, _MAX_DIST))

        return round(dist, 1)

    def estimate_distance_from_y_position(
        self,
        bbox: List[int],
        frame_height: int = 480,
    ) -> float:
        """
        Alternative distance estimation using vertical position in frame.
        Objects lower in the frame are closer.
        """
        y_bottom = bbox[3]

        if y_bottom >= frame_height - 1:
            y_bottom = frame_height - 2

        normalized_y = y_bottom / frame_height

        distance = 5.0 + (1.0 - normalized_y) * 95.0

        return max(_MIN_DIST, min(distance, _MAX_DIST))

    def compute_depth_confidence(
        self,
        bbox: List[int],
        label: str,
    ) -> float:
        """
        Compute confidence in distance estimate based on detection characteristics.

        Returns
        -------
        Confidence score between 0 and 1
        """
        box_height = bbox[3] - bbox[1]
        box_width = bbox[2] - bbox[0]
        aspect_ratio = box_width / max(box_height, 1)

        ref_info = _REF_CLASSES.get(label, _REF_CLASSES["_default"])
        expected_ratio = 1.5
        ratio_diff = abs(aspect_ratio - expected_ratio) / expected_ratio

        size_score = 1.0 - min(ratio_diff, 1.0)

        y_position_factor = bbox[3] / self.frame_height
        if 0.3 < y_position_factor < 0.9:
            position_score = 1.0
        else:
            position_score = 0.7

        confidence = (size_score * 0.6 + position_score * 0.4)

        return round(confidence, 2)

    def get_distance_category(self, distance_m: float) -> str:
        """Categorize distance for alert logic."""
        if distance_m < 10:
            return "very_close"
        elif distance_m < 25:
            return "close"
        elif distance_m < 50:
            return "moderate"
        elif distance_m < 80:
            return "far"
        else:
            return "very_far"


def estimate_distances_simple(detections: List[Dict], frame_height: int = 480) -> List[Dict]:
    """Simple functional interface for distance estimation."""
    estimator = EnhancedDistanceEstimator(frame_height)
    return estimator.estimate_distances(detections, frame_height)


def get_distance_with_confidence(detection: Dict, frame_height: int = 480) -> Tuple[float, float]:
    """Get distance estimate with confidence score."""
    estimator = EnhancedDistanceEstimator(frame_height)
    bbox = detection.get("bbox", [0, 0, 0, 0])
    label = detection.get("label", "unknown")
    distance = estimator._estimate_from_bbox(bbox, label)
    confidence = estimator.compute_depth_confidence(bbox, label)
    return distance, confidence