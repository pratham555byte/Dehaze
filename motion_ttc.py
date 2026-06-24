"""
motion_ttc.py
-------------
Relative motion analysis and Time-to-Collision (TTC) calculation.

This module analyzes object motion across frames to determine if objects
are approaching the vehicle and calculates time-to-collision for
collision threat assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MotionState:
    """Represents the motion state of a tracked object."""
    track_id: int
    label: str
    relative_velocity_x: float
    relative_velocity_y: float
    distance: float
    ttc: float
    is_approaching: bool
    risk_level: str


class MotionAnalyzer:
    """
    Analyzes relative motion and computes Time-to-Collision.
    """

    def __init__(
        self,
        fps: float = 30.0,
        camera_height_m: float = 1.0,
        camera_tilt_deg: float = 15.0,
        ego_speed_kmh: float = 50.0,
    ):
        """
        Parameters
        ----------
        fps : float
            Frames per second of the video
        camera_height_m : float
            Height of camera above ground (meters)
        camera_tilt_deg : float
            Camera tilt angle (degrees)
        ego_speed_kmh : float
            Assumed ego vehicle speed (km/h)
        """
        self.fps = fps
        self.camera_height = camera_height_m
        self.camera_tilt = np.radians(camera_tilt_deg)
        self.ego_speed_ms = ego_speed_kmh * 1000 / 3600

        self._horizon_y = None
        self._focal_length = None

    def analyze_motion(
        self,
        tracks: List[Dict],
        frame_height: int = 480,
        ego_speed_kmh: float = 50.0,
    ) -> List[Dict]:
        """
        Analyze motion of tracked objects.

        Parameters
        ----------
        tracks : List[Dict]
            List of tracked object dicts from object_tracker
        frame_height : int
            Frame height in pixels
        ego_speed_kmh : float
            Current ego vehicle speed (km/h)

        Returns
        -------
        List of motion analysis results
        """
        self.ego_speed_ms = ego_speed_kmh * 1000 / 3600

        if self._horizon_y is None:
            self._horizon_y = int(frame_height * 0.45)

        if self._focal_length is None:
            self._focal_length = frame_height / (2 * np.tan(np.pi / 3))

        results = []

        for track in tracks:
            motion = self._analyze_single_track(track)
            results.append(motion)

        return results

    def _analyze_single_track(self, track: Dict) -> Dict:
        """Analyze motion of a single track."""
        track_id = track.get("track_id", 0)
        label = track.get("label", "unknown")
        bbox = track.get("bbox", [0, 0, 0, 0])
        velocity = track.get("velocity", (0.0, 0.0))
        distance = track.get("distance", 10.0)

        vx, vy = velocity
        speed = np.sqrt(vx**2 + vy**2)

        is_approaching = vy > 1.0

        rel_velocity_y = -vy * self.fps
        ttc = self._calculate_ttc(distance, rel_velocity_y)

        risk_level = self._assess_risk(ttc, distance, is_approaching)

        return {
            "track_id": track_id,
            "label": label,
            "bbox": bbox,
            "distance": distance,
            "velocity_pixels": velocity,
            "speed_pixels": speed,
            "relative_velocity_ms": self._px_to_ms(rel_velocity_y),
            "ttc": ttc,
            "is_approaching": is_approaching,
            "risk_level": risk_level,
            "motion_direction": self._get_motion_direction(vx, vy),
        }

    def _calculate_ttc(self, distance_m: float, rel_velocity_px: float) -> float:
        """
        Calculate Time-to-Collision.

        Parameters
        ----------
        distance_m : float
            Distance to object (meters)
        rel_velocity_px : float
            Relative velocity in pixels per frame

        Returns
        -------
        TTC in seconds
        """
        rel_velocity_ms = self._px_to_ms(rel_velocity_px)

        if rel_velocity_ms >= -0.1:
            return float('inf')

        ttc = distance_m / (-rel_velocity_ms)

        return max(0.0, min(ttc, 100.0))

    def _px_to_ms(self, velocity_px: float) -> float:
        """Convert pixel velocity to meters per second."""
        return velocity_px * self.ego_speed_ms / 100.0

    def _assess_risk(self, ttc: float, distance: float, is_approaching: bool) -> str:
        """
        Assess risk level based on TTC and distance.

        Parameters
        ----------
        ttc : float
            Time-to-collision in seconds
        distance : float
            Distance to object in meters
        is_approaching : bool
            Whether object is approaching

        Returns
        -------
        Risk level: "critical", "high", "medium", "low", "safe"
        """
        if not is_approaching:
            return "safe"

        if ttc < 1.5:
            return "critical"
        elif ttc < 3.0:
            return "high"
        elif ttc < 5.0:
            return "medium"
        elif distance < 20.0:
            return "low"
        else:
            return "safe"

    def _get_motion_direction(self, vx: float, vy: float) -> str:
        """Determine motion direction from velocity components."""
        speed = np.sqrt(vx**2 + vy**2)

        if speed < 0.5:
            return "stationary"

        if abs(vy) > abs(vx):
            return "approaching" if vy > 0 else "receding"
        else:
            return "left" if vx < 0 else "right"

    def compute_ttc_for_detection(
        self,
        bbox: List[int],
        prev_bbox: Optional[List[int]],
        distance_m: float,
    ) -> float:
        """
        Compute TTC for a single detection (for non-tracked objects).

        Parameters
        ----------
        bbox : List[int]
            Current bounding box [x1, y1, x2, y2]
        prev_bbox : List[int], optional
            Previous bounding box
        distance_m : float
            Estimated distance in meters

        Returns
        -------
        TTC in seconds
        """
        if prev_bbox is None:
            return float('inf')

        bbox_center_y = (bbox[1] + bbox[3]) / 2
        prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2

        dy = bbox_center_y - prev_center_y
        rel_velocity_px = dy * self.fps

        return self._calculate_ttc(distance_m, rel_velocity_px)

    def get_collision_threats(
        self,
        motion_analysis: List[Dict],
        ttc_threshold: float = 5.0,
        min_distance: float = 5.0,
    ) -> List[Dict]:
        """
        Get list of objects that are collision threats.

        Parameters
        ----------
        motion_analysis : List[Dict]
            Results from analyze_motion
        ttc_threshold : float
            Maximum TTC to consider a threat (seconds)
        min_distance : float
            Minimum distance to consider (meters)

        Returns
        -------
        List of collision threats
        """
        threats = []

        for analysis in motion_analysis:
            if (
                analysis["is_approaching"]
                and analysis["ttc"] < ttc_threshold
                and analysis["distance"] > min_distance
            ):
                threats.append(analysis)

        threats.sort(key=lambda x: x["ttc"])

        return threats

    def estimate_distance_from_bbox(
        self,
        bbox: List[int],
        frame_height: int = 480,
        object_height_m: float = 1.5,
    ) -> float:
        """
        Estimate distance from bounding box using pinhole model.

        Parameters
        ----------
        bbox : List[int]
            Bounding box [x1, y1, x2, y2]
        frame_height : int
            Frame height in pixels
        object_height_m : float
            Real height of object in meters

        Returns
        -------
        Estimated distance in meters
        """
        box_height = bbox[3] - bbox[1]

        if box_height <= 0:
            return 50.0

        if self._focal_length is None:
            self._focal_length = frame_height / 2

        distance = (object_height_m * self._focal_length) / box_height

        return max(1.0, min(distance, 100.0))


def analyze_motion_simple(tracks: List[Dict], frame_height: int = 480) -> List[Dict]:
    """Simple functional interface for motion analysis."""
    analyzer = MotionAnalyzer()
    return analyzer.analyze_motion(tracks, frame_height)


def compute_ttc_simple(distance: float, velocity: float) -> float:
    """Simple functional interface for TTC calculation."""
    if velocity >= 0:
        return float('inf')
    ttc = distance / (-velocity)
    return max(0.0, min(ttc, 100.0))