"""
sensor_fusion.py
----------------
Camera-LiDAR sensor fusion with conflict resolution.

PLACEHOLDER - Uses vision-based fallback until real LiDAR connected.
Combines vision-based detections with distance measurements.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from lidar_sensor import LidarDistanceEstimator, LidarRoadCurvature


class SensorFusion:
    """
    Fuses camera and sensor data with documented decision rules.
    Uses vision-based estimation as fallback until real LiDAR connected.
    """

    def __init__(self):
        self.lidar_distance = LidarDistanceEstimator()
        self.lidar_curvature = LidarRoadCurvature()
        self.lidar_connected = False

        # Fusion parameters
        self.distance_confidence_threshold = 0.7
        self.vision_lidar_distance_tolerance = 5.0  # meters

    def connect_lidar(self) -> bool:
        """Connect to real LiDAR sensor."""
        connected = self.lidar_distance.connect_sensor()
        self.lidar_connected = connected
        if connected:
            print("[SensorFusion] LiDAR connected successfully")
        else:
            print("[SensorFusion] LiDAR not connected - using vision fallback")
        return connected

    def fuse_detections(self, camera_detections: List[Dict[str, Any]],
                       image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Fuse camera detections with distance measurements.

        Parameters
        ----------
        camera_detections : list
            YOLO detections from camera
        image_shape : tuple
            (height, width) of camera image

        Returns
        -------
        Fused detection results with distance source info
        """
        # Get distance-enhanced detections
        enhanced_detections = self.lidar_distance.estimate_distances(
            camera_detections, image_shape
        )

        # Track stats
        fusion_stats = {
            "total_detections": len(camera_detections),
            "lidar_distances_used": 0,
            "vision_distances_used": 0,
            "lidar_connected": self.lidar_connected
        }

        for det in enhanced_detections:
            source = det.get('distance_source', 'unknown')
            if source == 'lidar':
                fusion_stats["lidar_distances_used"] += 1
            else:
                fusion_stats["vision_distances_used"] += 1

        return {
            "fused_detections": enhanced_detections,
            "conflicts": [],
            "fusion_stats": fusion_stats
        }

    def get_road_context_fusion(self) -> Dict[str, Any]:
        """
        Get road context. Returns placeholder until real sensor connected.
        """
        curvature_data = self.lidar_curvature.analyze_curvature()

        road_context = {
            "road": "Unknown Road",
            "road_type": "unknown",
            "curvature": curvature_data["curvature"],
            "curvature_radius": curvature_data["radius"],
            "sensor_source": "placeholder",
            "confidence": 0.0,
            "lidar_connected": self.lidar_connected,
            "blackspots": []
        }

        return road_context

    def get_fusion_health(self) -> Dict[str, Any]:
        """
        Get health status of sensor fusion system.
        """
        return {
            "lidar_status": "connected" if self.lidar_connected else "not_connected",
            "camera_status": "operational",
            "calibration_status": "pending",
            "fusion_confidence": 0.5 if self.lidar_connected else 0.3
        }