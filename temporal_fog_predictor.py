"""
temporal_fog_predictor.py
-------------------------
Temporal fog drift prediction using rolling fog density values.

Implements exponential moving average and linear regression for
predicting next fog density state.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from collections import deque


class TemporalFogPredictor:
    """
    Predicts fog density drift using historical data.
    """

    def __init__(self, window_size: int = 10, alpha: float = 0.3):
        """
        Parameters
        ----------
        window_size : int
            Number of recent fog density readings to keep
        alpha : float
            Smoothing factor for exponential moving average (0-1)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.fog_history: deque = deque(maxlen=window_size)
        self.ema: Optional[float] = None

    def update(self, fog_density: float) -> Dict[str, Any]:
        """
        Update with new fog density reading and compute predictions.

        Parameters
        ----------
        fog_density : float
            Current fog density percentage (0-100)

        Returns
        -------
        dict with prediction data
        """
        self.fog_history.append(fog_density)

        # Update exponential moving average
        if self.ema is None:
            self.ema = fog_density
        else:
            self.ema = self.alpha * fog_density + (1 - self.alpha) * self.ema

        # Linear regression prediction
        trend_prediction = self._predict_next_linear()

        # Drift direction
        drift = "stable"
        if len(self.fog_history) >= 2:
            recent_change = fog_density - self.fog_history[-2]
            if recent_change > 2:
                drift = "increasing"
            elif recent_change < -2:
                drift = "decreasing"

        return {
            "current_fog": fog_density,
            "ema_fog": round(self.ema, 2),
            "predicted_next": round(trend_prediction, 2),
            "drift_direction": drift,
            "history_length": len(self.fog_history),
            "confidence": min(len(self.fog_history) / self.window_size, 1.0)
        }

    def _predict_next_linear(self) -> float:
        """
        Simple linear regression prediction for next fog density.
        """
        if len(self.fog_history) < 3:
            return self.fog_history[-1] if self.fog_history else 0.0

        # Use last 5 points for regression
        data = list(self.fog_history)[-5:]
        x = np.arange(len(data))
        y = np.array(data)

        # Linear regression
        try:
            slope, intercept = np.polyfit(x, y, 1)
            next_value = slope * len(data) + intercept
            # Clamp to reasonable range
            return np.clip(next_value, 0, 100)
        except:
            return self.fog_history[-1]

    def get_trend_analysis(self) -> Dict[str, Any]:
        """
        Analyze fog density trends.
        """
        if len(self.fog_history) < 3:
            return {"trend": "insufficient_data"}

        recent = list(self.fog_history)[-3:]
        slope = (recent[-1] - recent[0]) / 2  # Change over last 2 readings

        if slope > 5:
            trend = "rapidly_increasing"
        elif slope > 1:
            trend = "gradually_increasing"
        elif slope < -5:
            trend = "rapidly_decreasing"
        elif slope < -1:
            trend = "gradually_decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 2),
            "volatility": np.std(recent) if len(recent) > 1 else 0
        }