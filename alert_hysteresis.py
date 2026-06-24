"""
alert_hysteresis.py
-------------------
Alert hysteresis system with cooldown timers and escalation logic.

Implements warn → critical escalation and cooldown periods to prevent
alert spam while ensuring important alerts are not missed.
"""

import time
from typing import Dict, Any, Optional
from enum import Enum


class AlertLevel(Enum):
    NONE = 0
    WARN = 1
    CRITICAL = 2


class AlertHysteresis:
    """
    Manages alert escalation and cooldown to prevent alert fatigue.
    """

    def __init__(self,
                 warn_cooldown: float = 5.0,    # seconds
                 critical_cooldown: float = 10.0, # seconds
                 escalation_threshold: int = 3):  # consecutive triggers
        """
        Parameters
        ----------
        warn_cooldown : float
            Minimum time between warn alerts (seconds)
        critical_cooldown : float
            Minimum time between critical alerts (seconds)
        escalation_threshold : int
            Number of consecutive warn triggers before escalating to critical
        """
        self.warn_cooldown = warn_cooldown
        self.critical_cooldown = critical_cooldown
        self.escalation_threshold = escalation_threshold

        # Track alert states
        self.alert_states: Dict[str, Dict[str, Any]] = {}

    def process_alert(self, alert_type: str, should_trigger: bool) -> Dict[str, Any]:
        """
        Process an alert through hysteresis logic.

        Parameters
        ----------
        alert_type : str
            Type of alert (e.g., 'brake_light', 'close_vehicle')
        should_trigger : bool
            Whether the raw condition for this alert is met

        Returns
        -------
        dict with alert decision
        """
        now = time.time()

        if alert_type not in self.alert_states:
            self.alert_states[alert_type] = {
                "level": AlertLevel.NONE,
                "last_warn_time": 0.0,
                "last_critical_time": 0.0,
                "consecutive_triggers": 0,
                "last_trigger_time": 0.0,
                "escalated": False
            }

        state = self.alert_states[alert_type]

        # Update consecutive trigger count
        if should_trigger:
            if now - state["last_trigger_time"] < 2.0:  # Within 2 seconds
                state["consecutive_triggers"] += 1
            else:
                state["consecutive_triggers"] = 1
            state["last_trigger_time"] = now
        else:
            state["consecutive_triggers"] = 0
            state["escalated"] = False

        # Determine alert level
        alert_level = AlertLevel.NONE
        reason = "no_trigger"

        if should_trigger:
            # Check if we can escalate to critical
            if (state["consecutive_triggers"] >= self.escalation_threshold and
                now - state["last_critical_time"] >= self.critical_cooldown):
                alert_level = AlertLevel.CRITICAL
                state["level"] = AlertLevel.CRITICAL
                state["last_critical_time"] = now
                state["escalated"] = True
                reason = "escalation_threshold_met"

            # Check if we can issue warn alert
            elif (now - state["last_warn_time"] >= self.warn_cooldown and
                  now - state["last_critical_time"] >= self.critical_cooldown):
                alert_level = AlertLevel.WARN
                state["level"] = AlertLevel.WARN
                state["last_warn_time"] = now
                reason = "warn_cooldown_expired"

            else:
                alert_level = AlertLevel.NONE
                reason = "cooldown_active"

        # Reset escalation if condition cleared
        if not should_trigger and state["escalated"]:
            state["escalated"] = False

        return {
            "alert_type": alert_type,
            "alert_level": alert_level.name,
            "should_alert": alert_level != AlertLevel.NONE,
            "reason": reason,
            "consecutive_triggers": state["consecutive_triggers"],
            "escalated": state["escalated"],
            "time_since_last_warn": now - state["last_warn_time"],
            "time_since_last_critical": now - state["last_critical_time"]
        }

    def get_alert_status(self, alert_type: str) -> Dict[str, Any]:
        """
        Get current status of an alert type.
        """
        if alert_type not in self.alert_states:
            return {"alert_type": alert_type, "status": "unknown"}

        state = self.alert_states[alert_type]
        now = time.time()

        return {
            "alert_type": alert_type,
            "current_level": state["level"].name,
            "consecutive_triggers": state["consecutive_triggers"],
            "escalated": state["escalated"],
            "warn_cooldown_remaining": max(0, self.warn_cooldown - (now - state["last_warn_time"])),
            "critical_cooldown_remaining": max(0, self.critical_cooldown - (now - state["last_critical_time"]))
        }

    def reset_alert(self, alert_type: str):
        """
        Reset an alert type to NONE state.
        """
        if alert_type in self.alert_states:
            self.alert_states[alert_type] = {
                "level": AlertLevel.NONE,
                "last_warn_time": 0.0,
                "last_critical_time": 0.0,
                "consecutive_triggers": 0,
                "last_trigger_time": 0.0,
                "escalated": False
            }

    def update(self, current_risk_level: str) -> str:
        """
        Process the current risk level through hysteresis and return stabilized alert level string.
        """
        is_active = current_risk_level in ["HIGH", "CRITICAL", "MODERATE"]
        res = self.process_alert("risk_level", is_active)
        if res["should_alert"]:
            return current_risk_level
        return "LOW"

    def get_all_alerts_status(self) -> Dict[str, Any]:
        """
        Get status of all alert types.
        """
        return {
            alert_type: self.get_alert_status(alert_type)
            for alert_type in self.alert_states.keys()
        }