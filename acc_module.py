class AdaptiveCruiseControl:
    def __init__(self, target_distance=50.0, critical_distance=25.0, Kp=4.5, Kd=90.0):
        """
        target_distance: desired following distance in cm
        critical_distance: safety threshold in cm below which the car must stop
        Kp: proportional gain
        Kd: derivative gain
        """
        self.target_distance = target_distance
        self.critical_distance = critical_distance
        self.Kp = Kp
        self.Kd = Kd
        self.max_acc_speed = 100

    def compute_control(self, acc_enabled, target_throttle, filtered_distances, relative_velocity):
        """
        Computes final throttle.
        filtered_distances: [Left, Center, Right] in cm
        relative_velocity: obstacle relative velocity in m/s (positive = moving away, negative = approaching)
        """
        left_dist = filtered_distances[0]
        center_dist = filtered_distances[1]
        right_dist = filtered_distances[2]
        min_dist = min(left_dist, center_dist, right_dist)

        # Hazard check — ANY sensor below critical distance triggers emergency stop
        is_critical = min_dist < self.critical_distance
        # Warning zone — any sensor in cautionary range
        is_warning = min_dist < (self.critical_distance * 2.0)

        # TTC-based approaching entity detection
        # If the obstacle is approaching (negative relative velocity) and close
        is_approaching = (relative_velocity < -0.02) and (center_dist < self.target_distance * 1.5)

        if not acc_enabled:
            # Manual Mode Override
            if target_throttle < 0:
                final_throttle = target_throttle
                acc_status_text = "MANUAL (REVERSING)"
            elif center_dist < self.critical_distance:
                # Brakes applied automatically ONLY when obstacle is directly in front
                final_throttle = 0
                acc_status_text = f"OVERRIDE: EMERGENCY BRAKE ({center_dist:.0f}cm)"
            elif is_approaching and center_dist < self.critical_distance * 2.0:
                final_throttle = 0
                acc_status_text = f"OVERRIDE: APPROACHING ENTITY ({center_dist:.0f}cm)"
            else:
                final_throttle = target_throttle
                acc_status_text = "MANUAL"
        else:
            # Autonomous Cruise Control
            if is_critical:
                final_throttle = 0
                acc_status_text = f"ACC: CRITICAL STOP ({min_dist:.0f}cm)"
            elif is_approaching:
                final_throttle = 0
                acc_status_text = f"ACC: ENTITY APPROACHING ({center_dist:.0f}cm)"
            else:
                # Error in follow distance (cm)
                error = center_dist - self.target_distance
                
                # PD loop
                u = self.Kp * error + self.Kd * relative_velocity
                
                final_throttle = max(0, min(int(u), self.max_acc_speed))
                acc_status_text = f"ACC ACTIVE (Follow: {center_dist:.1f}cm)"

        return final_throttle, acc_status_text

