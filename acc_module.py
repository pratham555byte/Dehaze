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
        self.max_acc_speed = 150

    def compute_control(self, acc_enabled, target_throttle, filtered_distances, relative_velocity):
        """
        Computes final throttle.
        filtered_distances: [Left, Center, Right] in cm
        relative_velocity: obstacle relative velocity in m/s (positive = moving away, negative = approaching)
        """
        # Center sensor is the primary guide
        center_dist = filtered_distances[1]
        
        # Hazard check
        is_hazard = center_dist < self.critical_distance

        if not acc_enabled:
            # Manual Mode Override
            if is_hazard and target_throttle > 0:
                final_throttle = 0
                acc_status_text = "OVERRIDE: SAFETY BRAKE"
            else:
                final_throttle = target_throttle
                acc_status_text = "MANUAL"
        else:
            # Autonomous Cruise Control
            if is_hazard:
                final_throttle = 0
                acc_status_text = "ACC: CRITICAL STOP"
            else:
                # Error in follow distance (cm)
                error = center_dist - self.target_distance
                
                # PD loop
                u = self.Kp * error + self.Kd * relative_velocity
                
                final_throttle = max(0, min(int(u), self.max_acc_speed))
                acc_status_text = f"ACC ACTIVE (Follow: {center_dist:.1f}cm)"

        return final_throttle, acc_status_text
