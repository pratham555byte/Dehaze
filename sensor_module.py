import time

class SensorProcessing:
    def __init__(self, ema_alpha=0.20, max_valid_range=300.0):
        self.ema_alpha = ema_alpha
        self.max_valid_range = max_valid_range
        
        # History for velocity calculations
        self.last_distance = None
        self.last_time = None
        
        # Output values
        self.filtered_distances = [max_valid_range] * 3 # [Left, Center, Right]
        self.raw_relative_velocity = 0.0
        self.relative_velocity = 0.0 # smoothed

    def process(self, l, c, r):
        """
        Processes distance readings (in cm) and estimates relative velocity based on the center sensor.
        """
        current_time = time.time()
        
        # 1. Apply outlier filter
        raw_dists = [l, c, r]
        self.filtered_distances = [
            min(float(d), self.max_valid_range) if d is not None else self.max_valid_range 
            for d in raw_dists
        ]
        
        # 2. Primary follow distance comes from Center sensor
        current_center_distance = self.filtered_distances[1]
        
        # 3. Calculate Relative Velocity based on Center sensor
        if self.last_distance is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 1e-4:
                delta_d = current_center_distance - self.last_distance
                self.raw_relative_velocity = (delta_d / 100.0) / dt # convert cm to meters
                
                # Cap extremely large velocities (sensor jumps / noise)
                if -10.0 <= self.raw_relative_velocity <= 10.0:
                    self.relative_velocity = (
                        self.ema_alpha * self.raw_relative_velocity + 
                        (1.0 - self.ema_alpha) * self.relative_velocity
                    )
        else:
            self.relative_velocity = 0.0
            
        # Update history
        self.last_distance = current_center_distance
        self.last_time = current_time
        
        return self.filtered_distances, self.relative_velocity
