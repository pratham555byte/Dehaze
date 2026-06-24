import pygame

class KeyboardControl:
    def __init__(self, max_speed=200, max_steering=45):
        self.max_speed = max_speed
        self.max_steering = max_steering
        
        # Current states
        self.throttle = 0.0
        self.steering = 0.0
        self.acc_enabled = False
        
        # Smooth control parameters
        self.speed_step = 25.0 # how fast speed ramps up per second/frame
        self.steer_step = 90.0 # how fast wheels return to center or turn
        
        # Debounce for toggle keys
        self.m_key_pressed = False

    def update(self, events, dt):
        """
        Processes key inputs.
        events: Pygame events from pygame.event.get()
        dt: Time delta in seconds
        """
        # 1. Handle toggles from Pygame Event queue
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.acc_enabled = not self.acc_enabled
                    print(f"[ControlModule] ACC Enabled: {self.acc_enabled}")

        # 2. Get active keys pressed for continuous controls
        keys = pygame.key.get_pressed()
        
        # Handle Steering (Left/Right)
        target_steering = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            target_steering = -self.max_steering
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            target_steering = self.max_steering
            
        # Smooth steering transition (ramp up and center)
        steer_diff = target_steering - self.steering
        if steer_diff != 0:
            step = self.steer_step * dt * (1.0 if steer_diff > 0 else -1.0)
            if abs(step) > abs(steer_diff):
                self.steering = target_steering
            else:
                self.steering += step
        else:
            self.steering = target_steering

        # Handle Throttle (Forward/Backward)
        # If ACC is active, speed is controlled autonomously by the ACC module,
        # but the user can still steer.
        target_throttle = 0.0
        is_braking = keys[pygame.K_SPACE]
        
        if is_braking:
            self.throttle = 0.0
        else:
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                target_throttle = self.max_speed
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                target_throttle = -self.max_speed
            else:
                # Coasting to stop
                target_throttle = 0.0
                
            # Smooth throttle transition (ramping speed)
            if target_throttle > self.throttle:
                self.throttle = min(self.throttle + self.speed_step * dt * 8, target_throttle)
            elif target_throttle < self.throttle:
                # Decelerate faster than accelerate (adds responsiveness)
                self.throttle = max(self.throttle - self.speed_step * dt * 12, target_throttle)

        return int(self.throttle), int(self.steering), self.acc_enabled
