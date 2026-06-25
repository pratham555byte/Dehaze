import pygame
import math
import random

class DigitalTwin:
    def __init__(self, width=580, height=480):
        self.width = width
        self.height = height
        
        # In-memory rendering surface
        self.surface = pygame.Surface((self.width, self.height))
        
        # Ego vehicle dimensions
        self.width_car = 24.0
        self.length_car = 48.0
        
        # Vehicle visual/physics state
        self.x = width / 2.0 # virtual X coordinate (290 is center)
        self.speed = 0.0     # m/s
        self.steering_angle = 0.0 # radians
        
        # Lanes/Road boundaries
        self.road_scroll = 0.0
        
        # Sensor configuration (3 sensors: Left (-20°), Center (0°), Right (+20°))
        self.sensor_angles = [math.radians(-20), math.radians(0), math.radians(20)]
        self.max_sensor_range = 300.0 # cm
        self.sensor_readings = [self.max_sensor_range] * 3
        
        # Simulated obstacles for offline mode
        # Format: {"lane": 0/1/2, "y": float_pos_pixels, "type": "cone" | "barrier"}
        self.sim_obstacles = [
            {"lane": 0, "y": -100.0, "type": "barrier"},
            {"lane": 1, "y": -350.0, "type": "cone"},
            {"lane": 2, "y": -200.0, "type": "barrier"}
        ]
        self.pixel_scale = 1.2 # 1 cm = 1.2 pixels (300 cm max = 360 px)

    def update(self, throttle, steering_deg, connected, real_speed, dt, dist_l=300.0, dist_c=300.0, dist_r=300.0):
        """
        Updates the digital twin states.
        If connected, uses real sensor ranges.
        If disconnected, runs offline simulation.
        """
        self.connected = connected
        self.real_speed = real_speed
        if connected:
            self.speed = real_speed
            self.steering_angle = math.radians(steering_deg)
            # Use active sensor inputs
            self.sensor_readings = [
                min(self.max_sensor_range, max(0.0, dist_l)),
                min(self.max_sensor_range, max(0.0, dist_c)),
                min(self.max_sensor_range, max(0.0, dist_r))
            ]
            # Update virtual lateral position based on actual speed and steering
            self.x += self.speed * 20.0 * math.sin(self.steering_angle) * dt
            self.x = max(170.0, min(410.0, self.x))
        else:
            # Simulated offline kinetics
            target_speed = (throttle / 200.0) * 15.0 # Max simulated speed ~15 m/s
            self.speed += (target_speed - self.speed) * 4.0 * dt
            if self.speed < 0:
                self.speed = 0.0
                
            self.steering_angle = math.radians(steering_deg)
            
            # Update virtual lateral position
            # Steering changes lane position X
            self.x += self.speed * 20.0 * math.sin(self.steering_angle) * dt
            self.x = max(170.0, min(410.0, self.x))
            
            # Update simulated obstacles position
            for obs in self.sim_obstacles:
                obs["y"] += self.speed * 30.0 * dt # scroll speed
                
                # If obstacle goes behind the car, reset to top at a random distance
                if obs["y"] > self.height + 50.0:
                    obs["y"] = -random.randint(150, 450)
            
            # Compute simulated sensor readings based on distance to nearest obstacles
            for i, angle in enumerate(self.sensor_angles):
                lane_idx = i # Left sensor maps to Left lane (0), Center to Center (1), Right to Right (2)
                
                # Find matching obstacle
                matching_obs = [o for o in self.sim_obstacles if o["lane"] == lane_idx]
                if matching_obs:
                    obs = matching_obs[0]
                    # Car bumper is at y = 355
                    dist_px = 355.0 - obs["y"]
                    if dist_px > 0:
                        dist_cm = dist_px / self.pixel_scale
                        noise = random.gauss(0, 1.0)
                        self.sensor_readings[i] = min(self.max_sensor_range, max(0.0, dist_cm + noise))
                    else:
                        self.sensor_readings[i] = self.max_sensor_range
                else:
                    self.sensor_readings[i] = self.max_sensor_range

        # Scroll road lines at speed
        self.road_scroll = (self.road_scroll + self.speed * 30.0 * dt) % 80.0
        return self.sensor_readings

    def handle_mouse(self, mouse_pos, is_pressed):
        """No longer used since we removed manual coordinate-dragging of fake cars."""
        pass

    def render(self):
        """
        Renders the HUD and scrolling road in the style of a premium EV FSD display.
        The Ego vehicle remains centered, while the environment, obstacles and lanes scroll.
        """
        self.surface.fill((10, 12, 16)) # Dark obsidian cockpit background
        
        # Centering offset (camera follows car laterally)
        lateral_offset = -(self.x - self.width / 2.0)
        
        # 1. Draw Highway boundaries and lanes (moving laterally relative to ego car)
        left_edge = 140.0 + lateral_offset
        right_edge = 440.0 + lateral_offset
        lane_width = 100.0
        
        # Road asphalt fill
        pygame.draw.polygon(self.surface, (18, 22, 28), [
            (left_edge, 0), (right_edge, 0),
            (right_edge, self.height), (left_edge, self.height)
        ])
        
        # Road edge lines (solid blue/grey)
        pygame.draw.line(self.surface, (54, 73, 98), (left_edge, 0), (left_edge, self.height), 3)
        pygame.draw.line(self.surface, (54, 73, 98), (right_edge, 0), (right_edge, self.height), 3)
        
        # Lane divider dashed lines (scrolling downward)
        for offset_y in range(int(self.road_scroll) - 80, self.height + 80, 80):
            # Left lane marker
            pygame.draw.line(self.surface, (100, 116, 139), 
                             (left_edge + lane_width, offset_y), 
                             (left_edge + lane_width, offset_y + 40), 2)
            # Right lane marker
            pygame.draw.line(self.surface, (100, 116, 139), 
                             (left_edge + 2 * lane_width, offset_y), 
                             (left_edge + 2 * lane_width, offset_y + 40), 2)

        # 2. Project Sensor Beams and Obstacles
        car_front_x = self.width / 2.0
        car_front_y = self.height - 125.0 # Bumper y position
        
        for i, angle_offset in enumerate(self.sensor_angles):
            # Calculate beam angle
            ray_angle = self.steering_angle + angle_offset
            # Standard trigonometry: 0° points straight up (-y direction)
            dx = math.sin(angle_offset)
            dy = -math.cos(angle_offset)
            
            dist_cm = self.sensor_readings[i]
            dist_px = dist_cm * self.pixel_scale
            
            impact_x = car_front_x + dx * dist_px
            impact_y = car_front_y + dy * dist_px
            
            # Select color based on distance threat
            if dist_cm < 40.0:
                beam_color = (239, 68, 68)   # Alert Red
                glow_color = (254, 226, 226)
            elif dist_cm < 100.0:
                beam_color = (245, 158, 11)  # Alert Amber/Yellow
                glow_color = (254, 243, 199)
            else:
                beam_color = (16, 185, 129)  # ADAS Safe Green
                glow_color = (209, 250, 229)
                
            # Draw sensor beam path
            if dist_cm < self.max_sensor_range:
                # Solid sensor line to obstacle
                pygame.draw.line(self.surface, beam_color, (int(car_front_x), int(car_front_y)), (int(impact_x), int(impact_y)), 2)
                
                # Draw hazard obstacle barricade
                obs_w = 40
                obs_h = 16
                pygame.draw.rect(self.surface, beam_color, 
                                 (int(impact_x - obs_w/2), int(impact_y - obs_h/2), obs_w, obs_h), border_radius=4)
                pygame.draw.rect(self.surface, glow_color, 
                                 (int(impact_x - obs_w/2), int(impact_y - obs_h/2), obs_w, obs_h), 2, border_radius=4)
                
                # Draw warning light glow on the hazard
                pygame.draw.circle(self.surface, beam_color, (int(impact_x), int(impact_y)), 6)
                pygame.draw.circle(self.surface, (255, 255, 255), (int(impact_x), int(impact_y)), 2)
                
                # Draw distance text next to the obstacle
                font = pygame.font.SysFont("Consolas", 12, bold=True)
                txt = font.render(f"{int(dist_cm)}cm", True, (255, 255, 255))
                self.surface.blit(txt, (int(impact_x) + 24, int(impact_y) - 6))
            else:
                # Dotted faint ray showing clear range
                for step in range(0, int(self.max_sensor_range * self.pixel_scale), 15):
                    rx = car_front_x + dx * step
                    ry = car_front_y + dy * step
                    pygame.draw.circle(self.surface, (16, 185, 129, 120), (int(rx), int(ry)), 1)

        # 3. Draw Ego Car body (Centered statically at bottom, tilts with steering)
        car_surf = pygame.Surface((self.width_car + 20, self.length_car + 20), pygame.SRCALPHA)
        
        # Center of local surf
        cx, cy = car_surf.get_width() / 2, car_surf.get_height() / 2
        
        # Chassis body rect
        body_rect = pygame.Rect(cx - self.width_car/2, cy - self.length_car/2, self.width_car, self.length_car)
        pygame.draw.rect(car_surf, (59, 130, 246), body_rect, border_radius=6) # Blue chassis
        pygame.draw.rect(car_surf, (147, 197, 253), body_rect, 2, border_radius=6) # Cyan outline
        
        # Windshield
        pygame.draw.rect(car_surf, (30, 41, 59), (cx - 8, cy - 12, 16, 8), border_radius=2)
        # Headlights
        pygame.draw.rect(car_surf, (253, 224, 71), (cx - 10, cy - self.length_car/2, 4, 3))
        pygame.draw.rect(car_surf, (253, 224, 71), (cx + 6, cy - self.length_car/2, 4, 3))
        
        # Rotate car chassis based on steering visual yaw
        yaw_deg = -math.degrees(self.steering_angle) * 0.4
        yaw_deg = max(-25.0, min(25.0, yaw_deg))
        rotated_car = pygame.transform.rotate(car_surf, yaw_deg)
        rot_rect = rotated_car.get_rect(center=(int(car_front_x), int(car_front_y + self.length_car/2)))
        
        self.surface.blit(rotated_car, rot_rect.topleft)
        
        # 4. Premium Dashboard HUD Overlays
        font_hud = pygame.font.SysFont("Consolas", 14, bold=True)
        
        # HUD: speed display
        speed_kmh = self.speed * 3.6
        speed_txt = font_hud.render(f"SPEED: {speed_kmh:.1f} km/h", True, (255, 255, 255))
        self.surface.blit(speed_txt, (20, 20))
        
        # HUD: mode display
        mode_str = "CONNECTED TELEMETRY" if getattr(self, 'connected', False) else "ACC SIMULATING"
        mode_color = (59, 130, 246) if getattr(self, 'connected', False) else (245, 158, 11)
        mode_txt = font_hud.render(f"MODE: {mode_str}", True, mode_color)
        self.surface.blit(mode_txt, (20, 40))
        
        # Outer Border
        pygame.draw.rect(self.surface, (30, 41, 59), (0, 0, self.width, self.height), 3)
        
        return self.surface
