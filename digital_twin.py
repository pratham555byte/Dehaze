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
        
        # Define Bumper Position reference dynamically
        self.car_front_y = self.height - 125.0
        
        # Simulated obstacles at random places on the road
        self.sim_obstacles = []
        for _ in range(4):
            self.sim_obstacles.append({
                "lane": random.choice([0, 1, 2]),
                "y": random.uniform(-400.0, self.height - 150.0),
                "type": random.choice(["cone", "barrier", "car"])
            })
        self.pixel_scale = 1.2 # 1 cm = 1.2 pixels (300 cm max = 360 px)

    def update(self, throttle, steering_deg, connected, real_speed, dt, dist_l=300.0, dist_c=300.0, dist_r=300.0, space_pressed=False):
        """
        Updates the digital twin states.
        If connected, uses real sensor ranges.
        If disconnected, runs offline simulation.
        """
        self.connected = connected
        self.real_speed = real_speed
        self.throttle = throttle
        self.space_pressed = space_pressed
        
        if connected:
            self.speed = -real_speed if throttle < 0 else real_speed
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
            self.speed = max(-5.0, min(15.0, self.speed))
                
            self.steering_angle = math.radians(steering_deg)
            
            # Update virtual lateral position
            # Steering changes lane position X
            self.x += self.speed * 20.0 * math.sin(self.steering_angle) * dt
            self.x = max(170.0, min(410.0, self.x))
            
            # Update simulated obstacles position at a constant speed (completely independent of car speed)
            for obs in self.sim_obstacles:
                # Skip automatic updates for the currently dragged obstacle
                if getattr(self, "dragged_obs", None) == obs:
                    continue
                    
                obs["y"] += 120.0 * dt # constant scroll speed
                
                # If obstacle goes behind the car, reset to top at a random lane and Y
                if obs["y"] > self.height + 50.0:
                    obs["y"] = -random.randint(150, 450)
                    obs["lane"] = random.choice([0, 1, 2])
                    obs["type"] = random.choice(["cone", "barrier", "car"])
            
            # Compute simulated sensor readings based on distance to nearest obstacles using 2D ray casting
            self.sensor_readings = [self.max_sensor_range] * 3
            for i, angle_offset in enumerate(self.sensor_angles):
                # The sensor ray starts at (self.x, self.car_front_y)
                # Direction of the ray (steering yaw + sensor angle offset)
                ray_angle = self.steering_angle + angle_offset
                dx = math.sin(ray_angle)
                dy = -math.cos(ray_angle)
                
                min_sensor_dist = self.max_sensor_range
                
                # Check intersections with obstacles (circle radius R=25 pixels)
                for obs in self.sim_obstacles:
                    obs_x = 190.0 if obs["lane"] == 0 else (290.0 if obs["lane"] == 1 else 390.0)
                    obs_y = obs["y"]
                    
                    # Vector from car front bumper to obstacle center
                    vx = obs_x - self.x
                    vy = obs_y - self.car_front_y
                    
                    # Project vector onto sensor ray direction
                    proj = vx * dx + vy * dy
                    if proj > 0: # Obstacle is in front of the sensor ray
                        # Closest distance squared from circle center to ray line
                        dist_sq = (vx * vx + vy * vy) - (proj * proj)
                        R = 25.0 # obstacle radius in pixels (~20 cm)
                        if dist_sq <= R * R:
                            t = proj - math.sqrt(max(0.0, R * R - dist_sq))
                            dist_cm = t / self.pixel_scale
                            if dist_cm < min_sensor_dist:
                                min_sensor_dist = dist_cm
                                
                if min_sensor_dist < self.max_sensor_range:
                    noise = random.gauss(0, 0.5)
                    self.sensor_readings[i] = min(self.max_sensor_range, max(0.0, min_sensor_dist + noise))

        # Scroll road lines at speed
        self.road_scroll = (self.road_scroll + self.speed * 30.0 * dt) % 80.0
        return self.sensor_readings

    def handle_mouse(self, mouse_pos, is_pressed):
        """Allows clicking and dragging simulated obstacles in disconnected mode."""
        if getattr(self, "connected", False):
            return
            
        mx, my = mouse_pos
        if is_pressed[0]: # Left mouse button down
            if not hasattr(self, "dragged_obs") or self.dragged_obs is None:
                # Find closest obstacle to mouse position
                closest_obs = None
                min_dist = 60.0 # max drag target selection distance in pixels
                
                # Compute visual coordinates of each obstacle
                lateral_offset = -(self.x - self.width / 2.0)
                left_edge = 140.0 + lateral_offset
                lane_width = 100.0
                
                for obs in self.sim_obstacles:
                    obs_x = left_edge + obs["lane"] * lane_width + lane_width / 2.0
                    obs_y = obs["y"]
                    
                    dist = math.hypot(mx - obs_x, my - obs_y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_obs = obs
                self.dragged_obs = closest_obs
            
            if self.dragged_obs is not None:
                # Update dragged obstacle Y position
                self.dragged_obs["y"] = float(my)
                self.dragged_obs["y"] = max(-200.0, min(self.height - 50.0, self.dragged_obs["y"]))
                
                # Dynamically snap to closest lane based on horizontal coordinate
                lateral_offset = -(self.x - self.width / 2.0)
                left_edge = 140.0 + lateral_offset
                rel_mx = mx - left_edge
                if rel_mx < 100.0:
                    self.dragged_obs["lane"] = 0
                elif rel_mx < 200.0:
                    self.dragged_obs["lane"] = 1
                else:
                    self.dragged_obs["lane"] = 2
        else:
            self.dragged_obs = None

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

        # Draw Simulated Obstacles on the road (only in offline/disconnected mode)
        if not getattr(self, 'connected', False):
            for obs in self.sim_obstacles:
                obs_x = left_edge + obs["lane"] * lane_width + lane_width / 2.0
                obs_y = obs["y"]
                
                # Check if obstacle is on screen
                if -100.0 < obs_y < self.height + 100.0:
                    obs_type = obs.get("type", "barrier")
                    if obs_type == "car":
                        # Draw a premium leading car
                        car_w = 28.0
                        car_h = 52.0
                        # Shadow
                        pygame.draw.rect(self.surface, (5, 5, 5), 
                                         (int(obs_x - car_w/2 - 2), int(obs_y - car_h/2 - 2), car_w + 4, car_h + 4), border_radius=8)
                        # Main Body (red/crimson)
                        pygame.draw.rect(self.surface, (185, 28, 28), 
                                         (int(obs_x - car_w/2), int(obs_y - car_h/2), car_w, car_h), border_radius=6)
                        pygame.draw.rect(self.surface, (239, 68, 68), 
                                         (int(obs_x - car_w/2), int(obs_y - car_h/2), car_w, car_h), 2, border_radius=6)
                        # Windshield/Windows
                        pygame.draw.rect(self.surface, (30, 41, 59), 
                                         (int(obs_x - 10), int(obs_y - 12), 20, 8), border_radius=2)
                        # Rear glass
                        pygame.draw.rect(self.surface, (30, 41, 59), 
                                         (int(obs_x - 10), int(obs_y + 12), 20, 4), border_radius=1)
                        # Headlights (yellow)
                        pygame.draw.rect(self.surface, (253, 224, 71), (int(obs_x - 12), int(obs_y - car_h/2), 4, 3))
                        pygame.draw.rect(self.surface, (253, 224, 71), (int(obs_x + 8), int(obs_y - car_h/2), 4, 3))
                        # Tail lights (red)
                        pygame.draw.rect(self.surface, (220, 38, 38), (int(obs_x - 12), int(obs_y + car_h/2 - 3), 4, 3))
                        pygame.draw.rect(self.surface, (220, 38, 38), (int(obs_x + 8), int(obs_y + car_h/2 - 3), 4, 3))
                    elif obs_type == "cone":
                        # Draw a traffic cone (orange triangle)
                        pygame.draw.polygon(self.surface, (245, 158, 11), [
                            (int(obs_x), int(obs_y - 14)),
                            (int(obs_x - 10), int(obs_y + 14)),
                            (int(obs_x + 10), int(obs_y + 14))
                        ])
                        # Stripe
                        pygame.draw.polygon(self.surface, (255, 255, 255), [
                            (int(obs_x), int(obs_y - 4)),
                            (int(obs_x - 6), int(obs_y + 4)),
                            (int(obs_x + 6), int(obs_y + 4))
                        ])
                        # Cone base
                        pygame.draw.rect(self.surface, (194, 65, 12), 
                                         (int(obs_x - 14), int(obs_y + 12), 28, 4), border_radius=1)
                    else: # barrier
                        # Draw a striped barricade
                        bar_w = 48
                        bar_h = 16
                        # Legs/Stands
                        pygame.draw.rect(self.surface, (71, 85, 105), (int(obs_x - bar_w/2 + 4), int(obs_y - bar_h/2 - 2), 4, bar_h + 4))
                        pygame.draw.rect(self.surface, (71, 85, 105), (int(obs_x + bar_w/2 - 8), int(obs_y - bar_h/2 - 2), 4, bar_h + 4))
                        # Board
                        pygame.draw.rect(self.surface, (220, 38, 38), 
                                         (int(obs_x - bar_w/2), int(obs_y - bar_h/2), bar_w, bar_h), border_radius=2)
                        # Stripes (white lines on red board)
                        for stripe_x in range(int(obs_x - bar_w/2 + 6), int(obs_x + bar_w/2), 12):
                            pygame.draw.line(self.surface, (255, 255, 255), 
                                             (stripe_x, int(obs_y - bar_h/2)), 
                                             (stripe_x - 6, int(obs_y + bar_h/2)), 3)

        # 2. Project Sensor Beams and Obstacles
        car_front_x = self.width / 2.0
        car_front_y = self.height - 125.0 # Bumper y position
        
        for i, angle_offset in enumerate(self.sensor_angles):
            # Calculate beam angle
            ray_angle = self.steering_angle + angle_offset
            # Standard trigonometry: 0° points straight up (-y direction)
            dx = math.sin(ray_angle)
            dy = -math.cos(ray_angle)
            
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
        
        # HUD: Emergency Brake Indicator (SYSTEM_RULES.md)
        if getattr(self, 'emergency_brake_active', False):
            brake_reason = getattr(self, 'emergency_brake_reason', 'EMERGENCY BRAKE')
            # Flash red banner across top
            banner_surf = pygame.Surface((self.width, 30), pygame.SRCALPHA)
            banner_surf.fill((220, 38, 38, 200))
            self.surface.blit(banner_surf, (0, 60))
            font_alert = pygame.font.SysFont("Consolas", 16, bold=True)
            alert_txt = font_alert.render(f"[!] {brake_reason}", True, (255, 255, 255))
            txt_rect = alert_txt.get_rect(center=(self.width // 2, 75))
            self.surface.blit(alert_txt, txt_rect)
            
        # Draw brake lights if emergency brake is active, space is pressed, or reversing/braking throttle is active
        is_braking = getattr(self, 'emergency_brake_active', False) or getattr(self, 'space_pressed', False) or getattr(self, 'throttle', 0.0) < 0
        if is_braking:
            # Brake lights on car (red glow at rear)
            pygame.draw.circle(self.surface, (239, 68, 68), (int(car_front_x - 12), int(car_front_y + self.length_car - 10)), 5)
            pygame.draw.circle(self.surface, (239, 68, 68), (int(car_front_x + 12), int(car_front_y + self.length_car - 10)), 5)
        
        # HUD: Max safe speed indicator
        max_safe = getattr(self, 'max_safe_speed_kmh', 100.0)
        if speed_kmh > max_safe:
            overspeed_txt = font_hud.render(f"OVERSPEEDING! MAX: {max_safe:.0f} km/h", True, (239, 68, 68))
            self.surface.blit(overspeed_txt, (self.width - 300, 20))
        else:
            safe_txt = font_hud.render(f"SAFE SPEED: {max_safe:.0f} km/h", True, (16, 185, 129))
            self.surface.blit(safe_txt, (self.width - 260, 20))
        
        # Outer Border
        pygame.draw.rect(self.surface, (30, 41, 59), (0, 0, self.width, self.height), 3)
        
        return self.surface
