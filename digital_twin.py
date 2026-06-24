import pygame
import math
import random

class DigitalTwin:
    def __init__(self, width=580, height=480):
        self.width = width
        self.height = height
        
        # In-memory rendering surface (headless friendly)
        self.surface = pygame.Surface((self.width, self.height))
        
        # Vehicle physical properties
        self.wheelbase = 35.0
        self.width_car = 25.0
        self.length_car = 50.0
        
        # Vehicle state (positions in pixels, angles in radians)
        self.x = width / 2.0
        self.y = height - 100.0
        self.theta = -math.pi / 2.0
        self.speed = 0.0
        self.steering_angle = 0.0
        
        # Static obstacles
        self.obstacles = [
            {"x": 120.0, "y": 150.0, "r": 30.0, "color": (100, 100, 120)},
            {"x": 460.0, "y": 200.0, "r": 35.0, "color": (100, 100, 120)},
            {"x": 200.0, "y": 280.0, "r": 25.0, "color": (100, 100, 120)},
        ]
        
        # Dynamic leading vehicle
        self.lead_x = width / 2.0
        self.lead_y = height / 2.0 - 50.0
        self.lead_r = 18.0
        self.lead_speed = 35.0
        self.lead_direction = -1
        
        # Sensor configuration (3 sensors: Left (-20°), Center (0°), Right (+20°))
        self.sensor_angles = [math.radians(-20), math.radians(0), math.radians(20)]
        self.max_sensor_range = 300.0
        self.sensor_readings = [self.max_sensor_range] * 3
        self.sensor_impact_pts = [None] * 3
        
        # Mouse dragging state
        self.dragging_lead = False

    def update(self, throttle, steering_deg, connected, real_speed, dt):
        """
        Updates simulated ego car physics, lead car pathing, and raycasts.
        """
        # 1. Update vehicle state
        if connected:
            self.speed = real_speed * 10.0 # scale physical speed
            self.steering_angle = math.radians(steering_deg)
        else:
            target_speed = (throttle / 200.0) * 120.0
            self.speed += (target_speed - self.speed) * 5.0 * dt
            self.steering_angle = math.radians(steering_deg)

        # Kinematic Bicycle Model
        if abs(self.speed) > 1.0:
            self.theta += (self.speed / self.wheelbase) * math.tan(self.steering_angle) * dt
            self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
            
        self.x += self.speed * math.cos(self.theta) * dt
        self.y += self.speed * math.sin(self.theta) * dt
        
        self.x = max(20.0, min(self.width - 20.0, self.x))
        self.y = max(20.0, min(self.height - 20.0, self.y))

        # 2. Update leading vehicle movement (if not user dragged)
        if not self.dragging_lead:
            self.lead_y += self.lead_speed * self.lead_direction * dt
            if self.lead_y < 60.0:
                self.lead_y = 60.0
                self.lead_direction = 1
            elif self.lead_y > self.height - 60.0:
                self.lead_y = self.height - 60.0
                self.lead_direction = -1

        # 3. Simulate VL53L0X Raycasting
        car_front_x = self.x + (self.length_car / 2.0) * math.cos(self.theta)
        car_front_y = self.y + (self.length_car / 2.0) * math.sin(self.theta)
        
        for i, angle_offset in enumerate(self.sensor_angles):
            ray_angle = self.theta + angle_offset
            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)
            
            closest_dist = self.max_sensor_range
            impact_pt = (
                car_front_x + ray_dx * self.max_sensor_range,
                car_front_y + ray_dy * self.max_sensor_range
            )
            
            # Static Obstacles
            for obs in self.obstacles:
                dist = self._ray_circle_intersection(car_front_x, car_front_y, ray_dx, ray_dy, obs["x"], obs["y"], obs["r"])
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    impact_pt = (car_front_x + ray_dx * dist, car_front_y + ray_dy * dist)
                    
            # Lead Vehicle
            dist_lead = self._ray_circle_intersection(car_front_x, car_front_y, ray_dx, ray_dy, self.lead_x, self.lead_y, self.lead_r)
            if dist_lead is not None and dist_lead < closest_dist:
                closest_dist = dist_lead
                impact_pt = (car_front_x + ray_dx * dist_lead, car_front_y + ray_dy * dist_lead)
                
            # Boundaries
            dist_boundary = self._ray_boundary_intersection(car_front_x, car_front_y, ray_dx, ray_dy)
            if dist_boundary < closest_dist:
                closest_dist = dist_boundary
                impact_pt = (car_front_x + ray_dx * dist_boundary, car_front_y + ray_dy * dist_boundary)

            noise = random.gauss(0, 1.0)
            self.sensor_readings[i] = max(0.0, closest_dist + noise)
            self.sensor_impact_pts[i] = impact_pt

        return self.sensor_readings

    def handle_mouse(self, mouse_pos, is_pressed):
        """
        Receives mouse drag telemetry from web requests.
        is_pressed: boolean indicating if primary mouse button is held
        """
        mx, my = mouse_pos
        dx = mx - self.lead_x
        dy = my - self.lead_y
        dist = math.hypot(dx, dy)
        
        if is_pressed:
            if dist < self.lead_r + 15 or self.dragging_lead:
                # Constrain lead vehicle inside boundaries
                self.lead_x = max(15.0, min(self.width - 15.0, mx))
                self.lead_y = max(15.0, min(self.height - 15.0, my))
                self.dragging_lead = True
        else:
            self.dragging_lead = False

    def _ray_circle_intersection(self, rx, ry, rdx, rdy, cx, cy, cr):
        ocx = cx - rx
        ocy = cy - ry
        projection = ocx * rdx + ocy * rdy
        if projection < 0:
            return None
            
        closest_x = rx + rdx * projection
        closest_y = ry + rdy * projection
        
        dist_sq = (closest_x - cx)**2 + (closest_y - cy)**2
        r_sq = cr**2
        
        if dist_sq > r_sq:
            return None
            
        half_chord = math.sqrt(r_sq - dist_sq)
        t = projection - half_chord
        if t < 0:
            t = projection + half_chord
            
        return t if t >= 0 else None

    def _ray_boundary_intersection(self, rx, ry, rdx, rdy):
        t_min = self.max_sensor_range
        
        if rdx > 0:
            t = (self.width - rx) / rdx
            if 0 <= t < t_min: t_min = t
        elif rdx < 0:
            t = -rx / rdx
            if 0 <= t < t_min: t_min = t
            
        if rdy > 0:
            t = (self.height - ry) / rdy
            if 0 <= t < t_min: t_min = t
        elif rdy < 0:
            t = -ry / rdy
            if 0 <= t < t_min: t_min = t
            
        return t_min

    def render(self):
        """
        Renders the twin onto its internal pygame.Surface, returning it.
        """
        self.surface.fill((20, 20, 25))
        
        # Border
        pygame.draw.rect(self.surface, (60, 60, 75), (0, 0, self.width, self.height), 4)
        
        # Grid lines
        for x in range(0, self.width, 40):
            pygame.draw.line(self.surface, (30, 30, 35), (x, 0), (x, self.height))
        for y in range(0, self.height, 40):
            pygame.draw.line(self.surface, (30, 30, 35), (0, y), (self.width, y))

        # Static obstacles
        for obs in self.obstacles:
            pygame.draw.circle(self.surface, obs["color"], (int(obs["x"]), int(obs["y"])), int(obs["r"]))
            pygame.draw.circle(self.surface, (130, 130, 150), (int(obs["x"]), int(obs["y"])), int(obs["r"]), 2)

        # Leading vehicle
        lead_color = (235, 75, 75) if self.dragging_lead else (200, 50, 50)
        pygame.draw.circle(self.surface, lead_color, (int(self.lead_x), int(self.lead_y)), int(self.lead_r))
        pygame.draw.circle(self.surface, (255, 100, 100), (int(self.lead_x), int(self.lead_y)), int(self.lead_r), 2)
        
        lead_dir = self.lead_direction
        pygame.draw.polygon(self.surface, (255, 255, 255), [
            (self.lead_x, self.lead_y + lead_dir * 10),
            (self.lead_x - 5, self.lead_y - lead_dir * 3),
            (self.lead_x + 5, self.lead_y - lead_dir * 3)
        ])

        # Sensor rays (3 sensors)
        car_front_x = self.x + (self.length_car / 2.0) * math.cos(self.theta)
        car_front_y = self.y + (self.length_car / 2.0) * math.sin(self.theta)
        
        for i, impact in enumerate(self.sensor_impact_pts):
            if impact is not None:
                dist = self.sensor_readings[i]
                if dist < 25.0:
                    color = (255, 50, 50)
                elif dist < 50.0:
                    color = (255, 200, 50)
                else:
                    color = (50, 200, 50)
                    
                pygame.draw.line(self.surface, color, (int(car_front_x), int(car_front_y)), (int(impact[0]), int(impact[1])), 1)
                pygame.draw.circle(self.surface, color, (int(impact[0]), int(impact[1])), 3)

        # Vehicle body
        half_w = self.width_car / 2.0
        half_l = self.length_car / 2.0
        corners = [
            (-half_l, -half_w),
            (half_l, -half_w),
            (half_l, half_w),
            (-half_l, half_w)
        ]
        
        rotated_corners = []
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        for cx, cy in corners:
            rx = self.x + cx * cos_t - cy * sin_t
            ry = self.y + cx * sin_t + cy * cos_t
            rotated_corners.append((rx, ry))

        pygame.draw.polygon(self.surface, (60, 140, 240), rotated_corners)
        pygame.draw.polygon(self.surface, (100, 180, 255), rotated_corners, 2)

        # Wheels
        wheel_w = 10
        wheel_h = 4
        for is_right in [False, True]:
            f_offset_x = half_l - 6
            f_offset_y = half_w if is_right else -half_w
            fw_x = self.x + f_offset_x * cos_t - f_offset_y * sin_t
            fw_y = self.y + f_offset_x * sin_t + f_offset_y * cos_t
            
            wheel_theta = self.theta + self.steering_angle
            w_cos = math.cos(wheel_theta)
            w_sin = math.sin(wheel_theta)
            
            w_corners = [
                (-wheel_w/2, -wheel_h/2), (wheel_w/2, -wheel_h/2),
                (wheel_w/2, wheel_h/2), (-wheel_w/2, wheel_h/2)
            ]
            rot_w_corners = []
            for wx, wy in w_corners:
                rwx = fw_x + wx * w_cos - wy * w_sin
                rwy = fw_y + wx * w_sin + wy * w_cos
                rot_w_corners.append((rwx, rwy))
            pygame.draw.polygon(self.surface, (20, 20, 20), rot_w_corners)

        for is_right in [False, True]:
            r_offset_x = -half_l + 6
            r_offset_y = half_w if is_right else -half_w
            rw_x = self.x + r_offset_x * cos_t - r_offset_y * sin_t
            rw_y = self.y + r_offset_x * sin_t + r_offset_y * cos_t
            
            w_corners = [
                (-wheel_w/2, -wheel_h/2), (wheel_w/2, -wheel_h/2),
                (wheel_w/2, wheel_h/2), (-wheel_w/2, wheel_h/2)
            ]
            rot_w_corners = []
            for wx, wy in w_corners:
                rwx = rw_x + wx * cos_t - wy * sin_t
                rwy = rw_y + wx * sin_t + wy * cos_t
                rot_w_corners.append((rwx, rwy))
            pygame.draw.polygon(self.surface, (20, 20, 20), rot_w_corners)

        pygame.draw.line(self.surface, (255, 255, 255), 
                         (int(self.x), int(self.y)), 
                         (int(self.x + 18 * cos_t), int(self.y + 18 * sin_t)), 2)
                         
        return self.surface
