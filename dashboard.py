import pygame
import sys
import cv2
import time
import math

from dehaze_module import DehazeModule
from esp32_module import ESP32Communication
from sensor_module import SensorProcessing
from acc_module import AdaptiveCruiseControl
from digital_twin import DigitalTwin
from control_module import KeyboardControl

# Define Palette
BG_COLOR = (15, 15, 20)
PANEL_COLOR = (24, 24, 30)
BORDER_COLOR = (45, 45, 55)
TEXT_COLOR = (230, 230, 240)
ACCENT_BLUE = (64, 140, 240)
ACCENT_GREEN = (50, 200, 100)
ACCENT_YELLOW = (255, 200, 50)
ACCENT_RED = (240, 70, 70)

def draw_text(surface, text, font, color, x, y, align_right=False):
    text_surf = font.render(text, True, color)
    rect = text_surf.get_rect()
    if align_right:
        rect.topright = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(text_surf, rect)

def draw_bar(surface, x, y, w, h, val, max_val, color):
    pygame.draw.rect(surface, (45, 45, 50), (x, y, w, h))
    fill_w = int(w * (min(val, max_val) / max_val))
    pygame.draw.rect(surface, color, (x, y, fill_w, h))
    pygame.draw.rect(surface, (80, 80, 90), (x, y, w, h), 1)

def main():
    # 1. Initialize Pygame Window
    pygame.init()
    pygame.font.init()
    
    WINDOW_W = 1440
    WINDOW_H = 800
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Fog Vision Hazard Prevention & ACC Dashboard")
    
    clock = pygame.time.Clock()
    
    # Fonts
    font_large = pygame.font.SysFont("Outfit", 24, bold=True)
    font_medium = pygame.font.SysFont("Inter", 18, bold=True)
    font_small = pygame.font.SysFont("Inter", 14)
    font_btn = pygame.font.SysFont("Outfit", 14, bold=True)

    # 2. Initialize System Modules
    print("[Dashboard] Initializing modular dashboard components...")
    dehaze_mod = DehazeModule(model_name='dehazeformer-t', resolution=320, device='auto')
    esp32_comm = ESP32Communication(esp32_ip="192.168.4.1", esp32_port=5005, local_port=5006)
    control_mod = KeyboardControl(max_speed=200, max_steering=45)
    sensor_proc = SensorProcessing(ema_alpha=0.18, max_valid_range=300.0)
    acc_system = AdaptiveCruiseControl(target_distance=50.0, critical_distance=25.0, Kp=4.5, Kd=90.0)
    twin_sim = DigitalTwin(width=580, height=720)
    
    # Start background threads
    dehaze_mod.start()
    esp32_comm.start()
    
    # Setup interactive configuration buttons (Left Panel bottom)
    btn_cam0 = pygame.Rect(30, 740, 70, 26)
    btn_cam1 = pygame.Rect(110, 740, 70, 26)
    
    btn_mod_t = pygame.Rect(210, 740, 35, 26)
    btn_mod_s = pygame.Rect(250, 740, 35, 26)
    btn_mod_m = pygame.Rect(290, 740, 35, 26)
    btn_mod_b = pygame.Rect(330, 740, 35, 26)
    
    last_time = time.time()
    
    running = True
    while running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        if dt <= 0:
            dt = 0.001
            
        # 3. Read Events
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                
            # Handle interactive button clicks
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    mx, my = event.pos
                    
                    # Webcam click handler
                    if btn_cam0.collidepoint(mx, my):
                        dehaze_mod.set_camera(0)
                    elif btn_cam1.collidepoint(mx, my):
                        dehaze_mod.set_camera(1)
                        
                    # Model click handler
                    elif btn_mod_t.collidepoint(mx, my):
                        dehaze_mod.set_model("dehazeformer-t")
                    elif btn_mod_s.collidepoint(mx, my):
                        dehaze_mod.set_model("dehazeformer-s")
                    elif btn_mod_m.collidepoint(mx, my):
                        dehaze_mod.set_model("dehazeformer-m")
                    elif btn_mod_b.collidepoint(mx, my):
                        dehaze_mod.set_model("dehazeformer-b")
                        
        # Handle dragging of simulated vehicle in digital twin
        mouse_pos = pygame.mouse.get_pos()
        # Scale mouse position relative to twin screen coordinates: (430 <= x <= 1010), (50 <= y <= 770)
        twin_mx = mouse_pos[0] - 430
        twin_my = mouse_pos[1] - 50
        mouse_pressed = pygame.mouse.get_pressed()
        twin_sim.handle_mouse((twin_mx, twin_my), mouse_pressed)

        # 4. Update Inputs (Keyboard WASD)
        target_throttle, steering_angle, acc_enabled = control_mod.update(events, dt)

        # 5. Retrieve physical or simulated telemetry
        connected, l_dist, c_dist, r_dist, real_speed = esp32_comm.get_telemetry()
        
        # 6. Apply filter and compute relative velocity
        filtered_distances, relative_velocity = sensor_proc.process(l_dist, c_dist, r_dist)
        
        # 7. Apply ACC Speed driver
        final_throttle, acc_status_text = acc_system.compute_control(
            acc_enabled, target_throttle, filtered_distances, relative_velocity
        )

        # 8. Send control commands to car
        esp32_comm.send_control(final_throttle, steering_angle)

        # 9. Update digital twin kinematic physics
        sim_readings = twin_sim.update(final_throttle, steering_angle, connected, real_speed, dt)
        if not connected:
            esp32_comm.update_simulated_telemetry(
                sim_readings[0], sim_readings[1], sim_readings[2], 
                float(abs(twin_sim.speed) / 10.0)
            )

        # 10. Rendering Dashboard Panels
        screen.fill(BG_COLOR)
        
        # --- HEADER PANEL ---
        pygame.draw.rect(screen, PANEL_COLOR, (10, 10, WINDOW_W - 20, 35))
        pygame.draw.rect(screen, BORDER_COLOR, (10, 10, WINDOW_W - 20, 35), 1)
        draw_text(screen, "FOG VISION HAZARD PREVENTION & ADAPTIVE CRUISE CONTROL", font_large, ACCENT_BLUE, 20, 15)
        
        # --- LEFT PANEL: CAMERAS ---
        pygame.draw.rect(screen, PANEL_COLOR, (10, 50, 410, 740))
        pygame.draw.rect(screen, BORDER_COLOR, (10, 50, 410, 740), 1)
        
        # Original stream (height scaled to 390x270 for spacing)
        draw_text(screen, "ORIGINAL CAMERA FEED (FOGGY / HAZY)", font_medium, TEXT_COLOR, 20, 58)
        orig_img, dehazed_img, cam_fps, cam_latency = dehaze_mod.get_frames()
        if orig_img is not None:
            orig_resized = cv2.resize(orig_img, (390, 270))
            orig_surf = pygame.image.frombuffer(orig_resized.tobytes(), (390, 270), 'RGB')
            screen.blit(orig_surf, (20, 85))
            status_txt = "VIDEO FALLBACK" if dehaze_mod.is_fallback else "WEBCAM IN"
            status_color = ACCENT_YELLOW if dehaze_mod.is_fallback else ACCENT_GREEN
            draw_text(screen, status_txt, font_small, status_color, 20, 360)
        else:
            pygame.draw.rect(screen, (35, 35, 45), (20, 85, 390, 270))
            draw_text(screen, "Connecting to Video/Webcam...", font_medium, ACCENT_YELLOW, 90, 200)

        # Dehazed stream (height scaled to 390x270)
        draw_text(screen, "REAL-TIME DEHAZED OUTPUT", font_medium, ACCENT_BLUE, 20, 395)
        if dehazed_img is not None:
            dehazed_resized = cv2.resize(dehazed_img, (390, 270))
            dehazed_surf = pygame.image.frombuffer(dehazed_resized.tobytes(), (390, 270), 'RGB')
            screen.blit(dehazed_surf, (20, 420))
            draw_text(screen, f"Inference Latency: {cam_latency:.1f} ms | FPS: {cam_fps:.1f}", font_small, TEXT_COLOR, 20, 695)
        else:
            pygame.draw.rect(screen, (35, 35, 45), (20, 420, 390, 270))
            draw_text(screen, "Initializing Dehazing Network...", font_medium, ACCENT_YELLOW, 80, 530)

        # --- DYNAMIC CONFIG BUTTONS (Left Panel Bottom) ---
        pygame.draw.line(screen, BORDER_COLOR, (10, 715), (420, 715), 1)
        
        # Camera Buttons
        draw_text(screen, "CAM", font_small, TEXT_COLOR, 20, 720)
        cam_active = dehaze_mod.camera_index
        pygame.draw.rect(screen, ACCENT_BLUE if cam_active == 0 else (45, 45, 55), btn_cam0)
        draw_text(screen, "FRONT", font_btn, (255, 255, 255), 45, 745)
        pygame.draw.rect(screen, ACCENT_BLUE if cam_active == 1 else (45, 45, 55), btn_cam1)
        draw_text(screen, "BACK", font_btn, (255, 255, 255), 130, 745)

        # Model Buttons
        draw_text(screen, "DEHAZE MODEL", font_small, TEXT_COLOR, 210, 720)
        mod_active = dehaze_mod.model_name
        pygame.draw.rect(screen, ACCENT_BLUE if mod_active == "dehazeformer-t" else (45, 45, 55), btn_mod_t)
        draw_text(screen, "T", font_btn, (255, 255, 255), 223, 745)
        pygame.draw.rect(screen, ACCENT_BLUE if mod_active == "dehazeformer-s" else (45, 45, 55), btn_mod_s)
        draw_text(screen, "S", font_btn, (255, 255, 255), 263, 745)
        pygame.draw.rect(screen, ACCENT_BLUE if mod_active == "dehazeformer-m" else (45, 45, 55), btn_mod_m)
        draw_text(screen, "M", font_btn, (255, 255, 255), 303, 745)
        pygame.draw.rect(screen, ACCENT_BLUE if mod_active == "dehazeformer-b" else (45, 45, 55), btn_mod_b)
        draw_text(screen, "B", font_btn, (255, 255, 255), 343, 745)

        # --- CENTER PANEL: DIGITAL TWIN ---
        twin_surf = pygame.Surface((580, 740))
        twin_sim.render(twin_surf)
        screen.blit(twin_surf, (430, 50))
        
        # Legend overlay
        legend_y = 60
        pygame.draw.rect(screen, (20, 20, 25, 200), (440, legend_y, 200, 95))
        pygame.draw.rect(screen, BORDER_COLOR, (440, legend_y, 200, 95), 1)
        draw_text(screen, "Digital Twin Simulation:", font_small, TEXT_COLOR, 450, legend_y + 5)
        pygame.draw.circle(screen, (200, 50, 50), (455, legend_y + 30), 6)
        draw_text(screen, "Leading Vehicle (Drag me)", font_small, TEXT_COLOR, 470, legend_y + 22)
        pygame.draw.rect(screen, (60, 140, 240), (449, legend_y + 49, 12, 12))
        draw_text(screen, "Prototype Ego Car", font_small, TEXT_COLOR, 470, legend_y + 44)
        pygame.draw.line(screen, ACCENT_GREEN, (449, legend_y + 75), (461, legend_y + 75), 2)
        draw_text(screen, "VL53L0X Distance Rays", font_small, TEXT_COLOR, 470, legend_y + 67)

        # --- RIGHT PANEL: TELEMETRY & CONTROLS ---
        pygame.draw.rect(screen, PANEL_COLOR, (1020, 50, 410, 740))
        pygame.draw.rect(screen, BORDER_COLOR, (1020, 50, 410, 740), 1)
        
        panel_x = 1035
        # Connection Status
        draw_text(screen, "HARDWARE CONNECTION STATUS", font_medium, TEXT_COLOR, panel_x, 65)
        conn_text = "CONNECTED TO ESP32 (192.168.4.1)" if connected else "DISCONNECTED (SIMULATED TWIN)"
        conn_color = ACCENT_GREEN if connected else ACCENT_RED
        draw_text(screen, conn_text, font_large, conn_color, panel_x, 85)
        
        pygame.draw.line(screen, BORDER_COLOR, (1020, 125), (1430, 125), 1)

        # Driver Mode
        draw_text(screen, "ACC / VEHICLE SAFETY DRIVER", font_medium, TEXT_COLOR, panel_x, 140)
        mode_color = ACCENT_BLUE
        if "OVERRIDE" in acc_status_text or "CRITICAL" in acc_status_text:
            mode_color = ACCENT_RED
        elif "ACC ACTIVE" in acc_status_text:
            mode_color = ACCENT_GREEN
        elif "MANUAL" in acc_status_text:
            mode_color = ACCENT_YELLOW
        draw_text(screen, acc_status_text, font_large, mode_color, panel_x, 160)
        
        # Speed meters
        draw_text(screen, f"Vehicle Speed: {real_speed:.1f} m/s", font_medium, TEXT_COLOR, panel_x, 210)
        draw_bar(screen, panel_x, 230, 380, 15, abs(real_speed), 3.0, ACCENT_BLUE)
        
        # Steering indicator
        draw_text(screen, f"Steering Output: {steering_angle}°", font_medium, TEXT_COLOR, panel_x, 255)
        steer_bar_val = steering_angle + 45
        draw_bar(screen, panel_x, 275, 380, 15, steer_bar_val, 90.0, ACCENT_YELLOW)

        pygame.draw.line(screen, BORDER_COLOR, (1020, 310), (1430, 310), 1)

        # Distance sensors (3 sensors)
        draw_text(screen, "VL53L0X DISTANCE SENSORS (cm)", font_medium, TEXT_COLOR, panel_x, 325)
        
        sensor_names = ["Left Sensor (L)", "Center Sensor (C)", "Right Sensor (R)"]
        for idx in range(3):
            dist_val = filtered_distances[idx]
            dist_y = 355 + idx * 55
            
            draw_text(screen, f"{sensor_names[idx]}:", font_small, TEXT_COLOR, panel_x, dist_y)
            draw_text(screen, f"{dist_val:.1f} cm", font_small, TEXT_COLOR, panel_x + 380, dist_y, align_right=True)
            
            bar_color = ACCENT_GREEN
            if dist_val < 25.0:
                bar_color = ACCENT_RED
            elif dist_val < 50.0:
                bar_color = ACCENT_YELLOW
                
            draw_bar(screen, panel_x, dist_y + 18, 380, 12, dist_val, 300.0, bar_color)

        pygame.draw.line(screen, BORDER_COLOR, (1020, 535), (1430, 535), 1)

        # Relative Velocity
        draw_text(screen, "HAZARD DETECTION TELEMETRY", font_medium, TEXT_COLOR, panel_x, 550)
        
        vel_text_color = TEXT_COLOR
        vel_prefix = ""
        if relative_velocity < -0.05:
            vel_text_color = ACCENT_RED
            vel_prefix = "APPROACHING: "
        elif relative_velocity > 0.05:
            vel_text_color = ACCENT_GREEN
            vel_prefix = "RECEDING: "
        draw_text(screen, f"Relative Velocity: {vel_prefix}{relative_velocity:+.2f} m/s", font_medium, vel_text_color, panel_x, 570)
        
        # Time-to-Collision (TTC)
        center_d_cm = filtered_distances[1]
        if relative_velocity < -0.01:
            ttc = (center_d_cm / 100.0) / abs(relative_velocity)
            ttc_text = f"Time-to-Collision (TTC): {ttc:.2f} s"
            ttc_color = ACCENT_RED if ttc < 1.5 else ACCENT_YELLOW
        else:
            ttc_text = "Time-to-Collision (TTC): N/A"
            ttc_color = TEXT_COLOR
        draw_text(screen, ttc_text, font_medium, ttc_color, panel_x, 595)

        pygame.draw.line(screen, BORDER_COLOR, (1020, 635), (1430, 635), 1)

        # Help / Key list
        help_y = 650
        draw_text(screen, "KEYBOARD CONTROLS:", font_small, ACCENT_BLUE, panel_x, help_y)
        draw_text(screen, "W/S/A/D (or Arrows) = Drive / Steer | SPACE = Active Brake", font_small, TEXT_COLOR, panel_x, help_y + 18)
        draw_text(screen, "M = Toggle Manual vs Auto ACC Mode", font_small, TEXT_COLOR, panel_x, help_y + 36)
        draw_text(screen, "MOUSE = Click & Drag red vehicle in twin to test ACC", font_small, TEXT_COLOR, panel_x, help_y + 54)
        draw_text(screen, "PANEL BUTTONS = Click CAM/MODEL buttons to configure feeds", font_small, TEXT_COLOR, panel_x, help_y + 72)

        pygame.display.flip()
        clock.tick(30)

    # Cleanup
    print("[Dashboard] Closing system modules...")
    dehaze_mod.stop()
    esp32_comm.stop()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
