import time
import json
import math
import random
import socket
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Virtual Corridor State ---
# Vehicle position (meters)
x_pos = 0.0      # -1.5 to +1.5 (center of 3m wide lane)
y_pos = 0.0      # longitudinal position
speed = 0.0      # m/s
steering = 0.0   # steering angle in degrees

# Motor input commands
throttle_cmd = 0.0
steering_cmd = 0.0

# Obstacles list is cleared for free vehicle movement
obstacles = []

last_update = time.perf_counter()
state_lock = threading.Lock()

# --- Ray-Boundary and Ray-Circle Intersections ---
def get_sensor_distances():
    global x_pos, y_pos, speed, steering, throttle_cmd, steering_cmd, last_update
    
    with state_lock:
        now = time.perf_counter()
        dt = now - last_update
        last_update = now
        
        # 1. Update vehicle kinetics
        # Max speed 2.2 m/s based on ESP32 firmware constraints
        target_speed = (throttle_cmd / 200.0) * 2.2
        speed += (target_speed - speed) * 4.0 * dt
        
        steering += (steering_cmd - steering) * 5.0 * dt
        
        # Translate to coordinates
        steering_rad = math.radians(steering)
        y_pos += speed * math.cos(steering_rad) * dt
        x_pos += speed * math.sin(steering_rad) * dt
        
        # Constrain vehicle within corridor boundaries (-1.5m to +1.5m)
        if x_pos < -1.3:
            x_pos = -1.3
        elif x_pos > 1.3:
            x_pos = 1.3
            
        cur_x = x_pos
        cur_y = y_pos
        cur_steer = steering_rad
        cur_speed = speed

    # 2. Raycast from vehicle bumper
    # Ray angles in vehicle local frame: Left (-20 deg), Center (0 deg), Right (+20 deg)
    sensor_angles = [math.radians(-20), math.radians(0), math.radians(20)]
    max_range = 8.0 # 8 meters max sensor range
    readings_mm = []

    for angle_offset in sensor_angles:
        ray_angle = cur_steer + angle_offset
        dx = math.sin(ray_angle)
        dy = math.cos(ray_angle)
        
        closest_t = max_range
        
        # Intersection with Left Wall (x = -1.5)
        if dx < 0:
            t_wall = (-1.5 - cur_x) / dx
            if 0 <= t_wall < closest_t:
                closest_t = t_wall
        # Intersection with Right Wall (x = 1.5)
        elif dx > 0:
            t_wall = (1.5 - cur_x) / dx
            if 0 <= t_wall < closest_t:
                closest_t = t_wall
                
        # Intersection with Obstacles
        for obs in obstacles:
            # Only consider obstacles ahead
            if obs["y"] > cur_y - 1.0:
                ocx = obs["x"] - cur_x
                ocy = obs["y"] - cur_y
                # Project onto ray
                projection = ocx * dx + ocy * dy
                if projection >= 0:
                    closest_x = cur_x + dx * projection
                    closest_y = cur_y + dy * projection
                    dist_sq = (closest_x - obs["x"])**2 + (closest_y - obs["y"])**2
                    r_sq = obs["r"]**2
                    
                    if dist_sq <= r_sq:
                        half_chord = math.sqrt(r_sq - dist_sq)
                        t_obs = projection - half_chord
                        if 0 <= t_obs < closest_t:
                            closest_t = t_obs

        # Convert to mm and add small noise to simulate hardware sensor jitter
        noise_m = random.gauss(0, 0.015) # 1.5 cm noise
        dist_m = max(0.1, closest_t + noise_m)
        readings_mm.append(int(dist_m * 1000.0))

    return readings_mm, cur_speed

# --- HTTP Endpoints ---
@app.route('/sensors', methods=['GET'])
@app.route('/sensor', methods=['GET'])
def get_sensors():
    readings, _ = get_sensor_distances()
    return jsonify({
        "left": readings[0],
        "center": readings[1],
        "right": readings[2]
    })

@app.route('/motor', methods=['POST'])
def post_motor():
    global throttle_cmd, steering_cmd
    try:
        data = request.get_json(force=True)
        with state_lock:
            # Payloads range -200 to 200 (speed mapping in esp32_module)
            throttle_cmd = float(data.get("speed", 0))
            steering_cmd = float(data.get("turn", 0))
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# --- UDP Server Loop ---
# Handles incoming control CMD packages and returns telemetry packets
def udp_server_loop():
    local_port = 5005
    remote_port = 5006
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", local_port))
    sock.settimeout(0.5)
    
    print(f"[ESP32Mock] UDP Server listening on port {local_port}...")
    
    global throttle_cmd, steering_cmd
    
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            msg = data.decode("utf-8").strip()
            
            # Protocol parses: "CMD,throttle,steering"
            if msg.startswith("CMD,"):
                parts = msg.split(",")
                if len(parts) >= 3:
                    with state_lock:
                        throttle_cmd = float(parts[1])
                        steering_cmd = float(parts[2])
                        
                    # Calculate telemetry (in cm) to send back
                    readings_mm, current_speed = get_sensor_distances()
                    dist_l = readings_mm[0] / 10.0
                    dist_c = readings_mm[1] / 10.0
                    dist_r = readings_mm[2] / 10.0
                    
                    # Protocol replies: "TELE,l_dist,c_dist,r_dist,speed"
                    reply = f"TELE,{dist_l:.2f},{dist_c:.2f},{dist_r:.2f},{current_speed:.3f}"
                    sock.sendto(reply.encode("utf-8"), (addr[0], remote_port))
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[ESP32Mock] UDP error: {e}")
            time.sleep(0.1)

if __name__ == '__main__':
    # Launch UDP thread
    udp_thread = threading.Thread(target=udp_server_loop, daemon=True)
    udp_thread.start()
    
    # Launch HTTP server
    print("[ESP32Mock] HTTP Server starting on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
