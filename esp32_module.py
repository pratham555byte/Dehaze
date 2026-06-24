import socket
import threading
import time
import requests

class ESP32Communication:
    def __init__(self, esp32_ip="192.168.4.1", esp32_port=5005, local_port=5006, mode="udp"):
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.local_port = local_port
        self.mode = mode # "udp" or "http"
        
        self.sock = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Telemetry state
        self.connected = False
        self.last_packet_time = 0.0
        
        # Distances from the 3 sensors (in cm)
        self.dist_l = 300.0
        self.dist_c = 300.0
        self.dist_r = 300.0
        self.vehicle_speed = 0.0
        
        # HTTP Control State
        self.http_throttle = 0
        self.http_steering = 0
        self.http_control_updated = False
        
        # Simulation values (written to by the digital twin when disconnected)
        self.sim_dist_l = 300.0
        self.sim_dist_c = 300.0
        self.sim_dist_r = 300.0
        self.sim_speed = 0.0

    def start(self):
        self.running = True
        
        if self.mode == "udp":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.1)
            try:
                self.sock.bind(("", self.local_port))
                print(f"[ESP32Module] UDP Socket bound to local port {self.local_port}")
            except Exception as e:
                print(f"[ESP32Module] UDP Bind warning: {e}")
                
            self.thread = threading.Thread(target=self._receive_loop_udp, daemon=True)
            self.thread.start()
        else:
            print(f"[ESP32Module] HTTP Mode started for ESP32 IP: {self.esp32_ip}")
            self.thread = threading.Thread(target=self._receive_loop_http, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

    def _receive_loop_udp(self):
        buffer_size = 1024
        while self.running:
            try:
                data, addr = self.sock.recvfrom(buffer_size)
                msg = data.decode("utf-8").strip()
                
                # Expecting format: "TELE,dist_l,dist_c,dist_r,speed"
                if msg.startswith("TELE,"):
                    parts = msg.split(",")
                    if len(parts) >= 5:
                        with self.lock:
                            self.dist_l = float(parts[1])
                            self.dist_c = float(parts[2])
                            self.dist_r = float(parts[3])
                            self.vehicle_speed = float(parts[4])
                            self.last_packet_time = time.time()
                            self.connected = True
            except socket.timeout:
                pass
            except Exception as e:
                pass
            
            # Watchdog: mark disconnected if no packets for 2 seconds
            if time.time() - self.last_packet_time > 2.0:
                with self.lock:
                    self.connected = False
            
            time.sleep(0.01)

    def _receive_loop_http(self):
        while self.running:
            # 1. Poll Sensors
            url = f"http://{self.esp32_ip}/sensors"
            try:
                resp = requests.get(url, timeout=0.1)
                if resp.status_code == 200:
                    data = resp.json()
                    with self.lock:
                        # Convert mm from ESP32 to cm for simulator compatibility
                        self.dist_l = float(data.get("left", 8000.0)) / 10.0
                        self.dist_c = float(data.get("center", 8000.0)) / 10.0
                        self.dist_r = float(data.get("right", 8000.0)) / 10.0
                        # Speed estimation based on throttle
                        self.vehicle_speed = float(abs(self.http_throttle)) / 10.0
                        self.last_packet_time = time.time()
                        self.connected = True
            except Exception:
                # Connection timeout or fail
                pass
            
            # 2. Send Controls if updated
            with self.lock:
                control_updated = self.http_control_updated
                throttle = self.http_throttle
                steering = self.http_steering
                
            if control_updated:
                url_motor = f"http://{self.esp32_ip}/motor"
                try:
                    # Payload: {"speed": speed, "turn": turn} (Range: -100 to 100)
                    requests.post(url_motor, json={"speed": int(throttle), "turn": int(steering)}, timeout=0.1)
                    with self.lock:
                        self.http_control_updated = False
                except Exception:
                    pass

            # Watchdog: mark disconnected if no updates for 2 seconds
            if time.time() - self.last_packet_time > 2.0:
                with self.lock:
                    self.connected = False
            
            time.sleep(0.05)

    def send_control(self, throttle, steering):
        if self.mode == "udp":
            if not self.sock:
                return
            msg = f"CMD,{int(throttle)},{int(steering)}"
            try:
                self.sock.sendto(msg.encode("utf-8"), (self.esp32_ip, self.esp32_port))
            except Exception as e:
                pass
        else:
            with self.lock:
                self.http_throttle = throttle
                self.http_steering = steering
                self.http_control_updated = True

    def update_simulated_telemetry(self, l, c, r, speed):
        with self.lock:
            self.sim_dist_l = l
            self.sim_dist_c = c
            self.sim_dist_r = r
            self.sim_speed = speed

    def get_telemetry(self):
        with self.lock:
            if self.connected:
                return True, self.dist_l, self.dist_c, self.dist_r, self.vehicle_speed
            else:
                return False, self.sim_dist_l, self.sim_dist_c, self.sim_dist_r, self.sim_speed
