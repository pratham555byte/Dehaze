/*
  Fog Vision Hazard Prevention System - ESP32 Vehicle Firmware (3-Sensor Version)
  
  Pins configuration for L298N Motor Driver and 3x VL53L0X Sensors.
  Utilizes standard I2C bus with XSHUT pins to configure unique addresses.
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_VL53L0X.h>

// --- WIFI CONFIGURATION ---
const char* ssid = "ESP32_Car";
const char* password = "12345678";
const unsigned int localPort = 5005; // port to receive commands
const unsigned int remotePort = 5006; // port to send telemetry
IPAddress remoteIP(192.168.4.2); // default laptop IP in AP mode

WiFiUDP Udp;
char packetBuffer[255]; // buffer to hold incoming packet

// --- MOTOR CONTROLLER PINS (L298N) ---
// Differential Drive configuration
const int ENA = 15; // PWM Speed pin Left
const int IN1 = 2;  // Dir 1 Left
const int IN2 = 4;  // Dir 2 Left

const int ENB = 18; // PWM Speed pin Right
const int IN3 = 19; // Dir 1 Right
const int IN4 = 21; // Dir 2 Right

// --- SENSOR SHUTDOWN (XSHUT) PINS ---
// Used to boot sensors one by one and assign distinct I2C addresses
const int SENSOR_L_SHUT  = 13;
const int SENSOR_C_SHUT  = 12;
const int SENSOR_R_SHUT  = 14;

// Unique I2C addresses assigned dynamically during boot
const int ADDR_L  = 0x30;
const int ADDR_C  = 0x31;
const int ADDR_R  = 0x32;

// Sensor Driver Instances
Adafruit_VL53L0X sensorL  = Adafruit_VL53L0X();
Adafruit_VL53L0X sensorC  = Adafruit_VL53L0X();
Adafruit_VL53L0X sensorR  = Adafruit_VL53L0X();

// Vehicle Telemetry Variables
float currentSpeed = 0.0; // calculated or mocked speed (m/s)

void setup() {
  Serial.begin(115200);
  
  // 1. Initialize Motor Driver Pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  
  // Set motors to idle
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  // 2. Setup I2C VL53L0X Address Allocations
  pinMode(SENSOR_L_SHUT, OUTPUT);
  pinMode(SENSOR_C_SHUT, OUTPUT);
  pinMode(SENSOR_R_SHUT, OUTPUT);

  // Shutdown all sensors
  digitalWrite(SENSOR_L_SHUT, LOW);
  digitalWrite(SENSOR_C_SHUT, LOW);
  digitalWrite(SENSOR_R_SHUT, LOW);
  delay(10);

  Wire.begin(); // Join I2C Bus as Master

  // Initialize sensors one by one
  initSensor(SENSOR_L_SHUT, sensorL, ADDR_L, "Left (L)");
  initSensor(SENSOR_C_SHUT, sensorC, ADDR_C, "Center (C)");
  initSensor(SENSOR_R_SHUT, sensorR, ADDR_R, "Right (R)");

  // 3. Initialize WiFi Access Point
  Serial.println("\nStarting Access Point...");
  WiFi.softAP(ssid, password);
  IPAddress myIP = WiFi.softAPIP();
  Serial.print("AP IP Address: ");
  Serial.println(myIP);

  // Start UDP Listening
  Udp.begin(localPort);
  Serial.print("Listening for UDP commands on port ");
  Serial.println(localPort);
}

void initSensor(int shutPin, Adafruit_VL53L0X &sensor, int address, const char* name) {
  // Bring the specific sensor out of shutdown state
  digitalWrite(shutPin, HIGH);
  delay(10);
  
  Serial.print("Initializing sensor: ");
  Serial.print(name);
  Serial.print(" at address 0x");
  Serial.println(address, HEX);

  // Attempt to initialize and change standard 0x29 address to the custom address
  if (!sensor.begin(address, false, &Wire)) {
    Serial.print("Failed to initialize ");
    Serial.println(name);
    // Continue anyway to prevent entire car boot failure if one sensor is disconnected
  }
}

void loop() {
  // 1. Process incoming UDP control command from laptop
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    // Record sender IP to reply to
    remoteIP = Udp.remoteIP();
    
    int len = Udp.read(packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = 0;
    }
    
    String commandStr = String(packetBuffer);
    // Protocol parses: "CMD,throttle,steering"
    if (commandStr.startsWith("CMD,")) {
      int comma1 = commandStr.indexOf(',');
      int comma2 = commandStr.indexOf(',', comma1 + 1);
      
      if (comma1 != -1 && comma2 != -1) {
        int throttle = commandStr.substring(comma1 + 1, comma2).toInt();
        int steering = commandStr.substring(comma2 + 1).toInt();
        
        driveMotors(throttle, steering);
        
        // Mock speedometer based on throttle command
        currentSpeed = abs(throttle) * (2.2 / 255.0); // max speed ~2.2 m/s
      }
    }
  }

  // 2. Read sensor distance ranges (in cm, convert from mm)
  float dist_l = getDistance(sensorL);
  float dist_c = getDistance(sensorC);
  float dist_r = getDistance(sensorR);

  // 3. Send Telemetry back to Laptop via UDP
  Udp.beginPacket(remoteIP, remotePort);
  // Protocol replies: "TELE,l_dist,c_dist,r_dist,speed"
  Udp.print("TELE,");
  Udp.print(dist_l); Udp.print(",");
  Udp.print(dist_c); Udp.print(",");
  Udp.print(dist_r); Udp.print(",");
  Udp.print(currentSpeed);
  Udp.endPacket();

  delay(20); // main control loop cycle delay
}

float getDistance(Adafruit_VL53L0X &sensor) {
  VL53L0X_RangingMeasurementData_t measure;
  sensor.rangingTest(&measure, false);
  
  // Phase status 4 indicates valid measurement
  if (measure.RangeStatus != 4) {
    return (float)measure.RangeMilliMeter / 10.0; // convert mm to cm
  } else {
    return 300.0; // default to maximum sensor range if invalid reading / out of range
  }
}

void driveMotors(int throttle, int steering) {
  // L298N Differential Drive Control
  // Steering adds/subtracts from throttle for left/right motors
  int leftSpeed = throttle + steering;
  int rightSpeed = throttle - steering;
  
  // Constrain speeds to PWM range
  leftSpeed = constrain(leftSpeed, -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);
  
  // Set Left Motor
  if (leftSpeed >= 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, leftSpeed);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(ENA, -leftSpeed);
  }
  
  // Set Right Motor
  if (rightSpeed >= 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    analogWrite(ENB, rightSpeed);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    analogWrite(ENB, -rightSpeed);
  }
}
