#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "VisionDrive_Car";
const char* password = "password123";

WiFiUDP udp;
unsigned int localPort = 4210;
char packetBuffer[255];

// --- Motor Pins ---
const int IN1 = 26; const int IN2 = 25; // Right
const int IN3 = 14; const int IN4 = 27; // Left

// NEW SAFE PINS (CSA = 13, CSB = 33)
const int ENA = 32; 
const int ENB = 33; 

const int freq = 5000;
const int resolution = 8;
const int speedA = 210; 
const int speedB = 160; 

void setup() {
  Serial.begin(115200);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  ledcAttach(ENA, freq, resolution);
  ledcAttach(ENB, freq, resolution);
  moveForward();
  delay(1000); // Test both motors for 1 second
  stopCar();

  WiFi.softAP(ssid, password);
  udp.begin(localPort);
  stopCar();
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, 255);
    if (len > 0) packetBuffer[len] = 0;
    String cmd = String(packetBuffer);
    cmd.trim();

    if (cmd == "FORWARD") moveForward();
    else if (cmd == "BACK") moveBackward();
    else if (cmd == "FORWARD_LEFT") turnForwardLeft();
    else if (cmd == "FORWARD_RIGHT") turnForwardRight();
    else stopCar();
  }
}

void moveForward() {
  stopCar(); // Momentary clear
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speedA);  ledcWrite(ENB, speedB);
}

void moveBackward() {
  stopCar(); // Momentary clear
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  ledcWrite(ENA, speedA);  ledcWrite(ENB, speedB);
}

void turnForwardLeft() {
  stopCar();
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  ledcWrite(ENA, speedA);  ledcWrite(ENB, 0);
}

void turnForwardRight() {
  stopCar();
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, 0);       ledcWrite(ENB, speedB);
}

void stopCar() {
  ledcWrite(ENA, 0); ledcWrite(ENB, 0);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}