#include <WiFi.h>
#include <WiFiUdp.h>

// --- Hotspot Credentials ---
const char* ssid = "vivo 1915";
const char* password = "RijalNiraj";

WiFiUDP udp;
unsigned int localPort = 4210;
char packetBuffer[255];

// --- Motor Pins ---
const int IN1 = 26; const int IN2 = 25; // Right Motor
const int IN3 = 13; const int IN4 = 27; // Left Motor
const int ENA = 5; 
const int ENB = 18; 
const int SOS_LED = 2; // Internal Blue LED

// --- Speed Settings ---
const int freq = 5000;
const int resolution = 8;
const int speedA = 210;  // Increased significantly for reliable starting (0-255)
const int speedB = 200;  // Same for both motors during testing (adjust later if needed)

void setup() {
  Serial.begin(115200);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(SOS_LED, OUTPUT); 

  // Connect to Hotspot FIRST
  Serial.print("Connecting to: "); Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nCONNECTED!");
  Serial.print("IP Address for Python Code: ");
  Serial.println(WiFi.localIP());

  // --- NOW attach PWM (after WiFi to avoid timer conflicts) ---
  // Using the legacy ledcAttach API (compatible with older ESP32 Arduino cores)
  if (!ledcAttach(ENA, freq, resolution)) {
    Serial.println("ERROR: Failed to attach PWM to ENA (Right motor)");
  } else {
    Serial.println("PWM attached to ENA (Right motor) successfully");
  }

  if (!ledcAttach(ENB, freq, resolution)) {
    Serial.println("ERROR: Failed to attach PWM to ENB (Left motor)");
  } else {
    Serial.println("PWM attached to ENB (Left motor) successfully");
  }

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

    Serial.print("CMD: "); Serial.println(cmd);

    if (cmd == "SOS") triggerSOS(); 
    else if (cmd == "FORWARD") moveForward();
    else if (cmd == "BACK") moveBackward();
    else if (cmd == "FORWARD_LEFT") turnForwardLeft();
    else if (cmd == "FORWARD_RIGHT") turnForwardRight();
    else stopCar();
  }
}

void triggerSOS() {
  Serial.println("!!! SOS ALARM !!!");
  stopCar(); 
  for(int i = 0; i < 20; i++) {
    digitalWrite(SOS_LED, !digitalRead(SOS_LED));
    delay(60);
  }
  digitalWrite(SOS_LED, LOW);
}

void moveForward() {
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);  // Right forward
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);  // Left forward
  ledcWrite(ENA, speedA);
  ledcWrite(ENB, speedB);
}

void moveBackward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);   // Right backward
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);   // Left backward
  ledcWrite(ENA, speedA);
  ledcWrite(ENB, speedB);
}

void turnForwardLeft() {
  // Right motor forward, left motor coast-stop
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);  // Coast (both LOW)
  ledcWrite(ENA, speedA);
  ledcWrite(ENB, 0);
}

void turnForwardRight() {
  // Left motor forward, right motor coast-stop
  digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);   // Coast (both LOW)
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, 0);
  ledcWrite(ENB, speedB);
}

void stopCar() {
  ledcWrite(ENA, 0);
  ledcWrite(ENB, 0);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}