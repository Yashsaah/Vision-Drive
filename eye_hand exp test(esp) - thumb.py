import cv2
from gaze_tracking import GazeTracking
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
import socket  # Replaced serial with socket
import time

# ==================== WIFI CONNECTION (UDP) ====================
ESP32_IP = "192.168.4.1"  # Default IP when ESP32 is in Access Point mode     
PORT = 4210               # Must match the port in your ESP32 code

print(f"Targeting ESP32 at {ESP32_IP}...")
try:
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("SUCCESS: UDP Socket created. ready to send wireless commands.")
except Exception as e:    
    print(f"FATAL ERROR: Could not create socket. Details: {e}")
    sys.exit()
 

# ==================== AI INITIALIZATION ====================
gaze = GazeTracking()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

with open(MODEL_PATH, "rb") as f:
    model_data = f.read()

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_buffer=model_data),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
webcam = cv2.VideoCapture(0)
last_command = "STOP"

while True:
    _, frame = webcam.read()
    if frame is None: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Gaze Tracking
    gaze.refresh(frame_rgb)
    frame_display = cv2.cvtColor(gaze.annotated_frame(), cv2.COLOR_RGB2BGR)

    if gaze.is_blinking(): gaze_state = "BLINKING"
    elif gaze.is_right(): gaze_state = "RIGHT"
    elif gaze.is_left():  gaze_state = "LEFT"
    else:                 gaze_state = "CENTER"

    # 2. Hand Tracking
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    hand_gear = "NEUTRAL"
    if hand_result.hand_landmarks:
        landmarks = hand_result.hand_landmarks[0]
        
        # Check Index (8) and Middle (12) against Knuckle (5)
        index_raised = landmarks[8].y < landmarks[5].y - 0.12
        middle_raised = landmarks[12].y < landmarks[5].y - 0.12

        if index_raised and not middle_raised:
            hand_gear = "DRIVE"
        elif middle_raised and not index_raised:
            hand_gear = "REVERSE"

        for lm in landmarks:
            x, y = int(lm.x * frame_display.shape[1]), int(lm.y * frame_display.shape[0])
            cv2.circle(frame_display, (x, y), 7, (0, 255, 0), -1)

    # 3. COMMAND FUSION
    command_to_send = "STOP"
    if hand_gear == "DRIVE":
        if gaze_state == "LEFT": command_to_send = "FORWARD_LEFT"
        elif gaze_state == "RIGHT": command_to_send = "FORWARD_RIGHT"
        else: command_to_send = "FORWARD"
    elif hand_gear == "REVERSE":
        command_to_send = "BACK"


    # 4. SEND TO ESP32
    if command_to_send != last_command:
        sock.sendto(command_to_send.encode(), (ESP32_IP, PORT))
        print(f"SENT: {command_to_send}")
        last_command = command_to_send

    # UI Feedback
    cv2.putText(frame_display, f"GEAR: {hand_gear} | MOTOR: {command_to_send}", (20, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("VisionDrive Control Center", frame_display)
    if cv2.waitKey(1) == 27: break

webcam.release()
cv2.destroyAllWindows()