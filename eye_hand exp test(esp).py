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

# ==================== INITIALIZATION ====================
gaze = GazeTracking()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

with open(MODEL_PATH, "rb") as f:
    model_data = f.read()

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_buffer=model_data),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
webcam = cv2.VideoCapture(0)

last_command = "STOP" 

while True:
    _, frame = webcam.read()
    if frame is None: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gaze.refresh(frame_rgb)
    frame_annotated = gaze.annotated_frame()
    frame_display = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)

    # 1. Determine Gaze State
    if gaze.is_blinking(): gaze_state = "Blinking"
    elif gaze.is_right(): gaze_state = "Right"
    elif gaze.is_left(): gaze_state = "Left"
    else: gaze_state = "Center"

    # 2. Determine Hand State
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    hand_state = "No Hand"
    if hand_result.hand_landmarks:
        for idx, handedness in enumerate(hand_result.handedness):
            if handedness[0].category_name == "Right":
                landmarks = hand_result.hand_landmarks[idx]
                # Index finger raised logic
                index_raised = landmarks[8].y < landmarks[5].y - 0.12
                middle_folded = landmarks[12].y > landmarks[5].y
                if index_raised and middle_folded:
                    hand_state = "FORWARD"
                
                for lm in landmarks:
                    x, y = int(lm.x * frame_display.shape[1]), int(lm.y * frame_display.shape[0])
                    cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)

    # ==================== 3. COMMAND LOGIC ====================
    command_to_send = "STOP"
    if hand_state == "FORWARD":
        if gaze_state == "Left": command_to_send = "FORWARD_LEFT"
        elif gaze_state == "Right": command_to_send = "FORWARD_RIGHT"
        else: command_to_send = "FORWARD"
    elif gaze_state == "Left":
        command_to_send = "LEFT"
    elif gaze_state == "Right":
        command_to_send = "RIGHT"
    else:
        command_to_send = "STOP"

    # ==================== 4. ACTUAL WIFI SEND (UDP) ====================
    if command_to_send != last_command:
        try:
            # Send the string to the ESP32 IP and Port
            sock.sendto(command_to_send.encode('utf-8'), (ESP32_IP, PORT))
            print(f"WIFI SENT: {command_to_send}")
            last_command = command_to_send
        except Exception as e:
            print(f"Wifi Transmission error: {e}")

    # UI Feedback
    cv2.putText(frame_display, f"CMD: {command_to_send}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("VisionDrive Wireless Control", frame_display)

    if cv2.waitKey(1) == 27: break

webcam.release()
cv2.destroyAllWindows()