import cv2
from gaze_tracking import GazeTracking
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import socket
import time
import requests

# ==================== CONFIGURATION ====================
# 1. Run the ESP32 code, Serial Monitor,paste that IP here:
ESP32_IP = "192.168.252.17" 
PORT = 4210

# 2. Telegram Details
TELEGRAM_TOKEN = "8541021099:AAFeD1ExwRj4kWsTYNoktKU7AwKc1PF2rfY"
CHAT_ID = "8058094794"

# SOS Logic
blink_count = 0
last_blink_time = 0
first_blink_time = 0
detection_window = 4.0
cooldown_period = 0.4
sos_active = False
sos_start_time = 0
last_command = "STOP"

def send_telegram_alert():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": "🚨 EMERGENCY: VisionDrive SOS Triggered! Patient needs immediate help."}
    try:
        requests.post(url, data=payload, timeout=2)
        print("Telegram Alert Sent!")
    except:
        print("Telegram Failed (Check Internet)")

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("UDP Socket Active")
except:
    print("Socket Error")

# ==================== AI INITIALIZATION ====================
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

while True:
    _, frame = webcam.read()
    if frame is None: break

    h_img, w_img, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- Eye Gaze Processing ---
    gaze.refresh(frame_rgb)
    frame_annotated = gaze.annotated_frame()
    frame_display = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)

    is_blinking = gaze.is_blinking()
    text = ""
    if is_blinking:
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    # --- SOS Blink Mechanism ---
    current_time = time.time()
    if blink_count > 0 and (current_time - first_blink_time > detection_window):
        blink_count = 0 

    if is_blinking:
        if current_time - last_blink_time > cooldown_period:
            if blink_count == 0: first_blink_time = current_time
            blink_count += 1
            last_blink_time = current_time
            print(f"Blink Detected: {blink_count}/5")
            if blink_count >= 5:
                sos_active = True
                sos_start_time = current_time
                send_telegram_alert() # TRIGGER TELEGRAM
                blink_count = 0

    # --- YOUR ORIGINAL UI (Gaze & Pupils) ---
    cv2.putText(frame_display, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame_display, "Left pupil: " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame_display, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # --- Hand Gesture Logic ---
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    hand_gear = "STOP"
    forward_text = ""

    if hand_result.hand_landmarks and hand_result.handedness:
        for idx, handedness in enumerate(hand_result.handedness):
            if handedness[0].category_name == "Right":
                landmarks = hand_result.hand_landmarks[idx]
                
                index_raised = landmarks[8].y < landmarks[5].y - 0.12
                pinky_raised = landmarks[20].y < landmarks[17].y - 0.12
                middle_folded = landmarks[12].y > landmarks[5].y
                ring_folded = landmarks[16].y > landmarks[5].y

                if index_raised and middle_folded and ring_folded and not pinky_raised:
                    hand_gear = "DRIVE"
                    forward_text = "FORWARD (Index Raised)"
                elif pinky_raised and middle_folded and ring_folded and not index_raised:
                    hand_gear = "REVERSE"
                    forward_text = "BACKWARD (Pinky Raised)"

                for lm in landmarks:
                    x = int(lm.x * w_img)
                    y = int(lm.y * h_img)
                    cv2.circle(frame_display, (x, y), 9, (0, 255, 0), -1)

    # --- Command Fusion ---
    command_to_send = "STOP"
    if sos_active:
        command_to_send = "SOS"
        if current_time - sos_start_time > 4: 
            sos_active = False
    elif hand_gear == "DRIVE":
        if gaze.is_left(): command_to_send = "FORWARD_LEFT"
        elif gaze.is_right(): command_to_send = "FORWARD_RIGHT"
        else: command_to_send = "FORWARD"
    elif hand_gear == "REVERSE":
        command_to_send = "BACK"

    if command_to_send != last_command:
        sock.sendto(command_to_send.encode(), (ESP32_IP, PORT))
        print(f"SENT: {command_to_send}")
        last_command = command_to_send

    # --- SOS Visual Overlays ---
    if sos_active:
        cv2.rectangle(frame_display, (0,0), (w_img, h_img), (0,0,255), 20)
        cv2.putText(frame_display, "SOS SIGNAL SENT", (w_img//4, h_img//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 3)

    # --- YOUR ORIGINAL UI (Forward/Backward Text) ---
    cv2.putText(frame_display, forward_text, (90, 220), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 2)
    
    cv2.imshow("Assistive Drive Hub", frame_display)

    if cv2.waitKey(1) == 27: break

webcam.release()
cv2.destroyAllWindows()