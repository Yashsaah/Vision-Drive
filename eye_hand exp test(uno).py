import cv2
from gaze_tracking import GazeTracking
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
import serial
import time

# ==================== SERIAL CONNECTION ====================
ARDUINO_PORT = 'COM7' 
BAUD_RATE = 115200

print("Attempting to grab COM7...")
try:
    # dsrdtr=True is key for Windows to trigger the Arduino reset
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1, dsrdtr=True)
    print("Resetting Arduino...")
    ser.setDTR(False)
    time.sleep(1)
    ser.setDTR(True)
    time.sleep(3) # Wait for Arduino to boot up
    ser.reset_input_buffer()
    print("SUCCESS: Arduino connected and reset.")
except Exception as e:
    print(f"FATAL ERROR: Could not open {ARDUINO_PORT}. Ensure Serial Monitor is CLOSED.")
    print(f"Details: {e}")
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

last_command = "STOP" # Initialize outside loop

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
                index_raised = landmarks[8].y < landmarks[5].y - 0.12
                middle_folded = landmarks[12].y > landmarks[5].y
                if index_raised and middle_folded:
                    hand_state = "FORWARD"
                for lm in landmarks:
                    x, y = int(lm.x * frame_display.shape[1]), int(lm.y * frame_display.shape[0])
                    cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)

    # ==================== 3. COMMAND LOGIC (THE BRAIN) ====================
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

    # ==================== 4. ACTUAL SERIAL SEND ====================
    if ser and command_to_send != last_command:
        try:
            ser.write((command_to_send + "\n").encode('utf-8'))
            ser.flush()
            print(f"SENT TO ARDUINO: {command_to_send}")
            last_command = command_to_send
        except Exception as e:
            print(f"Write error: {e}")

    # UI Feedback
    cv2.putText(frame_display, f"CMD: {command_to_send}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Car Control", frame_display)

    if cv2.waitKey(1) == 27: break

webcam.release()
cv2.destroyAllWindows()
if ser: ser.close()