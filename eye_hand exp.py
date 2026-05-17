import cv2
from gaze_tracking import GazeTracking
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys  # Added for real-time printing

gaze = GazeTracking()

# Hand gesture model loading
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
    if frame is None:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Eye gaze processing
    gaze.refresh(frame_rgb)

    # Get gaze annotated frame (RGB)
    frame_annotated = gaze.annotated_frame()

    # Convert to BGR early so we can draw hand on it
    frame_display = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)

    # Eye gaze state (accurate, no up/down)
    if gaze.is_blinking():
        gaze_state = "Blinking"
    elif gaze.is_right():
        gaze_state = "Right"
    elif gaze.is_left():
        gaze_state = "Left"
    else:
        gaze_state = "Center"

    # Original on-screen text
    text = ""
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    else:
        text = "Looking center"

    cv2.putText(frame_display, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Pupil coordinates
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame_display, "Left pupil: " + str(left_pupil), (90, 130),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame_display, "Right pupil: " + str(right_pupil), (90, 165),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # ==================== HAND GESTURE DETECTION ====================
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    forward_text = ""
    hand_state = "No Hand"

    if hand_result.hand_landmarks and hand_result.handedness:
        for idx, handedness in enumerate(hand_result.handedness):
            if handedness[0].category_name == "Right":
                hand_state = "Right Hand"
                landmarks = hand_result.hand_landmarks[idx]

                index_tip = landmarks[8]
                index_mcp = landmarks[5]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]

                middle_folded = middle_tip.y > index_mcp.y
                ring_folded = ring_tip.y > index_mcp.y
                pinky_folded = pinky_tip.y > index_mcp.y
                index_raised = index_tip.y < index_mcp.y - 0.12

                if index_raised and middle_folded and ring_folded and pinky_folded:
                    forward_text = "FORWARD (Index Raised)"
                    hand_state = "FORWARD"

                # Draw green circles on hand
                for lm in landmarks:
                    x = int(lm.x * frame_display.shape[1])
                    y = int(lm.y * frame_display.shape[0])
                    cv2.circle(frame_display, (x, y), 9, (0, 255, 0), -1)

    # Show forward status on screen
    cv2.putText(frame_display, forward_text, (90, 220),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 2)

    # =================================================================

        # REAL-TIME PRINT IN TERMINAL
    print(f"GAZE: {gaze_state} | HAND: {hand_state}", flush=True)  # flush=True forces immediate output

    cv2.imshow("Demo - Eye Gaze + Hand Gesture", frame_display)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

webcam.release()
cv2.destroyAllWindows()