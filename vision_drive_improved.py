"""
VisionDrive - Stable Gaze & Gesture Control (Fixed & Simplified)
"""

import cv2
from gaze_tracking import GazeTracking
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
import socket
import time
from collections import deque


# ==================== CONFIGURATION ====================
class Config:
    # ESP32 Connection
    ESP32_IP = "192.168.4.1"
    PORT = 4210
    
    # Camera Settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS_TARGET = 30
    
    # Gaze Detection - SIMPLIFIED AND STABLE
    # These values work for most people when looking straight ahead
    GAZE_CENTER_MIN = 0.40   # Below this = looking RIGHT
    GAZE_CENTER_MAX = 0.60   # Above this = looking LEFT
    
    # Hand Detection
    FINGER_RAISE_THRESHOLD = 0.12
    HAND_CONFIDENCE_MIN = 0.5
    
    # Smoothing
    SMOOTHING_WINDOW = 5  # Increased for more stability
    
    # Model Path
    HAND_MODEL = 'hand_landmarker.task'


# ==================== HAND LANDMARKS ====================
class HandLandmarks:
    WRIST = 0
    
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# ==================== COMMAND SMOOTHER ====================
class CommandSmoother:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)
    
    def add(self, command):
        self.history.append(command)
    
    def get_smoothed(self):
        if not self.history:
            return "STOP"
        # Return most common command
        return max(set(self.history), key=self.history.count)


# ==================== GAZE TRACKER (FIXED) ====================
class GazeTracker:
    def __init__(self):
        self.gaze = GazeTracking()
        self.ratio_history = deque(maxlen=10)  # Smooth gaze readings
        
    def refresh(self, frame):
        self.gaze.refresh(frame)
        ratio = self.gaze.horizontal_ratio()
        if ratio is not None:
            self.ratio_history.append(ratio)
    
    def get_smoothed_ratio(self):
        """Get average of recent ratios for stability"""
        if not self.ratio_history:
            return None
        return sum(self.ratio_history) / len(self.ratio_history)
    
    def get_state(self):
        """Get gaze direction with proper logic"""
        if self.gaze.is_blinking():
            return "BLINKING"
        
        # Use smoothed ratio
        ratio = self.get_smoothed_ratio()
        
        if ratio is None:
            return "CENTER"
        
        # CORRECT LOGIC for mirrored frame:
        # Lower ratio (0.0-0.4) = Eyes to the RIGHT side
        # Middle ratio (0.4-0.6) = Eyes CENTER
        # Higher ratio (0.6-1.0) = Eyes to the LEFT side
        
        if ratio < Config.GAZE_CENTER_MIN:
            return "RIGHT"
        elif ratio > Config.GAZE_CENTER_MAX:
            return "LEFT"
        else:
            return "CENTER"
    
    def get_annotated_frame(self):
        return self.gaze.annotated_frame()


# ==================== HAND DETECTOR ====================
class HandGestureDetector:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            model_data = f.read()
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=model_data),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=Config.HAND_CONFIDENCE_MIN,
            min_hand_presence_confidence=Config.HAND_CONFIDENCE_MIN
        )
        
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    
    def detect(self, frame_rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not hand_result.hand_landmarks:
            return "NEUTRAL", None
        
        landmarks = hand_result.hand_landmarks[0]
        
        # Check index and middle fingers
        index_raised = landmarks[HandLandmarks.INDEX_TIP].y < landmarks[HandLandmarks.INDEX_MCP].y - Config.FINGER_RAISE_THRESHOLD
        middle_raised = landmarks[HandLandmarks.MIDDLE_TIP].y < landmarks[HandLandmarks.MIDDLE_MCP].y - Config.FINGER_RAISE_THRESHOLD
        
        if index_raised and not middle_raised:
            gear = "DRIVE"
        elif middle_raised and not index_raised:
            gear = "REVERSE"
        else:
            gear = "NEUTRAL"
        
        return gear, landmarks
    
    def draw_hand(self, frame, landmarks):
        """Simple hand drawing - just lines and dots"""
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Define finger connections
        connections = [
            # Thumb
            (1, 2), (2, 3), (3, 4),
            # Index
            (5, 6), (6, 7), (7, 8),
            # Middle
            (9, 10), (10, 11), (11, 12),
            # Ring
            (13, 14), (14, 15), (15, 16),
            # Pinky
            (17, 18), (18, 19), (19, 20),
            # Palm
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            (5, 9), (9, 13), (13, 17)
        ]
        
        # Draw lines
        for start_idx, end_idx in connections:
            start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw points
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        return frame


# ==================== ESP32 CONTROLLER ====================
class ESP32Controller:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.last_command = "STOP"
        self.last_send_time = 0
        self.send_interval = 0.1
        
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.5)
            print(f"✓ Connected to {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def send_command(self, command, force=False):
        current_time = time.time()
        
        if command != self.last_command or force or \
           (current_time - self.last_send_time) > self.send_interval:
            
            try:
                self.sock.sendto(command.encode(), (self.ip, self.port))
                print(f"→ {command}")
                self.last_command = command
                self.last_send_time = current_time
                return True
            except:
                return False
        return False
    
    def close(self):
        if self.sock:
            self.sock.close()


# ==================== COMMAND FUSION ====================
class CommandFusion:
    @staticmethod
    def fuse(gaze_state, hand_gear):
        if hand_gear == "DRIVE":
            if gaze_state == "LEFT":
                return "FORWARD_LEFT"
            elif gaze_state == "RIGHT":
                return "FORWARD_RIGHT"
            else:
                return "FORWARD"
        
        elif hand_gear == "REVERSE":
            if gaze_state == "LEFT":
                return "BACK_LEFT"
            elif gaze_state == "RIGHT":
                return "BACK_RIGHT"
            else:
                return "BACK"
        
        else:
            return "STOP"


# ==================== MAIN APPLICATION ====================
class VisionDriveApp:
    def __init__(self):
        self.webcam = None
        self.gaze_tracker = None
        self.hand_detector = None
        self.esp32 = None
        self.smoother = CommandSmoother(Config.SMOOTHING_WINDOW)
        self.fps_counter = deque(maxlen=30)
        
    def initialize(self):
        print("\n" + "="*50)
        print("VisionDrive - Initializing")
        print("="*50)
        
        # Camera
        print("\n[1/4] Camera...")
        self.webcam = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not self.webcam.isOpened():
            print("✗ Camera failed")
            return False
        
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.webcam.set(cv2.CAP_PROP_FPS, Config.FPS_TARGET)
        print("✓ Camera ready")
        
        # Gaze Tracker
        print("\n[2/4] Gaze Tracker...")
        self.gaze_tracker = GazeTracker()
        print("✓ Gaze ready")
        
        # Hand Detector
        print("\n[3/4] Hand Detector...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, Config.HAND_MODEL)
        
        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            return False
        
        self.hand_detector = HandGestureDetector(model_path)
        print("✓ Hand ready")
        
        # ESP32
        print("\n[4/4] ESP32...")
        self.esp32 = ESP32Controller(Config.ESP32_IP, Config.PORT)
        if not self.esp32.connect():
            print("✗ ESP32 failed")
            return False
        
        print("\n" + "="*50)
        print("✓ SYSTEM READY")
        print("="*50)
        print("\nControls:")
        print("  Index Finger UP = DRIVE")
        print("  Middle Finger UP = REVERSE")
        print("  Look LEFT/RIGHT to steer")
        print("  ESC = Quit")
        print("\n" + "="*50 + "\n")
        
        return True
    
    def draw_simple_ui(self, frame, gaze_state, hand_gear, command, fps):
        """Simple, clean UI"""
        h, w = frame.shape[:2]
        
        # Black background bar
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Status
        cv2.putText(frame, f"GEAR: {hand_gear}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"GAZE: {gaze_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Command in color
        cmd_color = (0, 255, 0) if command != "STOP" else (0, 0, 255)
        cv2.putText(frame, f"CMD: {command}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("Running... (Press ESC to quit)\n")
        
        while True:
            loop_start = time.time()
            
            # Read frame
            ret, frame = self.webcam.read()
            if not ret:
                break
            
            # Mirror flip
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Gaze tracking
            self.gaze_tracker.refresh(frame_rgb)
            gaze_state = self.gaze_tracker.get_state()
            
            # Get annotated frame
            annotated_rgb = self.gaze_tracker.get_annotated_frame()
            frame_display = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            
            # Hand tracking
            timestamp_ms = int(time.time() * 1000)
            hand_gear, landmarks = self.hand_detector.detect(frame_rgb, timestamp_ms)
            
            # Draw hand
            frame_display = self.hand_detector.draw_hand(frame_display, landmarks)
            
            # Command fusion
            raw_command = CommandFusion.fuse(gaze_state, hand_gear)
            self.smoother.add(raw_command)
            final_command = self.smoother.get_smoothed()
            
            # Send to ESP32
            self.esp32.send_command(final_command)
            
            # Calculate FPS
            loop_time = time.time() - loop_start
            if loop_time > 0:
                self.fps_counter.append(1.0 / loop_time)
            avg_fps = sum(self.fps_counter) / len(self.fps_counter) if self.fps_counter else 0
            
            # Draw UI
            frame_display = self.draw_simple_ui(frame_display, gaze_state, hand_gear, 
                                               final_command, avg_fps)
            
            # Show
            cv2.imshow("VisionDrive", frame_display)
            
            # Check for ESC
            if cv2.waitKey(1) == 27:
                break
        
        self.cleanup()
    
    def cleanup(self):
        print("\nShutting down...")
        
        if self.esp32:
            self.esp32.send_command("STOP", force=True)
            time.sleep(0.2)
            self.esp32.close()
        
        if self.webcam:
            self.webcam.release()
        
        cv2.destroyAllWindows()
        print("✓ Done")


# ==================== MAIN ====================
def main():
    app = VisionDriveApp()
    
    if not app.initialize():
        print("\n✗ Initialization failed")
        sys.exit(1)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
