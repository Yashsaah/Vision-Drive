"""
VisionDrive - Self-Calibrating Gaze Tracking (Improved)
Seamless auto-calibration without blocking popups
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
import numpy as np
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
    
    # Gaze Detection - Auto-calibrated
    GAZE_CENTER_MIN = 0.60
    GAZE_CENTER_MAX = 0.80
    
    # Calibration Settings
    CALIBRATION_SAMPLES = 20  # Reduced from 30 for faster calibration
    CALIBRATION_TIMEOUT = 5.0  # Max 5 seconds for calibration
    
    # Hand Detection Thresholds
    FINGER_RAISE_THRESHOLD = 0.12
    HAND_CONFIDENCE_MIN = 0.5
    
    # Performance Settings
    FPS_TARGET = 30
    SMOOTHING_WINDOW = 5
    
    # Visualization Settings
    LANDMARK_COLOR = (0, 255, 0)
    CONNECTION_COLOR = (255, 255, 0)
    ACTIVE_FINGER_COLOR = (0, 255, 255)
    LANDMARK_RADIUS = 5
    LINE_THICKNESS = 2
    
    # Model Path
    HAND_MODEL = 'hand_landmarker.task'


# MediaPipe Hand Landmark Indices
class HandLandmarks:
    """Hand landmark indices for MediaPipe"""
    WRIST = 0
    
    # Thumb
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    
    # Index finger
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    
    # Middle finger
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    
    # Ring finger
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    
    # Pinky finger
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    # Hand connections (bone structure)
    HAND_CONNECTIONS = [
        # Palm
        (WRIST, THUMB_CMC), (WRIST, INDEX_MCP), (WRIST, MIDDLE_MCP),
        (WRIST, RING_MCP), (WRIST, PINKY_MCP),
        (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
        
        # Thumb
        (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
        
        # Index finger
        (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
        
        # Middle finger
        (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
        
        # Ring finger
        (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
        
        # Pinky finger
        (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    ]


# ==================== UTILITIES ====================
class CommandSmoother:
    """Smooth commands to prevent jittery behavior"""
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)
    
    def add(self, command):
        self.history.append(command)
    
    def get_smoothed(self):
        if not self.history:
            return "STOP"
        return max(set(self.history), key=self.history.count)


class CalibratingGazeTracker:
    """
    Fast self-calibrating gaze tracker
    Calibrates in background without blocking UI
    """
    def __init__(self):
        self.gaze = GazeTracking()
        self.ratio_history = deque(maxlen=10)
        
        # Quick background calibration
        self.calibrating = True
        self.calibration_samples = []
        self.calibration_start_time = time.time()
        
        # Adjustable thresholds
        self.center_min = Config.GAZE_CENTER_MIN
        self.center_max = Config.GAZE_CENTER_MAX
        self.threshold_adjustment = 0.02
        
        # Status
        self.calibrated = False
        
    def refresh(self, frame):
        self.gaze.refresh(frame)
        
        current_ratio = self.gaze.horizontal_ratio()
        if current_ratio is not None:
            self.ratio_history.append(current_ratio)
            
            # Quick background calibration
            if self.calibrating:
                elapsed = time.time() - self.calibration_start_time
                
                # Collect samples for up to 5 seconds or until we have enough
                if len(self.calibration_samples) < Config.CALIBRATION_SAMPLES and elapsed < Config.CALIBRATION_TIMEOUT:
                    self.calibration_samples.append(current_ratio)
                else:
                    # Auto-complete calibration
                    self.complete_calibration()
    
    def get_smoothed_ratio(self):
        """Get smoothed horizontal ratio"""
        if not self.ratio_history:
            return None
        return sum(self.ratio_history) / len(self.ratio_history)
    
    def complete_calibration(self):
        """Complete calibration silently"""
        if len(self.calibration_samples) < 5:
            # Not enough samples, use defaults
            self.calibrating = False
            self.calibrated = True
            print("⚠ Quick calibration - using default thresholds")
            return
        
        # Calculate center position
        center_position = sum(self.calibration_samples) / len(self.calibration_samples)
        
        # Set thresholds with margin
        margin = 0.08
        self.center_min = center_position - margin
        self.center_max = center_position + margin
        
        self.calibrating = False
        self.calibrated = True
        
        print(f"\n✓ Calibrated - Center: {center_position:.3f}, Range: [{self.center_min:.3f}, {self.center_max:.3f}]")
    
    def adjust_left_threshold(self, increase=True):
        """Adjust the left (max) threshold"""
        if increase:
            self.center_max += self.threshold_adjustment
        else:
            self.center_max -= self.threshold_adjustment
        print(f"Left threshold: {self.center_max:.3f}")
    
    def adjust_right_threshold(self, increase=True):
        """Adjust the right (min) threshold"""
        if increase:
            self.center_min += self.threshold_adjustment
        else:
            self.center_min -= self.threshold_adjustment
        print(f"Right threshold: {self.center_min:.3f}")
    
    def get_state(self):
        """Get gaze state using calibrated thresholds"""
        if self.gaze.is_blinking():
            return "BLINKING"
        
        ratio = self.get_smoothed_ratio()
        
        if ratio is None:
            return "CENTER"
        
        # Use calibrated thresholds
        if ratio < self.center_min:
            return "LEFT"
        elif ratio > self.center_max:
            return "RIGHT"
        else:
            return "CENTER"
    
    def get_horizontal_ratio(self):
        """Get the current smoothed horizontal ratio for display"""
        return self.get_smoothed_ratio()
    
    def get_annotated_frame(self):
        return self.gaze.annotated_frame()
    
    def get_debug_info(self):
        """Get detailed debug information"""
        ratio = self.get_smoothed_ratio()
        raw_ratio = self.gaze.horizontal_ratio()
        
        return {
            'raw_ratio': raw_ratio,
            'smoothed_ratio': ratio,
            'history_size': len(self.ratio_history),
            'center_min': self.center_min,
            'center_max': self.center_max,
            'calibrating': self.calibrating,
            'calibrated': self.calibrated,
            'calibration_progress': len(self.calibration_samples)
        }
    
    def start_recalibration(self):
        """Start a new calibration"""
        self.calibrating = True
        self.calibrated = False
        self.calibration_samples = []
        self.calibration_start_time = time.time()
        print("\n🎯 Recalibrating...")


class HandGestureDetector:
    """Enhanced hand gesture detection with full skeleton visualization"""
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
        """Detect hand and return gear state with landmarks"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not hand_result.hand_landmarks:
            return "NEUTRAL", None, {}
        
        landmarks = hand_result.hand_landmarks[0]
        
        finger_states = self.detect_finger_states(landmarks)
        gear = self.determine_gear(finger_states)
        
        return gear, landmarks, finger_states
    
    def detect_finger_states(self, landmarks):
        """Detect which fingers are raised"""
        states = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        index_raised = landmarks[HandLandmarks.INDEX_TIP].y < landmarks[HandLandmarks.INDEX_MCP].y - Config.FINGER_RAISE_THRESHOLD
        middle_raised = landmarks[HandLandmarks.MIDDLE_TIP].y < landmarks[HandLandmarks.INDEX_MCP].y - Config.FINGER_RAISE_THRESHOLD
        ring_raised = landmarks[HandLandmarks.RING_TIP].y < landmarks[HandLandmarks.INDEX_MCP].y - Config.FINGER_RAISE_THRESHOLD
        pinky_raised = landmarks[HandLandmarks.PINKY_TIP].y < landmarks[HandLandmarks.INDEX_MCP].y - Config.FINGER_RAISE_THRESHOLD
        thumb_raised = landmarks[HandLandmarks.THUMB_TIP].x < landmarks[HandLandmarks.THUMB_IP].x - 0.05
        
        states['thumb'] = thumb_raised
        states['index'] = index_raised
        states['middle'] = middle_raised
        states['ring'] = ring_raised
        states['pinky'] = pinky_raised
        
        return states
    
    def determine_gear(self, finger_states):
        """Determine gear from finger states"""
        if finger_states['index'] and not finger_states['middle']:
            return "DRIVE"
        elif finger_states['middle'] and not finger_states['index']:
            return "REVERSE"
        else:
            return "NEUTRAL"
    
    def draw_hand_skeleton(self, frame, landmarks, finger_states):
        """Draw complete hand skeleton with highlighting for active fingers"""
        h, w = frame.shape[:2]
        
        for connection in HandLandmarks.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            
            start_point = (
                int(landmarks[start_idx].x * w),
                int(landmarks[start_idx].y * h)
            )
            end_point = (
                int(landmarks[end_idx].x * w),
                int(landmarks[end_idx].y * h)
            )
            
            color = self.get_connection_color(start_idx, end_idx, finger_states)
            cv2.line(frame, start_point, end_point, color, Config.LINE_THICKNESS)
        
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            color = self.get_landmark_color(idx, finger_states)
            cv2.circle(frame, (x, y), Config.LANDMARK_RADIUS, color, -1)
            cv2.circle(frame, (x, y), Config.LANDMARK_RADIUS + 2, (255, 255, 255), 1)
        
        return frame
    
    def get_connection_color(self, start_idx, end_idx, finger_states):
        """Get color for connection based on which finger it belongs to"""
        if start_idx in [5, 6, 7, 8] or end_idx in [5, 6, 7, 8]:
            if finger_states['index']:
                return Config.ACTIVE_FINGER_COLOR
        
        if start_idx in [9, 10, 11, 12] or end_idx in [9, 10, 11, 12]:
            if finger_states['middle']:
                return Config.ACTIVE_FINGER_COLOR
        
        if start_idx in [13, 14, 15, 16] or end_idx in [13, 14, 15, 16]:
            if finger_states['ring']:
                return (255, 0, 255)
        
        if start_idx in [17, 18, 19, 20] or end_idx in [17, 18, 19, 20]:
            if finger_states['pinky']:
                return (255, 128, 0)
        
        if start_idx in [1, 2, 3, 4] or end_idx in [1, 2, 3, 4]:
            if finger_states['thumb']:
                return (128, 0, 255)
        
        return Config.CONNECTION_COLOR
    
    def get_landmark_color(self, idx, finger_states):
        """Get color for landmark based on which finger it belongs to"""
        if idx in [5, 6, 7, 8]:
            return Config.ACTIVE_FINGER_COLOR if finger_states['index'] else Config.LANDMARK_COLOR
        if idx in [9, 10, 11, 12]:
            return Config.ACTIVE_FINGER_COLOR if finger_states['middle'] else Config.LANDMARK_COLOR
        if idx in [13, 14, 15, 16]:
            return (255, 0, 255) if finger_states['ring'] else Config.LANDMARK_COLOR
        if idx in [17, 18, 19, 20]:
            return (255, 128, 0) if finger_states['pinky'] else Config.LANDMARK_COLOR
        if idx in [1, 2, 3, 4]:
            return (128, 0, 255) if finger_states['thumb'] else Config.LANDMARK_COLOR
        return (255, 255, 255)


class ESP32Controller:
    """Handle ESP32 communication"""
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.last_command = "STOP"
    
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.5)
            print(f"✓ UDP Socket created for {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Socket creation failed: {e}")
            return False
    
    def send_command(self, command, force=False):
        """Send command only if it changed or force=True"""
        if command != self.last_command or force:
            try:
                self.sock.sendto(command.encode(), (self.ip, self.port))
                self.last_command = command
                return True
            except Exception as e:
                print(f"✗ Send failed: {e}")
                return False
        return False
    
    def close(self):
        if self.sock:
            self.sock.close()


class CommandFusion:
    """Combine gaze and gesture into motor commands"""
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
            return "BACK"
        else:
            return "STOP"


# ==================== MAIN APPLICATION ====================
class VisionDriveApp:
    """Main application controller"""
    def __init__(self):
        self.running = False
        self.webcam = None
        self.gaze_tracker = None
        self.hand_detector = None
        self.esp32 = None
        self.smoother = CommandSmoother(Config.SMOOTHING_WINDOW)
        self.fps_counter = deque(maxlen=30)
        self.show_debug = True
    
    def initialize(self):
        print("\n" + "="*50)
        print("VisionDrive - Quick Calibration System")
        print("="*50)
        
        # Initialize Camera
        print("\n[1/4] Initializing Camera...")
        self.webcam = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not self.webcam.isOpened():
            print("✗ Failed to open camera")
            return False
        
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.webcam.set(cv2.CAP_PROP_FPS, Config.FPS_TARGET)
        self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✓ Camera initialized")
        
        # Initialize Gaze Tracker
        print("\n[2/4] Initializing Gaze Tracker...")
        self.gaze_tracker = CalibratingGazeTracker()
        print("✓ Gaze tracker ready - calibrating in background...")
        
        # Initialize Hand Detector
        print("\n[3/4] Initializing Hand Detector...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, Config.HAND_MODEL)
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            return False
        
        self.hand_detector = HandGestureDetector(model_path)
        print("✓ Hand detector ready")
        
        # Initialize ESP32
        print("\n[4/4] Connecting to ESP32...")
        self.esp32 = ESP32Controller(Config.ESP32_IP, Config.PORT)
        if not self.esp32.connect():
            print("✗ ESP32 connection failed")
            return False
        
        print("\n" + "="*50)
        print("✓ ALL SYSTEMS READY")
        print("="*50)
        print("\n📍 System is auto-calibrating in background")
        print("   Just start using - calibration happens automatically!")
        print("\nControls:")
        print("  • Index finger UP = Drive Forward")
        print("  • Middle finger UP = Reverse")
        print("  • Gaze LEFT/RIGHT = Steer")
        print("\nKeyboard:")
        print("  • Press 'C' to recalibrate")
        print("  • Press 'D' to toggle debug info")
        print("  • Press '←/→' to adjust left threshold")
        print("  • Press '[/]' to adjust right threshold")
        print("  • Press ESC to quit")
        print("\n" + "="*50 + "\n")
        
        return True
    
    def draw_calibration_indicator(self, frame):
        """Small non-intrusive calibration indicator"""
        if not self.gaze_tracker.calibrating:
            return frame
        
        h, w = frame.shape[:2]
        
        # Small indicator in top-right corner
        indicator_x = w - 200
        indicator_y = 60
        
        progress = self.gaze_tracker.get_debug_info()['calibration_progress']
        target = Config.CALIBRATION_SAMPLES
        progress_pct = min(100, int((progress / target) * 100))
        
        # Compact progress bar
        bar_width = 150
        bar_height = 8
        filled_width = int(bar_width * (progress / target))
        
        # Background
        cv2.rectangle(frame, (indicator_x, indicator_y), 
                     (indicator_x + bar_width, indicator_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress
        cv2.rectangle(frame, (indicator_x, indicator_y), 
                     (indicator_x + filled_width, indicator_y + bar_height), 
                     (0, 255, 255), -1)
        
        # Text
        cv2.putText(frame, f"Calibrating {progress_pct}%", 
                   (indicator_x, indicator_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame
    
    def draw_gaze_debug_bar(self, frame, gaze_state, ratio):
        """Draw detailed gaze debug visualization"""
        if ratio is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Get current thresholds
        center_min = self.gaze_tracker.center_min
        center_max = self.gaze_tracker.center_max
        
        # Bar dimensions
        bar_y = h - 40
        bar_height = 20
        bar_start_x = 50
        bar_width = w - 100
        
        # Draw background bar
        cv2.rectangle(frame, (bar_start_x, bar_y), 
                     (bar_start_x + bar_width, bar_y + bar_height), 
                     (40, 40, 40), -1)
        
        # Calculate zone positions
        center_start_x = int(bar_start_x + bar_width * center_min)
        center_end_x = int(bar_start_x + bar_width * center_max)
        
        # Draw LEFT zone (BLUE)
        cv2.rectangle(frame, (bar_start_x, bar_y), 
                     (center_start_x, bar_y + bar_height), 
                     (200, 100, 0), -1)
        
        # Draw CENTER zone (GREEN)
        cv2.rectangle(frame, (center_start_x, bar_y), 
                     (center_end_x, bar_y + bar_height), 
                     (0, 150, 0), -1)
        
        # Draw RIGHT zone (RED)
        cv2.rectangle(frame, (center_end_x, bar_y), 
                     (bar_start_x + bar_width, bar_y + bar_height), 
                     (0, 100, 200), -1)
        
        # Draw current position indicator
        current_x = int(bar_start_x + bar_width * ratio)
        current_x = max(bar_start_x, min(current_x, bar_start_x + bar_width))
        
        cv2.circle(frame, (current_x, bar_y + bar_height // 2), 12, (0, 255, 255), -1)
        cv2.circle(frame, (current_x, bar_y + bar_height // 2), 14, (255, 255, 255), 2)
        
        # Labels
        cv2.putText(frame, "LEFT", (bar_start_x + 5, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 2)
        cv2.putText(frame, "CENTER", (center_start_x + 20, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "RIGHT", (bar_start_x + bar_width - 50, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 2)
        
        # Show exact ratio value
        ratio_text = f"Ratio: {ratio:.3f}"
        cv2.putText(frame, ratio_text, (current_x - 50, bar_y + bar_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_ui(self, frame, gaze_state, hand_gear, command, fps, finger_states):
        """Draw enhanced UI overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Calibration status indicator
        if self.gaze_tracker.calibrated:
            status_color = (0, 255, 0)
            status_text = "✓ CALIBRATED"
        else:
            status_color = (0, 255, 255)
            status_text = "○ CALIBRATING"
        
        cv2.putText(frame, status_text, (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Status text
        y_offset = 35
        cv2.putText(frame, f"GEAR: {hand_gear}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 35
        
        # Color code gaze state
        gaze_color = (255, 255, 255)
        if gaze_state == "LEFT":
            gaze_color = (200, 100, 0)
        elif gaze_state == "RIGHT":
            gaze_color = (0, 100, 200)
        elif gaze_state == "CENTER":
            gaze_color = (0, 255, 0)
        
        cv2.putText(frame, f"GAZE: {gaze_state}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, gaze_color, 2)
        y_offset += 35
        
        cmd_color = (0, 255, 0) if command != "STOP" else (0, 0, 255)
        cv2.putText(frame, f"CMD: {command}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_color, 2)
        
        if finger_states:
            y_offset += 35
            finger_text = "FINGERS: "
            for finger, state in finger_states.items():
                if state:
                    finger_text += f"{finger.upper()[0]} "
            cv2.putText(frame, finger_text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Debug info
        if self.show_debug:
            debug_info = self.gaze_tracker.get_debug_info()
            debug_y = h - 120
            cv2.putText(frame, "DEBUG:", (w - 250, debug_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            debug_y += 20
            cv2.putText(frame, f"Raw: {debug_info['raw_ratio']:.3f}" if debug_info['raw_ratio'] else "Raw: None", 
                       (w - 250, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            debug_y += 15
            cv2.putText(frame, f"Smooth: {debug_info['smoothed_ratio']:.3f}" if debug_info['smoothed_ratio'] else "Smooth: None",
                       (w - 250, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            debug_y += 15
            cv2.putText(frame, f"Min: {debug_info['center_min']:.3f}",
                       (w - 250, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 1)
            debug_y += 15
            cv2.putText(frame, f"Max: {debug_info['center_max']:.3f}",
                       (w - 250, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 0), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        self.running = True
        
        while self.running:
            loop_start = time.time()
            
            ret, frame = self.webcam.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Gaze Tracking (calibrates in background)
            self.gaze_tracker.refresh(frame_rgb)
            gaze_state = self.gaze_tracker.get_state()
            annotated_rgb = self.gaze_tracker.get_annotated_frame()
            frame_display = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            
            # Hand Tracking
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            hand_gear, landmarks, finger_states = self.hand_detector.detect(frame_rgb, timestamp_ms)
            
            # Draw hand skeleton
            if landmarks:
                frame_display = self.hand_detector.draw_hand_skeleton(
                    frame_display, landmarks, finger_states
                )
            
            # Command Fusion
            raw_command = CommandFusion.fuse(gaze_state, hand_gear)
            self.smoother.add(raw_command)
            final_command = self.smoother.get_smoothed()
            
            # Send to ESP32 (always, even during calibration)
            self.esp32.send_command(final_command)
            
            # Calculate FPS
            loop_time = time.time() - loop_start
            self.fps_counter.append(1.0 / loop_time if loop_time > 0 else 0)
            avg_fps = sum(self.fps_counter) / len(self.fps_counter)
            
            # Draw UI
            frame_display = self.draw_ui(frame_display, gaze_state, hand_gear,
                                        final_command, avg_fps, finger_states)
            
            # Small calibration indicator if still calibrating
            frame_display = self.draw_calibration_indicator(frame_display)
            
            # Draw gaze debug bar
            ratio = self.gaze_tracker.get_horizontal_ratio()
            frame_display = self.draw_gaze_debug_bar(frame_display, gaze_state, ratio)
            
            cv2.imshow("VisionDrive", frame_display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('d') or key == ord('D'):
                self.show_debug = not self.show_debug
                print(f"Debug info: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('c') or key == ord('C'):
                self.gaze_tracker.start_recalibration()
            elif key == 81:  # Left arrow
                self.gaze_tracker.adjust_left_threshold(False)
            elif key == 83:  # Right arrow
                self.gaze_tracker.adjust_left_threshold(True)
            elif key == ord('['):
                self.gaze_tracker.adjust_right_threshold(False)
            elif key == ord(']'):
                self.gaze_tracker.adjust_right_threshold(True)
        
        self.cleanup()
    
    def cleanup(self):
        print("\n\nShutting down...")
        if self.esp32:
            self.esp32.send_command("STOP", force=True)
            time.sleep(0.2)
            self.esp32.close()
        if self.webcam:
            self.webcam.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


# ==================== ENTRY POINT ====================
def main():
    app = VisionDriveApp()
    
    if not app.initialize():
        print("\n✗ Initialization failed. Exiting.")
        sys.exit(1)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()