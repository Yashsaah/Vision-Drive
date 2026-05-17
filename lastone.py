"""
VisionDrive - Accurate Gaze-Based Control System
Steering: Look LEFT = Turn LEFT
         Look RIGHT = Turn RIGHT
         Look CENTER = Go STRAIGHT

Author: Advanced Computer Vision System
Version: 5.0 - Ultra Accurate Gaze Control
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
import json


# ==================== CONFIGURATION ====================
class Config:
    """System configuration"""
    
    def __init__(self):
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.fps_target = 30
        
        # ESP32 settings
        self.esp32_ip = "192.168.4.1"
        self.esp32_port = 4210
        
        # Hand detection
        self.finger_raise_threshold = 0.12
        self.hand_confidence = 0.5
        
        # Gaze detection - HIGHLY TUNED FOR ACCURACY
        # These thresholds work with horizontal_ratio() which returns 0.0-1.0
        # Lower values = looking LEFT (from user's perspective)
        # Higher values = looking RIGHT (from user's perspective)
        self.gaze_left_threshold = 0.45   # If ratio < 0.45 = LEFT
        self.gaze_right_threshold = 0.55  # If ratio > 0.55 = RIGHT
        # Between 0.45-0.55 = CENTER (10% dead zone for stability)
        
        # Smoothing for super stable detection
        self.gaze_smooth_window = 8  # More smoothing for accuracy
        self.command_smooth_window = 5
        
        # Model path
        self.hand_model = 'hand_landmarker.task'


# ==================== ULTRA-ACCURATE GAZE TRACKER ====================
class AccurateGazeTracker:
    """
    High-precision gaze tracking using horizontal pupil ratio
    """
    
    def __init__(self, config):
        self.config = config
        self.gaze = GazeTracking()
        
        # Multi-level smoothing for maximum accuracy
        self.ratio_history = deque(maxlen=config.gaze_smooth_window)
        
        # State tracking with hysteresis
        self.current_state = "CENTER"
        self.state_confidence = 0
        self.min_confidence = 3  # Frames needed to change state
        
        # Statistics for monitoring
        self.last_ratio = 0.5
        
    def refresh(self, frame):
        """Process new frame"""
        try:
            self.gaze.refresh(frame)
        except Exception as e:
            print(f"Gaze refresh error: {e}")
    
    def get_gaze_direction(self):
        """
        Get current gaze direction with high accuracy
        Returns: "LEFT", "RIGHT", "CENTER", or "BLINKING"
        """
        try:
            # Check for blinking first
            if self.gaze.is_blinking():
                return "BLINKING"
            
            # Get horizontal ratio (0.0 = far right, 1.0 = far left)
            ratio = self.gaze.horizontal_ratio()
            
            # If ratio is None, use last known value
            if ratio is None:
                ratio = self.last_ratio
            else:
                # Add to history for smoothing
                self.ratio_history.append(ratio)
                self.last_ratio = ratio
            
            # Calculate smoothed ratio using weighted average
            if len(self.ratio_history) > 0:
                # Use exponential moving average for smoother tracking
                weights = np.exp(np.linspace(-1, 0, len(self.ratio_history)))
                weights /= weights.sum()
                smoothed_ratio = np.average(list(self.ratio_history), weights=weights)
            else:
                smoothed_ratio = ratio
            
            # Determine direction based on thresholds
            if smoothed_ratio < self.config.gaze_left_threshold:
                new_state = "LEFT"
            elif smoothed_ratio > self.config.gaze_right_threshold:
                new_state = "RIGHT"
            else:
                new_state = "CENTER"
            
            # Apply hysteresis - require multiple consistent frames before switching
            if new_state == self.current_state:
                # Same state - increase confidence
                self.state_confidence = min(self.state_confidence + 1, self.min_confidence)
            else:
                # Different state - decrease confidence
                self.state_confidence -= 1
                
                # Only change state if confidence drops to zero
                if self.state_confidence <= 0:
                    self.current_state = new_state
                    self.state_confidence = self.min_confidence
            
            return self.current_state
            
        except Exception as e:
            print(f"Gaze direction error: {e}")
            return self.current_state
    
    def get_debug_info(self):
        """Get debug information"""
        try:
            if len(self.ratio_history) > 0:
                weights = np.exp(np.linspace(-1, 0, len(self.ratio_history)))
                weights /= weights.sum()
                smoothed = np.average(list(self.ratio_history), weights=weights)
            else:
                smoothed = self.last_ratio
            
            return {
                'raw_ratio': self.last_ratio,
                'smoothed_ratio': smoothed,
                'state': self.current_state,
                'confidence': self.state_confidence
            }
        except:
            return {
                'raw_ratio': None,
                'smoothed_ratio': None,
                'state': 'UNKNOWN',
                'confidence': 0
            }
    
    def get_annotated_frame(self):
        """Get frame with gaze annotations"""
        try:
            return self.gaze.annotated_frame()
        except Exception as e:
            print(f"Frame annotation error: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)


# ==================== HAND DETECTOR ====================
class HandDetector:
    """Hand tracking with gesture recognition"""
    
    def __init__(self, model_path, config):
        self.config = config
        
        # Load MediaPipe model
        with open(model_path, "rb") as f:
            model_data = f.read()
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=model_data),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.hand_confidence,
            min_hand_presence_confidence=config.hand_confidence
        )
        
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        self.gesture_history = deque(maxlen=5)
    
    def detect(self, frame_rgb, timestamp_ms):
        """Detect hand and determine gesture"""
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if not hand_result.hand_landmarks:
                self.gesture_history.append("NEUTRAL")
                return self._get_stable_gesture(), None
            
            landmarks = hand_result.hand_landmarks[0]
            
            # Detect raised fingers
            index_raised = landmarks[8].y < landmarks[5].y - self.config.finger_raise_threshold
            middle_raised = landmarks[12].y < landmarks[9].y - self.config.finger_raise_threshold
            
            # Determine gesture
            if index_raised and not middle_raised:
                gesture = "DRIVE"
            elif middle_raised and not index_raised:
                gesture = "REVERSE"
            else:
                gesture = "NEUTRAL"
            
            self.gesture_history.append(gesture)
            return self._get_stable_gesture(), landmarks
        
        except Exception as e:
            print(f"Hand detection error: {e}")
            self.gesture_history.append("NEUTRAL")
            return "NEUTRAL", None
    
    def _get_stable_gesture(self):
        """Get most common gesture from recent history"""
        if not self.gesture_history:
            return "NEUTRAL"
        return max(set(self.gesture_history), key=self.gesture_history.count)
    
    def draw_hand(self, frame, landmarks):
        """Draw hand skeleton"""
        if landmarks is None:
            return frame
        
        try:
            h, w = frame.shape[:2]
            
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            
            for start_idx, end_idx in connections:
                start = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, start, end, (0, 255, 0), 2)
            
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        except Exception as e:
            print(f"Hand drawing error: {e}")
        
        return frame


# ==================== COMMAND CONTROLLER ====================
class CommandController:
    """Command generation with smoothing"""
    
    def __init__(self, config):
        self.config = config
        self.command_history = deque(maxlen=config.command_smooth_window)
        self.last_command = "STOP"
    
    def generate_command(self, gaze_direction, gesture):
        """
        Generate motor command from gaze and hand gesture
        
        Gaze: LEFT, RIGHT, CENTER, BLINKING
        Gestures: DRIVE, REVERSE, NEUTRAL
        """
        
        # During blinking, maintain last command
        if gaze_direction == "BLINKING":
            raw_command = self.last_command if self.last_command != "STOP" else "STOP"
        
        # Generate command based on gesture and gaze
        elif gesture == "DRIVE":
            if gaze_direction == "LEFT":
                raw_command = "FORWARD_LEFT"
            elif gaze_direction == "RIGHT":
                raw_command = "FORWARD_RIGHT"
            else:
                raw_command = "FORWARD"
        
        elif gesture == "REVERSE":
            if gaze_direction == "LEFT":
                raw_command = "BACK_LEFT"
            elif gaze_direction == "RIGHT":
                raw_command = "BACK_RIGHT"
            else:
                raw_command = "BACK"
        
        else:
            raw_command = "STOP"
        
        # Smooth commands
        self.command_history.append(raw_command)
        smoothed_command = max(set(self.command_history), key=self.command_history.count)
        self.last_command = smoothed_command
        
        return smoothed_command


# ==================== ESP32 COMMUNICATOR ====================
class ESP32Communicator:
    """UDP communication with ESP32"""
    
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.last_command = None
        self.last_send_time = 0
        self.send_interval = 0.05
        
    def connect(self):
        """Initialize connection"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.5)
            print(f"✓ ESP32 ready: {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ ESP32 failed: {e}")
            return False
    
    def send(self, command, force=False):
        """Send command"""
        try:
            current_time = time.time()
            
            if force or command != self.last_command or \
               (current_time - self.last_send_time) > self.send_interval:
                
                if self.sock:
                    self.sock.sendto(command.encode(), (self.ip, self.port))
                    self.last_command = command
                    self.last_send_time = current_time
                    return True
            
            return False
        except:
            return False
    
    def close(self):
        """Close connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass


# ==================== MAIN APPLICATION ====================
class VisionDriveSystem:
    """Main application"""
    
    def __init__(self):
        self.config = Config()
        self.gaze_tracker = None
        self.hand_detector = None
        self.command_controller = None
        self.esp32 = None
        self.webcam = None
        self.fps_history = deque(maxlen=30)
        self.running = False
        
    def initialize(self):
        """Initialize all subsystems"""
        print("\n" + "="*70)
        print("VISIONDRIVE - ULTRA ACCURATE GAZE CONTROL")
        print("="*70)
        
        # Camera
        print("\n[1/4] Initializing Camera...")
        self.webcam = cv2.VideoCapture(self.config.camera_index)
        if not self.webcam.isOpened():
            print("✗ Camera failed")
            return False
        
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.webcam.set(cv2.CAP_PROP_FPS, self.config.fps_target)
        self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ret, _ = self.webcam.read()
        if not ret:
            print("✗ Camera test failed")
            self.webcam.release()
            return False
        
        print("✓ Camera ready")
        
        # Gaze tracker
        print("\n[2/4] Initializing Gaze Tracking...")
        try:
            self.gaze_tracker = AccurateGazeTracker(self.config)
            print("✓ Gaze tracker ready")
        except Exception as e:
            print(f"✗ Gaze tracker failed: {e}")
            self.webcam.release()
            return False
        
        # Hand detector
        print("\n[3/4] Initializing Hand Detection...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, self.config.hand_model)
        
        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            self.webcam.release()
            return False
        
        try:
            self.hand_detector = HandDetector(model_path, self.config)
            print("✓ Hand detector ready")
        except Exception as e:
            print(f"✗ Hand detector failed: {e}")
            self.webcam.release()
            return False
        
        # ESP32
        print("\n[4/4] Connecting to ESP32...")
        self.esp32 = ESP32Communicator(self.config.esp32_ip, self.config.esp32_port)
        if not self.esp32.connect():
            print("⚠ ESP32 offline (continuing anyway)")
        
        self.command_controller = CommandController(self.config)
        
        print("\n" + "="*70)
        print("✓ SYSTEM READY")
        print("="*70)
        
        print("\n👀 GAZE CONTROLS:")
        print("  ← Look LEFT    = Turn LEFT")
        print("  → Look RIGHT   = Turn RIGHT")
        print("  ● Look CENTER  = Go STRAIGHT")
        
        print("\n🖐️  HAND GESTURES:")
        print("  ☝️  Index UP   = DRIVE")
        print("  🖕 Middle UP   = REVERSE")
        print("  ✋ No fingers  = STOP")
        
        print("\n⌨️  CONTROLS:")
        print("  [ESC] - Quit")
        print("  [C]   - Calibrate (adjust sensitivity)")
        print("\n" + "="*70 + "\n")
        
        return True
    
    def draw_ui(self, frame, gaze_dir, gesture, command, fps, debug_info):
        """Draw UI with gaze visualization"""
        try:
            h, w = frame.shape[:2]
            
            # Dark top bar
            cv2.rectangle(frame, (0, 0), (w, 140), (0, 0, 0), -1)
            
            # Gesture
            cv2.putText(frame, f"GEAR: {gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Gaze direction with color coding
            gaze_color = (255, 255, 0)  # Yellow default
            if gaze_dir == "LEFT":
                gaze_color = (0, 255, 255)  # Cyan
            elif gaze_dir == "RIGHT":
                gaze_color = (255, 0, 255)  # Magenta
            elif gaze_dir == "CENTER":
                gaze_color = (0, 255, 0)  # Green
            
            cv2.putText(frame, f"GAZE: {gaze_dir}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
            
            # Command
            cmd_color = (0, 255, 0) if command != "STOP" else (0, 0, 255)
            cv2.putText(frame, f"CMD: {command}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_color, 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Gaze ratio visualization (bottom)
            ratio_y = h - 60
            
            # Draw ratio bar background
            bar_x = 150
            bar_width = w - 300
            bar_height = 30
            cv2.rectangle(frame, (bar_x, ratio_y), (bar_x + bar_width, ratio_y + bar_height), (50, 50, 50), -1)
            
            # Draw threshold lines
            left_threshold_x = bar_x + int(bar_width * self.config.gaze_left_threshold)
            right_threshold_x = bar_x + int(bar_width * self.config.gaze_right_threshold)
            
            cv2.line(frame, (left_threshold_x, ratio_y), (left_threshold_x, ratio_y + bar_height), (0, 255, 255), 2)
            cv2.line(frame, (right_threshold_x, ratio_y), (right_threshold_x, ratio_y + bar_height), (255, 0, 255), 2)
            
            # Draw current ratio position
            if debug_info.get('smoothed_ratio') is not None:
                ratio = debug_info['smoothed_ratio']
                ratio_x = bar_x + int(bar_width * ratio)
                cv2.circle(frame, (ratio_x, ratio_y + bar_height // 2), 12, (0, 255, 0), -1)
                cv2.circle(frame, (ratio_x, ratio_y + bar_height // 2), 15, (255, 255, 255), 2)
            
            # Labels
            cv2.putText(frame, "LEFT", (bar_x - 60, ratio_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, "CENTER", (bar_x + bar_width // 2 - 35, ratio_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "RIGHT", (bar_x + bar_width + 10, ratio_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Debug info
            if debug_info.get('smoothed_ratio') is not None:
                cv2.putText(frame, f"Ratio: {debug_info['smoothed_ratio']:.3f}", (10, ratio_y + 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(frame, f"Confidence: {debug_info.get('confidence', 0)}", (10, ratio_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        except Exception as e:
            print(f"UI error: {e}")
        
        return frame
    
    def run(self):
        """Main loop"""
        print("System running... (Press ESC to quit)\n")
        self.running = True
        
        consecutive_errors = 0
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Capture frame
                ret, frame = self.webcam.read()
                if not ret:
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        print("✗ Too many frame errors")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                
                # Process frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Gaze tracking
                self.gaze_tracker.refresh(frame_rgb)
                gaze_direction = self.gaze_tracker.get_gaze_direction()
                debug_info = self.gaze_tracker.get_debug_info()
                
                # Get annotated frame
                annotated_rgb = self.gaze_tracker.get_annotated_frame()
                frame_display = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                
                # Hand detection
                timestamp_ms = int(time.time() * 1000)
                gesture, landmarks = self.hand_detector.detect(frame_rgb, timestamp_ms)
                frame_display = self.hand_detector.draw_hand(frame_display, landmarks)
                
                # Generate command
                command = self.command_controller.generate_command(gaze_direction, gesture)
                
                # Send to ESP32
                if self.esp32:
                    self.esp32.send(command)
                
                # Calculate FPS
                loop_time = time.time() - loop_start
                if loop_time > 0:
                    self.fps_history.append(1.0 / loop_time)
                fps = np.mean(self.fps_history) if self.fps_history else 0
                
                # Draw UI
                frame_display = self.draw_ui(frame_display, gaze_direction, gesture, command, fps, debug_info)
                
                # Display
                cv2.imshow("VisionDrive - Accurate Gaze Control", frame_display)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                elif key == ord('c') or key == ord('C'):
                    print("\nCalibration mode - adjust thresholds:")
                    print(f"Current: LEFT < {self.config.gaze_left_threshold}, RIGHT > {self.config.gaze_right_threshold}")
                    print("Look around and observe the green dot on the bar at the bottom")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted")
                break
            
            except Exception as e:
                consecutive_errors += 1
                print(f"✗ Error: {e}")
                if consecutive_errors >= 10:
                    break
                time.sleep(0.1)
        
        self.running = False
        self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        print("\nShutting down...")
        
        try:
            if self.esp32:
                self.esp32.send("STOP", force=True)
                time.sleep(0.2)
                self.esp32.close()
        except:
            pass
        
        try:
            if self.webcam and self.webcam.isOpened():
                self.webcam.release()
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
            time.sleep(0.5)
        except:
            pass
        
        print("✓ Shutdown complete")


# ==================== ENTRY POINT ====================
def main():
    """Entry point"""
    system = VisionDriveSystem()
    
    try:
        if not system.initialize():
            print("\n✗ Initialization failed")
            sys.exit(1)
        
        system.run()
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    
    except Exception as e:
        print(f"\n✗ Critical error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if system:
            system.cleanup()


if __name__ == "__main__":
    main()
