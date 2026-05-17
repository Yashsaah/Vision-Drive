"""
VisionDrive Diagnostic Tool
Run this to test your system and identify issues
"""

import cv2
import sys
import os
import socket
import time


class DiagnosticTool:
    """System diagnostic and testing utility"""
    
    def __init__(self):
        self.results = {}
        
    def run_all_tests(self):
        """Run complete diagnostic suite"""
        print("\n" + "="*60)
        print("VISIONDRIVE SYSTEM DIAGNOSTICS")
        print("="*60 + "\n")
        
        tests = [
            ("Python Version", self.test_python_version),
            ("Camera Access", self.test_camera),
            ("Camera Properties", self.test_camera_properties),
            ("GazeTracking Library", self.test_gaze_tracking),
            ("MediaPipe Library", self.test_mediapipe),
            ("Hand Model File", self.test_hand_model),
            ("Network Connectivity", self.test_network),
            ("ESP32 Connection", self.test_esp32_connection),
        ]
        
        for name, test_func in tests:
            print(f"\n[Testing] {name}...")
            try:
                result = test_func()
                self.results[name] = result
                if result:
                    print(f"  ✓ PASS")
                else:
                    print(f"  ✗ FAIL")
            except Exception as e:
                self.results[name] = False
                print(f"  ✗ ERROR: {e}")
        
        self.print_summary()
        
    def test_python_version(self):
        """Check Python version"""
        version = sys.version_info
        print(f"  Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major >= 3 and version.minor >= 7:
            return True
        else:
            print(f"  ⚠ Warning: Python 3.7+ recommended")
            return False
    
    def test_camera(self):
        """Test camera access"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(f"  ✗ Cannot open camera 0")
            # Try other indices
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"  ℹ Camera found at index {i}")
                    cap.release()
                    return True
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  Resolution: {w}x{h}")
            return True
        else:
            print(f"  ✗ Cannot read from camera")
            return False
    
    def test_camera_properties(self):
        """Test camera property settings"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        # Test setting properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Configured: {actual_w}x{actual_h} @ {actual_fps}fps")
        
        cap.release()
        return True
    
    def test_gaze_tracking(self):
        """Test gaze tracking library"""
        try:
            from gaze_tracking import GazeTracking
            gaze = GazeTracking()
            print(f"  GazeTracking module loaded")
            return True
        except ImportError as e:
            print(f"  ✗ Import failed: {e}")
            print(f"  Run: pip install gaze-tracking")
            return False
    
    def test_mediapipe(self):
        """Test MediaPipe library"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            print(f"  MediaPipe version: {mp.__version__}")
            return True
        except ImportError as e:
            print(f"  ✗ Import failed: {e}")
            print(f"  Run: pip install mediapipe")
            return False
    
    def test_hand_model(self):
        """Test hand model file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'hand_landmarker.task')
        
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  Model file found: {size_mb:.2f} MB")
            return True
        else:
            print(f"  ✗ Model file not found: {model_path}")
            print(f"  Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            return False
    
    def test_network(self):
        """Test basic network functionality"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.close()
            print(f"  UDP socket creation successful")
            return True
        except Exception as e:
            print(f"  ✗ Socket error: {e}")
            return False
    
    def test_esp32_connection(self):
        """Test ESP32 connection"""
        ESP32_IP = "192.168.4.1"
        PORT = 4210
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2)
            
            # Send test message
            test_msg = "TEST"
            sock.sendto(test_msg.encode(), (ESP32_IP, PORT))
            print(f"  Test packet sent to {ESP32_IP}:{PORT}")
            
            sock.close()
            
            # We can't receive confirmation with UDP, but no error means route exists
            print(f"  ℹ No routing errors (ESP32 may or may not be receiving)")
            return True
            
        except socket.timeout:
            print(f"  ⚠ No response from ESP32 (this is normal for UDP)")
            return True
        except Exception as e:
            print(f"  ✗ Connection error: {e}")
            print(f"  Check: ESP32 is powered on and in AP mode")
            print(f"  Check: Computer connected to ESP32's WiFi")
            return False
    
    def print_summary(self):
        """Print diagnostic summary"""
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        print(f"\nTests Passed: {passed}/{total}")
        
        if passed == total:
            print("\n✓ ALL TESTS PASSED - System ready!")
        else:
            print("\n⚠ ISSUES DETECTED:")
            for name, result in self.results.items():
                if not result:
                    print(f"  • {name}")
            
            print("\nRecommendations:")
            self.print_recommendations()
        
        print("\n" + "="*60 + "\n")
    
    def print_recommendations(self):
        """Print recommendations based on failures"""
        if not self.results.get("Camera Access"):
            print("  1. Check camera is connected and not in use")
            print("  2. Check camera permissions in system settings")
            print("  3. Try a different USB port")
        
        if not self.results.get("GazeTracking Library"):
            print("  1. Install: pip install gaze-tracking")
        
        if not self.results.get("MediaPipe Library"):
            print("  1. Install: pip install mediapipe")
        
        if not self.results.get("Hand Model File"):
            print("  1. Download hand_landmarker.task model file")
            print("  2. Place in same directory as scripts")
        
        if not self.results.get("ESP32 Connection"):
            print("  1. Ensure ESP32 is powered on")
            print("  2. Connect to ESP32's WiFi network")
            print("  3. Verify ESP32_IP in code matches actual IP")


class InteractiveTest:
    """Interactive component testing"""
    
    @staticmethod
    def test_gaze_live():
        """Live gaze tracking test"""
        print("\n" + "="*60)
        print("LIVE GAZE TRACKING TEST")
        print("="*60)
        print("\nInstructions:")
        print("  • Look at different parts of the screen")
        print("  • Check if gaze direction is detected correctly")
        print("  • Press ESC to exit")
        print("\n")
        
        try:
            from gaze_tracking import GazeTracking
            gaze = GazeTracking()
            webcam = cv2.VideoCapture(0)
            
            while True:
                ret, frame = webcam.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                gaze.refresh(frame_rgb)
                frame_display = cv2.cvtColor(gaze.annotated_frame(), cv2.COLOR_RGB2BGR)
                
                h_ratio = gaze.horizontal_ratio()
                v_ratio = gaze.vertical_ratio()
                
                # Display ratios
                if h_ratio is not None:
                    cv2.putText(frame_display, f"H-Ratio: {h_ratio:.3f}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if v_ratio is not None:
                    cv2.putText(frame_display, f"V-Ratio: {v_ratio:.3f}", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display state
                if gaze.is_blinking():
                    state = "BLINKING"
                elif gaze.is_left():
                    state = "LEFT"
                elif gaze.is_right():
                    state = "RIGHT"
                else:
                    state = "CENTER"
                
                cv2.putText(frame_display, f"State: {state}", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                cv2.imshow("Gaze Test", frame_display)
                
                if cv2.waitKey(1) == 27:
                    break
            
            webcam.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error: {e}")
    
    @staticmethod
    def test_hand_live():
        """Live hand tracking test"""
        print("\n" + "="*60)
        print("LIVE HAND TRACKING TEST")
        print("="*60)
        print("\nInstructions:")
        print("  • Show your hand to camera")
        print("  • Raise index finger (should detect DRIVE)")
        print("  • Raise middle finger (should detect REVERSE)")
        print("  • Press ESC to exit")
        print("\n")
        
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Load model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'hand_landmarker.task')
            
            with open(model_path, "rb") as f:
                model_data = f.read()
            
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=model_data),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1
            )
            
            hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            webcam = cv2.VideoCapture(0)
            
            while True:
                ret, frame = webcam.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect hand
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                gear = "NEUTRAL"
                if hand_result.hand_landmarks:
                    landmarks = hand_result.hand_landmarks[0]
                    
                    # Check fingers
                    index_raised = landmarks[8].y < landmarks[5].y - 0.12
                    middle_raised = landmarks[12].y < landmarks[9].y - 0.12
                    
                    if index_raised and not middle_raised:
                        gear = "DRIVE"
                    elif middle_raised and not index_raised:
                        gear = "REVERSE"
                    
                    # Draw landmarks
                    for lm in landmarks:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # Display gear
                color = (0, 255, 0) if gear != "NEUTRAL" else (0, 0, 255)
                cv2.putText(frame, f"GEAR: {gear}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                cv2.imshow("Hand Test", frame)
                
                if cv2.waitKey(1) == 27:
                    break
            
            webcam.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main diagnostic menu"""
    print("\nVisionDrive Diagnostic Tool")
    print("="*60)
    print("\n1. Run System Diagnostics")
    print("2. Test Gaze Tracking (Live)")
    print("3. Test Hand Tracking (Live)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        diag = DiagnosticTool()
        diag.run_all_tests()
    elif choice == "2":
        InteractiveTest.test_gaze_live()
    elif choice == "3":
        InteractiveTest.test_hand_live()
    elif choice == "4":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
