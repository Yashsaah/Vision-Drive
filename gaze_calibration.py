"""
Gaze Calibration Utility
Use this to find optimal gaze thresholds for your setup
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np
from collections import deque


class GazeCalibrator:
    """Interactive gaze calibration tool"""
    
    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        
        # Calibration data storage
        self.center_samples = deque(maxlen=100)
        self.left_samples = deque(maxlen=100)
        self.right_samples = deque(maxlen=100)
        
        self.current_mode = "CENTER"
        self.modes = ["CENTER", "LEFT", "RIGHT", "TEST"]
        self.mode_index = 0
        
    def run(self):
        """Run calibration process"""
        print("\n" + "="*60)
        print("GAZE CALIBRATION UTILITY")
        print("="*60)
        print("\nInstructions:")
        print("  1. Position yourself comfortably in front of camera")
        print("  2. Follow on-screen instructions for each mode")
        print("  3. Press SPACE to move to next calibration step")
        print("  4. Press ESC when done to see results")
        print("\n" + "="*60 + "\n")
        
        while True:
            ret, frame = self.webcam.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process gaze
            self.gaze.refresh(frame_rgb)
            annotated = cv2.cvtColor(self.gaze.annotated_frame(), cv2.COLOR_RGB2BGR)
            
            # Get current ratio
            h_ratio = self.gaze.horizontal_ratio()
            v_ratio = self.gaze.vertical_ratio()
            
            # Record samples based on current mode
            if h_ratio is not None:
                if self.current_mode == "CENTER":
                    self.center_samples.append(h_ratio)
                elif self.current_mode == "LEFT":
                    self.left_samples.append(h_ratio)
                elif self.current_mode == "RIGHT":
                    self.right_samples.append(h_ratio)
            
            # Draw UI
            annotated = self.draw_calibration_ui(annotated, h_ratio, v_ratio)
            
            cv2.imshow("Gaze Calibration", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                self.next_mode()
        
        self.show_results()
        self.cleanup()
    
    def draw_calibration_ui(self, frame, h_ratio, v_ratio):
        """Draw calibration UI"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Mode indicator
        mode_text = f"MODE: {self.current_mode}"
        cv2.putText(frame, mode_text, (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
        
        # Instructions
        instructions = {
            "CENTER": "Look straight at the camera",
            "LEFT": "Look to your LEFT (screen right)",
            "RIGHT": "Look to your RIGHT (screen left)",
            "TEST": "Test your calibration - look around"
        }
        
        cv2.putText(frame, instructions[self.current_mode], (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Sample count
        sample_counts = {
            "CENTER": len(self.center_samples),
            "LEFT": len(self.left_samples),
            "RIGHT": len(self.right_samples)
        }
        
        cv2.putText(frame, f"Samples: {sample_counts.get(self.current_mode, 0)}/100",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current ratios
        if h_ratio is not None:
            cv2.putText(frame, f"H-Ratio: {h_ratio:.3f}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if v_ratio is not None:
            cv2.putText(frame, f"V-Ratio: {v_ratio:.3f}", (250, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Bottom instructions
        cv2.putText(frame, "Press SPACE for next step | ESC to finish",
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Visual indicator for gaze direction in TEST mode
        if self.current_mode == "TEST" and h_ratio is not None:
            center_x = w // 2
            center_y = h // 2
            
            # Calculate gaze point
            gaze_x = int(center_x + (h_ratio - 0.5) * 400)
            gaze_y = center_y
            
            # Draw indicator
            cv2.circle(frame, (gaze_x, gaze_y), 20, (0, 0, 255), -1)
            cv2.circle(frame, (gaze_x, gaze_y), 25, (0, 255, 255), 2)
        
        return frame
    
    def next_mode(self):
        """Move to next calibration mode"""
        self.mode_index = (self.mode_index + 1) % len(self.modes)
        self.current_mode = self.modes[self.mode_index]
        print(f"\n→ Switched to {self.current_mode} mode")
    
    def show_results(self):
        """Display calibration results"""
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        
        if len(self.center_samples) > 0:
            center_avg = np.mean(self.center_samples)
            center_std = np.std(self.center_samples)
            print(f"\nCENTER:")
            print(f"  Average: {center_avg:.3f}")
            print(f"  Std Dev: {center_std:.3f}")
        
        if len(self.left_samples) > 0:
            left_avg = np.mean(self.left_samples)
            left_std = np.std(self.left_samples)
            print(f"\nLEFT:")
            print(f"  Average: {left_avg:.3f}")
            print(f"  Std Dev: {left_std:.3f}")
        
        if len(self.right_samples) > 0:
            right_avg = np.mean(self.right_samples)
            right_std = np.std(self.right_samples)
            print(f"\nRIGHT:")
            print(f"  Average: {right_avg:.3f}")
            print(f"  Std Dev: {right_std:.3f}")
        
        # Calculate recommended thresholds
        if len(self.center_samples) > 0 and len(self.left_samples) > 0:
            # LEFT threshold: midpoint between center and left averages
            left_threshold = (center_avg + left_avg) / 2
            print(f"\n{'='*60}")
            print("RECOMMENDED THRESHOLDS:")
            print(f"{'='*60}")
            print(f"\nGAZE_LEFT_THRESHOLD  = {left_threshold:.3f}")
        
        if len(self.center_samples) > 0 and len(self.right_samples) > 0:
            # RIGHT threshold: midpoint between center and right averages
            right_threshold = (center_avg + right_avg) / 2
            print(f"GAZE_RIGHT_THRESHOLD = {right_threshold:.3f}")
        
        print("\nUpdate these values in your Config class!")
        print("="*60 + "\n")
    
    def cleanup(self):
        """Clean up resources"""
        self.webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrator = GazeCalibrator()
    calibrator.run()
