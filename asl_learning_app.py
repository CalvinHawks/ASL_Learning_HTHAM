#!/usr/bin/env python3
"""
ASL Learning Application
A computer vision application for learning American Sign Language using MediaPipe hand tracking.
"""

import cv2
import numpy as np
import time
import sys
import os
from typing import Optional, Tuple

# Import our custom modules
from hand_tracker import HandTracker
from asl_detector import ASLDetector, FingerspellingDetector
from learning_interface import LearningInterface

class ASLLearningApp:
    """Main ASL Learning Application"""
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize the ASL Learning App
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.hand_tracker = None
        self.asl_detector = None
        self.fingerspelling_detector = None
        self.learning_interface = None
        self.running = False
        self.detection_mode = "fingerspelling"  # "fingerspelling", "asl", "auto"
        
        # Detection settings
        self.min_confidence = 0.6
        self.detection_interval = 0.5  # Seconds between detections
        self.last_detection_time = 0
        
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def initialize(self) -> bool:
        """
        Initialize all components of the application
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize hand tracker
            self.hand_tracker = HandTracker(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            # Initialize ASL detectors
            self.asl_detector = ASLDetector()
            self.fingerspelling_detector = FingerspellingDetector()
            
            # Initialize learning interface
            self.learning_interface = LearningInterface("ASL Learning App")
            
            print("ASL Learning App initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing application: {e}")
            return False
    
    def run(self) -> None:
        """Run the main application loop"""
        if not self.initialize():
            return
        
        print("\n=== ASL Learning App ===")
        print("Controls:")
        print("  'n' - Next lesson")
        print("  'r' - Restart current lesson")
        print("  'm' - Toggle detection mode (fingerspelling/ASL/auto)")
        print("  't' - Add training sample for current sign")
        print("  's' - Save progress and train model")
        print("  'l' - Load progress")
        print("  'q' or ESC - Quit")
        print("\nTraining Instructions:")
        print("  1. Start a lesson (fingerspelling works immediately)")
        print("  2. For ASL signs: Press 't' while signing each letter/word")
        print("  3. Collect 5+ samples per sign, then press 's' to train")
        print("  4. Switch to ASL mode with 'm' to use trained model")
        print("\nStarting camera...")
        
        # Start with the first lesson
        self.learning_interface.start_lesson(0)
        
        self.running = True
        while self.running:
            try:
                # Read frame from camera
                success, frame = self.cap.read()
                if not success:
                    print("Error: Could not read from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("ASL Learning App", processed_frame)
                
                # Handle keyboard input with longer wait time for better responsiveness
                key = cv2.waitKey(30) & 0xFF
                if key != 255:  # Key was pressed
                    self.handle_keypress(key)
                
                # Update FPS counter
                self.update_fps()
                
            except KeyboardInterrupt:
                print("\nApplication interrupted by user")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                break
        
        self.cleanup()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for ASL learning
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with annotations
        """
        # Track hands
        annotated_frame, hand_landmarks_list = self.hand_tracker.process_frame(frame)
        
        # Process hand detection if hand is present
        if hand_landmarks_list and len(hand_landmarks_list) > 0:
            landmarks = np.array(hand_landmarks_list[0])
            
            # Detect ASL sign based on current mode
            detected_sign, confidence = self.detect_sign(landmarks)
            
            # Check answer with learning interface
            if detected_sign and confidence > self.min_confidence:
                current_time = time.time()
                if current_time - self.last_detection_time > self.detection_interval:
                    self.learning_interface.check_answer(detected_sign, confidence)
                    self.last_detection_time = current_time
            
            # Draw detection info on frame
            self.draw_detection_info(annotated_frame, detected_sign, confidence)
        
        # Draw learning interface
        annotated_frame = self.learning_interface.draw_interface(annotated_frame)
        
        # Draw FPS
        self.draw_fps(annotated_frame)
        
        return annotated_frame
    
    def detect_sign(self, landmarks: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect ASL sign from hand landmarks based on current mode
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            Tuple of (detected_sign, confidence)
        """
        if self.detection_mode == "fingerspelling":
            return self.fingerspelling_detector.detect_fingerspelling(landmarks)
        elif self.detection_mode == "asl":
            features = self.hand_tracker.extract_features(landmarks)
            return self.asl_detector.predict(features)
        else:  # auto mode
            # Try fingerspelling first, then ASL
            sign, conf = self.fingerspelling_detector.detect_fingerspelling(landmarks)
            if conf > 0.7:
                return sign, conf
            else:
                features = self.hand_tracker.extract_features(landmarks)
                return self.asl_detector.predict(features)
    
    def draw_detection_info(self, frame: np.ndarray, detected_sign: str, confidence: float) -> None:
        """
        Draw detection information on the frame
        
        Args:
            frame: Frame to draw on
            detected_sign: Detected sign
            confidence: Detection confidence
        """
        if detected_sign and confidence > 0.3:
            # Draw detection box
            height, width = frame.shape[:2]
            box_width = 300
            box_height = 80
            x = width - box_width - 10
            y = 10
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
            alpha = 0.7
            frame[y:y+box_height, x:x+box_width] = cv2.addWeighted(
                frame[y:y+box_height, x:x+box_width], 1-alpha,
                overlay[y:y+box_height, x:x+box_width], alpha, 0
            )
            
            # Draw text
            sign_text = f"Detected: {detected_sign}"
            conf_text = f"Confidence: {confidence:.2f}"
            mode_text = f"Mode: {self.detection_mode}"
            
            cv2.putText(frame, sign_text, (x + 10, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (x + 10, y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, mode_text, (x + 10, y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_fps(self, frame: np.ndarray) -> None:
        """Draw FPS counter on frame"""
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def update_fps(self) -> None:
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    
    def handle_keypress(self, key: int) -> None:
        """
        Handle keyboard input
        
        Args:
            key: Key code from cv2.waitKey()
        """
        # Handle learning interface keys
        action = self.learning_interface.handle_keypress(key)
        
        if action == 'quit':
            self.running = False
            return
        
        # Handle additional app-specific keys
        if key == ord('m') or key == ord('M'):
            self.toggle_detection_mode()
        
        elif key == ord('t') or key == ord('T'):
            self.train_current_position()
        
        elif key == ord('s') or key == ord('S'):
            self.save_and_train_model()
        
        elif key == ord('l') or key == ord('L'):
            self.learning_interface.load_progress()
            print("Progress loaded!")
        
    
    def toggle_detection_mode(self) -> None:
        """Toggle between detection modes"""
        modes = ["fingerspelling", "asl", "auto"]
        current_index = modes.index(self.detection_mode)
        next_index = (current_index + 1) % len(modes)
        self.detection_mode = modes[next_index]
        print(f"Detection mode changed to: {self.detection_mode}")
    
    def train_current_position(self) -> None:
        """Train ASL model with current hand position"""
        # Get current hand landmarks
        success, frame = self.cap.read()
        if not success:
            print("Could not capture frame for training")
            return
        
        _, hand_landmarks_list = self.hand_tracker.process_frame(frame)
        
        if hand_landmarks_list and len(hand_landmarks_list) > 0:
            landmarks = np.array(hand_landmarks_list[0])
            features = self.hand_tracker.extract_features(landmarks)
            
            if features.size == 0:
                print("Could not extract features from hand position")
                return
            
            # Get target from current prompt
            target = self.learning_interface.get_current_target()
            if target:
                self.asl_detector.add_training_data(features, target)
                print(f"✓ Added training data for: {target}")
                print(f"  Features extracted: {len(features)} values")
                
                # Check if we have enough data to train
                if hasattr(self.asl_detector, 'training_features'):
                    total_samples = len(self.asl_detector.training_features)
                    print(f"  Total training samples: {total_samples}")
                    
                    if total_samples >= 5:  # Minimum samples to train
                        print("  Ready to train! Press 's' to save and train the model.")
                    else:
                        print(f"  Need {5 - total_samples} more samples to train.")
            else:
                print("No current target for training. Make sure you're in a lesson.")
        else:
            print("No hand detected for training. Make sure your hand is visible in the camera.")
    
    def save_and_train_model(self) -> None:
        """Save progress and train the ASL model"""
        # Save learning progress
        self.learning_interface.save_progress()
        print("Learning progress saved!")
        
        # Check if we have training data
        if not hasattr(self.asl_detector, 'training_features') or len(self.asl_detector.training_features) == 0:
            print("No training data available. Use 't' to add training samples first.")
            return
        
        # Train the model
        print("Training ASL model...")
        self.asl_detector.train_model()
        
        # Show training summary
        if hasattr(self.asl_detector, 'training_features'):
            total_samples = len(self.asl_detector.training_features)
            unique_signs = len(set(self.asl_detector.training_labels)) if hasattr(self.asl_detector, 'training_labels') else 0
            print(f"✓ Model trained successfully!")
            print(f"  Total samples: {total_samples}")
            print(f"  Unique signs: {unique_signs}")
            print("  You can now use ASL detection mode!")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        
        if self.hand_tracker:
            self.hand_tracker.release()
        
        cv2.destroyAllWindows()
        print("Application closed successfully!")

def main():
    """Main function to run the ASL Learning App"""
    print("ASL Learning Application")
    print("=======================")
    
    # Check if camera is available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No camera found. Please connect a camera and try again.")
        return
    cap.release()
    
    # Create and run the application
    app = ASLLearningApp(camera_index=0)
    app.run()

if __name__ == "__main__":
    main()
