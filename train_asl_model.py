#!/usr/bin/env python3
"""
ASL Model Training Script
Interactive script to train the ASL recognition model with custom data.
"""

import cv2
import numpy as np
import time
import os
from hand_tracker import HandTracker
from asl_detector import ASLDetector

class ASLModelTrainer:
    """Interactive trainer for ASL recognition model"""
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize the ASL model trainer
        
        Args:
            camera_index: Index of the camera to use
        """
        self.camera_index = camera_index
        self.cap = None
        self.hand_tracker = None
        self.asl_detector = ASLDetector()
        self.training_data = {}
        self.current_sign = ""
        self.samples_per_sign = 20
        self.current_sample = 0
        
    def initialize(self) -> bool:
        """Initialize the trainer"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Initialize hand tracker
            self.hand_tracker = HandTracker(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            print("ASL Model Trainer initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing trainer: {e}")
            return False
    
    def run_training_session(self) -> None:
        """Run the interactive training session"""
        if not self.initialize():
            return
        
        print("\n=== ASL Model Training ===")
        print("This script will help you train the ASL recognition model.")
        print("You'll be prompted to sign different ASL signs multiple times.")
        print("\nControls:")
        print("  'c' - Capture current hand position")
        print("  'n' - Next sign")
        print("  's' - Save and train model")
        print("  'q' - Quit without saving")
        print("\nPress any key to start...")
        cv2.waitKey(0)
        
        # Define signs to train
        signs_to_train = [
            "HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES", "NO",
            "GOOD", "BAD", "LOVE", "HELP", "WATER", "FOOD",
            "MOTHER", "FATHER", "SISTER", "BROTHER", "FRIEND", "FAMILY"
        ]
        
        for sign in signs_to_train:
            self.train_sign(sign)
            if not self.continue_training():
                break
        
        # Train the model
        if self.training_data:
            self.train_model()
        
        self.cleanup()
    
    def train_sign(self, sign: str) -> None:
        """Train a specific ASL sign"""
        self.current_sign = sign
        self.current_sample = 0
        
        print(f"\n=== Training Sign: {sign} ===")
        print(f"Please sign '{sign}' {self.samples_per_sign} times.")
        print("Press 'c' to capture each sample, 'n' to skip this sign, 'q' to quit.")
        
        if sign not in self.training_data:
            self.training_data[sign] = []
        
        while self.current_sample < self.samples_per_sign:
            success, frame = self.cap.read()
            if not success:
                continue
            
            # Process frame
            annotated_frame, hand_landmarks_list = self.hand_tracker.process_frame(frame)
            
            # Draw training info
            self.draw_training_info(annotated_frame, sign)
            
            # Display frame
            cv2.imshow("ASL Training", annotated_frame)
            
            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') or key == ord('C'):
                self.capture_sample(hand_landmarks_list)
            elif key == ord('n') or key == ord('N'):
                print(f"Skipping {sign}")
                break
            elif key == ord('q') or key == ord('Q'):
                return
    
    def capture_sample(self, hand_landmarks_list) -> None:
        """Capture a training sample"""
        if not hand_landmarks_list or len(hand_landmarks_list) == 0:
            print("No hand detected. Please make sure your hand is visible.")
            return
        
        landmarks = np.array(hand_landmarks_list[0])
        features = self.hand_tracker.extract_features(landmarks)
        
        if features.size > 0:
            self.training_data[self.current_sign].append(features)
            self.current_sample += 1
            print(f"Captured sample {self.current_sample}/{self.samples_per_sign} for {self.current_sign}")
        else:
            print("Could not extract features. Please try again.")
    
    def draw_training_info(self, frame: np.ndarray, sign: str) -> None:
        """Draw training information on frame"""
        height, width = frame.shape[:2]
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        alpha = 0.7
        frame[10:120, 10:width-10] = cv2.addWeighted(
            frame[10:120, 10:width-10], 1-alpha,
            overlay[10:120, 10:width-10], alpha, 0
        )
        
        # Draw text
        sign_text = f"Training: {sign}"
        progress_text = f"Sample: {self.current_sample}/{self.samples_per_sign}"
        instruction_text = "Press 'c' to capture, 'n' for next, 'q' to quit"
        
        cv2.putText(frame, sign_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, progress_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, instruction_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def continue_training(self) -> bool:
        """Ask user if they want to continue training"""
        print(f"\nCompleted training for {self.current_sign}")
        print("Press 'y' to continue with next sign, 'n' to finish training, 'q' to quit")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y') or key == ord('Y'):
                return True
            elif key == ord('n') or key == ord('N'):
                return False
            elif key == ord('q') or key == ord('Q'):
                return False
    
    def train_model(self) -> None:
        """Train the ASL model with collected data"""
        print("\n=== Training Model ===")
        
        # Prepare training data
        all_features = []
        all_labels = []
        
        for sign, features_list in self.training_data.items():
            if len(features_list) > 0:
                all_features.extend(features_list)
                all_labels.extend([sign] * len(features_list))
                print(f"Added {len(features_list)} samples for {sign}")
        
        if not all_features:
            print("No training data available!")
            return
        
        # Add data to detector
        for features, label in zip(all_features, all_labels):
            self.asl_detector.add_training_data(features, label)
        
        # Train the model
        print("Training model...")
        self.asl_detector.train_model()
        
        print(f"Model trained successfully with {len(set(all_labels))} different signs!")
        print("You can now use the main ASL Learning App with your trained model.")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        
        if self.hand_tracker:
            self.hand_tracker.release()
        
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("ASL Model Training Script")
    print("========================")
    
    # Check if camera is available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No camera found. Please connect a camera and try again.")
        return
    cap.release()
    
    # Create and run trainer
    trainer = ASLModelTrainer(camera_index=0)
    trainer.run_training_session()

if __name__ == "__main__":
    main()
