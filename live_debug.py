#!/usr/bin/env python3
"""
Live Debug Tool
Real-time debugging of hand detection and feature extraction
"""

import cv2
import numpy as np
import time
from hand_tracker import HandTracker
from asl_detector import ASLDetector, FingerspellingDetector

class LiveDebugger:
    def __init__(self):
        self.hand_tracker = HandTracker()
        self.asl_detector = ASLDetector()
        self.fingerspelling_detector = FingerspellingDetector()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
    def run(self):
        print("=== Live ASL Debug Tool ===")
        print("Controls:")
        print("  '1' - Test fingerspelling detection")
        print("  '2' - Test ASL model detection")
        print("  '3' - Compare both detections")
        print("  't' - Add training sample for 'A'")
        print("  'r' - Retrain model")
        print("  'q' - Quit")
        
        mode = "compare"  # "fingerspelling", "asl", "compare"
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Process frame
            annotated_frame, hand_landmarks_list = self.hand_tracker.process_frame(frame)
            
            if hand_landmarks_list and len(hand_landmarks_list) > 0:
                landmarks = np.array(hand_landmarks_list[0])
                features = self.hand_tracker.extract_features(landmarks)
                
                # Test different detection methods
                if mode == "fingerspelling" or mode == "compare":
                    finger_sign, finger_conf = self.fingerspelling_detector.detect_fingerspelling(landmarks)
                    
                if mode == "asl" or mode == "compare":
                    asl_sign, asl_conf = self.asl_detector.predict(features)
                
                # Display results
                y_offset = 30
                if mode == "fingerspelling":
                    cv2.putText(annotated_frame, f"Fingerspelling: {finger_sign} ({finger_conf:.2f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif mode == "asl":
                    cv2.putText(annotated_frame, f"ASL Model: {asl_sign} ({asl_conf:.2f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:  # compare
                    cv2.putText(annotated_frame, f"Fingerspelling: {finger_sign} ({finger_conf:.2f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ASL Model: {asl_sign} ({asl_conf:.2f})", 
                               (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Show feature info
                    cv2.putText(annotated_frame, f"Features: {len(features)} values", 
                               (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                # Show training data info
                cv2.putText(annotated_frame, f"Training samples: {len(self.asl_detector.training_features)}", 
                           (10, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(annotated_frame, f"Model trained: {self.asl_detector.is_trained}", 
                           (10, annotated_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(annotated_frame, f"Available signs: {list(self.asl_detector.label_to_sign.values())}", 
                           (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Show current mode
            cv2.putText(annotated_frame, f"Mode: {mode}", (10, 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Live ASL Debug", annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                mode = "fingerspelling"
                print("Switched to fingerspelling mode")
            elif key == ord('2'):
                mode = "asl"
                print("Switched to ASL model mode")
            elif key == ord('3'):
                mode = "compare"
                print("Switched to comparison mode")
            elif key == ord('t'):
                if hand_landmarks_list and len(hand_landmarks_list) > 0:
                    landmarks = np.array(hand_landmarks_list[0])
                    features = self.hand_tracker.extract_features(landmarks)
                    self.asl_detector.add_training_data(features, 'A')
                    print(f"Added training sample for 'A'. Total samples: {len(self.asl_detector.training_features)}")
                else:
                    print("No hand detected for training")
            elif key == ord('r'):
                print("Retraining model...")
                self.asl_detector.train_model()
                print("Model retrained!")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    debugger = LiveDebugger()
    debugger.run()

if __name__ == "__main__":
    main()
