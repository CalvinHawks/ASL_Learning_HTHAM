#!/usr/bin/env python3
"""
Example usage of the ASL Learning Application
This script demonstrates how to use the individual components.
"""

import cv2
import numpy as np
from hand_tracker import HandTracker
from asl_detector import FingerspellingDetector
from learning_interface import LearningInterface

def basic_hand_tracking_example():
    """Basic example of hand tracking with MediaPipe"""
    print("=== Basic Hand Tracking Example ===")
    print("This example shows basic hand tracking functionality.")
    print("Press 'q' to quit.")
    
    # Initialize hand tracker
    hand_tracker = HandTracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting camera...")
    
    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        # Process frame
        annotated_frame, hand_landmarks_list = hand_tracker.process_frame(frame)
        
        # Display frame
        cv2.imshow("Hand Tracking Example", annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    hand_tracker.release()
    cv2.destroyAllWindows()
    print("Hand tracking example completed!")

def fingerspelling_example():
    """Example of fingerspelling recognition"""
    print("\n=== Fingerspelling Recognition Example ===")
    print("This example shows fingerspelling recognition.")
    print("Try signing letters A-Z. Press 'q' to quit.")
    
    # Initialize components
    hand_tracker = HandTracker()
    fingerspelling_detector = FingerspellingDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting camera...")
    
    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        # Process frame
        annotated_frame, hand_landmarks_list = hand_tracker.process_frame(frame)
        
        # Detect fingerspelling if hand is present
        if hand_landmarks_list and len(hand_landmarks_list) > 0:
            landmarks = np.array(hand_landmarks_list[0])
            detected_letter, confidence = fingerspelling_detector.detect_fingerspelling(landmarks)
            
            # Draw detection result
            if detected_letter and confidence > 0.6:
                cv2.putText(annotated_frame, f"Letter: {detected_letter}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Fingerspelling Example", annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    hand_tracker.release()
    cv2.destroyAllWindows()
    print("Fingerspelling example completed!")

def learning_interface_example():
    """Example of the learning interface"""
    print("\n=== Learning Interface Example ===")
    print("This example shows the learning interface.")
    print("Press 'n' for next lesson, 'r' to restart, 'q' to quit.")
    
    # Initialize components
    hand_tracker = HandTracker()
    learning_interface = LearningInterface()
    
    # Start a lesson
    learning_interface.start_lesson(0)  # Start with fingerspelling lesson
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting camera...")
    
    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        # Process frame
        annotated_frame, hand_landmarks_list = hand_tracker.process_frame(frame)
        
        # Draw learning interface
        annotated_frame = learning_interface.draw_interface(annotated_frame)
        
        # Display frame
        cv2.imshow("Learning Interface Example", annotated_frame)
        
        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC key
            break
        elif key != 255:  # Key was pressed
            learning_interface.handle_keypress(key)
    
    # Cleanup
    cap.release()
    hand_tracker.release()
    cv2.destroyAllWindows()
    print("Learning interface example completed!")

def main():
    """Main function to run examples"""
    print("ASL Learning Application - Examples")
    print("===================================")
    print("Choose an example to run:")
    print("1. Basic hand tracking")
    print("2. Fingerspelling recognition")
    print("3. Learning interface")
    print("4. Run all examples")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            basic_hand_tracking_example()
        elif choice == '2':
            fingerspelling_example()
        elif choice == '3':
            learning_interface_example()
        elif choice == '4':
            basic_hand_tracking_example()
            fingerspelling_example()
            learning_interface_example()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
