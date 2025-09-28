import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class HandTracker:
    """Hand tracking class using MediaPipe for ASL recognition"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        """
        Initialize the hand tracker
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Process a single frame and return annotated frame with hand landmarks
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, hand_landmarks_list)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        annotated_frame = frame.copy()
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_draw.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                hand_landmarks_list.append(landmarks)
        
        return annotated_frame, hand_landmarks_list
    
    def get_landmark_coordinates(self, hand_landmarks: List) -> np.ndarray:
        """
        Extract normalized landmark coordinates from hand landmarks
        
        Args:
            hand_landmarks: List of landmark coordinates
            
        Returns:
            Numpy array of shape (21, 3) with x, y, z coordinates
        """
        if not hand_landmarks:
            return np.array([])
        
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def calculate_angles(self, landmarks: np.ndarray) -> List[float]:
        """
        Calculate angles between key finger joints for ASL recognition
        
        Args:
            landmarks: Array of hand landmarks (21, 3)
            
        Returns:
            List of calculated angles
        """
        if landmarks.size == 0:
            return []
        
        angles = []
        
        # Define finger joint connections for angle calculation
        finger_joints = [
            # Thumb
            [0, 1, 2], [1, 2, 3], [2, 3, 4],
            # Index finger
            [0, 5, 6], [5, 6, 7], [6, 7, 8],
            # Middle finger
            [0, 9, 10], [9, 10, 11], [10, 11, 12],
            # Ring finger
            [0, 13, 14], [13, 14, 15], [14, 15, 16],
            # Pinky
            [0, 17, 18], [17, 18, 19], [18, 19, 20]
        ]
        
        for joint in finger_joints:
            if len(joint) == 3:
                p1, p2, p3 = landmarks[joint[0]], landmarks[joint[1]], landmarks[joint[2]]
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        return angles
    
    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract features from hand landmarks for ASL recognition
        
        Args:
            landmarks: Array of hand landmarks (21, 3)
            
        Returns:
            Feature vector for machine learning
        """
        if landmarks.size == 0:
            return np.array([])
        
        features = []
        
        # Add normalized landmark coordinates
        features.extend(landmarks.flatten())
        
        # Add calculated angles
        angles = self.calculate_angles(landmarks)
        features.extend(angles)
        
        # Add distances between key points
        key_distances = self.calculate_key_distances(landmarks)
        features.extend(key_distances)
        
        return np.array(features)
    
    def calculate_key_distances(self, landmarks: np.ndarray) -> List[float]:
        """
        Calculate distances between key hand landmarks
        
        Args:
            landmarks: Array of hand landmarks (21, 3)
            
        Returns:
            List of key distances
        """
        if landmarks.size == 0:
            return []
        
        distances = []
        
        # Distance from wrist to fingertips
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        
        for tip in fingertips:
            dist = np.linalg.norm(landmarks[tip] - wrist)
            distances.append(dist)
        
        # Distance between fingertips
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
                distances.append(dist)
        
        return distances
    
    def release(self):
        """Release resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
