import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Optional
import json

class ASLDetector:
    """ASL sign detection and recognition system"""
    
    def __init__(self, model_path: str = "asl_model.pkl", scaler_path: str = "asl_scaler.pkl", training_data_path: str = "training_data.json"):
        """
        Initialize the ASL detector
        
        Args:
            model_path: Path to save/load the trained model
            scaler_path: Path to save/load the feature scaler
            training_data_path: Path to save/load training data
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.training_data_path = training_data_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_to_sign = {}
        self.sign_to_label = {}
        self.is_trained = False
        
        # Initialize training data storage
        self.training_features = []
        self.training_labels = []
        
        # Load existing model and training data if available
        self.load_model()
        self.load_training_data()
    
    def add_training_data(self, features: np.ndarray, label: str) -> None:
        """
        Add training data for a specific ASL sign
        
        Args:
            features: Feature vector from hand landmarks
            label: ASL sign label (e.g., "A", "B", "HELLO")
        """
        self.training_features.append(features.tolist())  # Convert to list for JSON serialization
        self.training_labels.append(label)
        
        # Save training data immediately
        self.save_training_data()
    
    def train_model(self) -> None:
        """Train the ASL recognition model"""
        if len(self.training_features) == 0:
            print("No training data available. Please add training data first.")
            return
        
        # Convert to numpy arrays
        X = np.array(self.training_features)
        y = np.array(self.training_labels)
        
        # Create label mappings
        unique_labels = np.unique(y)
        self.label_to_sign = {i: label for i, label in enumerate(unique_labels)}
        self.sign_to_label = {label: i for i, label in enumerate(unique_labels)}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Save model and scaler
        self.save_model()
        
        print(f"Model trained successfully with {len(unique_labels)} signs: {list(unique_labels)}")
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict ASL sign from hand features
        
        Args:
            features: Feature vector from hand landmarks
            
        Returns:
            Tuple of (predicted_sign, confidence)
        """
        if not self.is_trained:
            return "No model trained", 0.0
        
        if features.size == 0:
            return "No hand detected", 0.0
        
        # Reshape for single prediction
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled))
        
        return prediction, confidence
    
    def save_model(self) -> None:
        """Save the trained model and scaler"""
        if self.is_trained:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # Save label mappings
            mappings = {
                'label_to_sign': self.label_to_sign,
                'sign_to_label': self.sign_to_label
            }
            with open('asl_mappings.json', 'w') as f:
                json.dump(mappings, f)
    
    def load_model(self) -> None:
        """Load existing model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                # Load label mappings
                if os.path.exists('asl_mappings.json'):
                    with open('asl_mappings.json', 'r') as f:
                        mappings = json.load(f)
                        self.label_to_sign = mappings['label_to_sign']
                        self.sign_to_label = mappings['sign_to_label']
                
                self.is_trained = True
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.is_trained = False
    
    def save_training_data(self) -> None:
        """Save training data to file"""
        try:
            training_data = {
                'features': self.training_features,
                'labels': self.training_labels
            }
            with open(self.training_data_path, 'w') as f:
                json.dump(training_data, f)
        except Exception as e:
            print(f"Error saving training data: {e}")
    
    def load_training_data(self) -> None:
        """Load training data from file"""
        if os.path.exists(self.training_data_path):
            try:
                with open(self.training_data_path, 'r') as f:
                    training_data = json.load(f)
                    self.training_features = training_data.get('features', [])
                    self.training_labels = training_data.get('labels', [])
                print(f"Loaded {len(self.training_features)} training samples")
            except Exception as e:
                print(f"Error loading training data: {e}")
                self.training_features = []
                self.training_labels = []

class FingerspellingDetector:
    """Fingerspelling recognition for individual letters A-Z"""
    
    def __init__(self):
        """Initialize fingerspelling detector with predefined letter patterns"""
        self.letter_patterns = self._create_letter_patterns()
    
    def _create_letter_patterns(self) -> Dict[str, Dict]:
        """
        Create predefined patterns for ASL fingerspelling letters A-Z
        
        Returns:
            Dictionary mapping letters to their hand configurations
        """
        patterns = {}
        
        # Letter A: Fist with thumb on side
        patterns['A'] = {
            'extended_fingers': 0,
            'thumb_position': 'side',
            'hand_closed': True,
            'fingers_touching': False
        }
        
        # Letter B: All fingers extended, thumb tucked
        patterns['B'] = {
            'extended_fingers': 4,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False
        }
        
        # Letter C: Curved hand like C shape
        patterns['C'] = {
            'extended_fingers': 5,
            'thumb_position': 'extended',
            'hand_curved': True,
            'fingers_touching': False
        }
        
        # Letter D: Index finger extended, others tucked
        patterns['D'] = {
            'extended_fingers': 1,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_finger': 'index'
        }
        
        # Letter E: All fingers tucked, thumb on side
        patterns['E'] = {
            'extended_fingers': 0,
            'thumb_position': 'side',
            'hand_closed': True,
            'fingers_touching': False
        }
        
        # Letter F: Index and thumb touching, others extended
        patterns['F'] = {
            'extended_fingers': 3,
            'thumb_position': 'touching_index',
            'hand_closed': False,
            'fingers_touching': False,
            'thumb_index_touch': True
        }
        
        # Letter G: Index finger pointing, others tucked
        patterns['G'] = {
            'extended_fingers': 1,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_finger': 'index'
        }
        
        # Letter H: Index and middle fingers extended, others tucked
        patterns['H'] = {
            'extended_fingers': 2,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'middle']
        }
        
        # Letter I: Pinky extended, others tucked
        patterns['I'] = {
            'extended_fingers': 1,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_finger': 'pinky'
        }
        
        # Letter J: Pinky extended, others tucked, with J motion
        patterns['J'] = {
            'extended_fingers': 1,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_finger': 'pinky',
            'motion': 'J_curve'
        }
        
        # Letter K: Index and thumb extended, others tucked
        patterns['K'] = {
            'extended_fingers': 2,
            'thumb_position': 'extended',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'thumb']
        }
        
        # Letter L: Index and thumb extended, others tucked
        patterns['L'] = {
            'extended_fingers': 2,
            'thumb_position': 'extended',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'thumb']
        }
        
        # Letter M: Thumb between ring and pinky, others tucked
        patterns['M'] = {
            'extended_fingers': 0,
            'thumb_position': 'between_ring_pinky',
            'hand_closed': True,
            'fingers_touching': False
        }
        
        # Letter N: Thumb between middle and ring, others tucked
        patterns['N'] = {
            'extended_fingers': 0,
            'thumb_position': 'between_middle_ring',
            'hand_closed': True,
            'fingers_touching': False
        }
        
        # Letter O: All fingers curled to form O
        patterns['O'] = {
            'extended_fingers': 5,
            'thumb_position': 'extended',
            'hand_curved': True,
            'fingers_touching': True
        }
        
        # Letter P: Index and middle fingers extended down, thumb up
        patterns['P'] = {
            'extended_fingers': 3,
            'thumb_position': 'up',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'middle', 'thumb']
        }
        
        # Letter Q: Index and thumb extended, others tucked
        patterns['Q'] = {
            'extended_fingers': 2,
            'thumb_position': 'extended',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'thumb']
        }
        
        # Letter R: Index and middle fingers crossed
        patterns['R'] = {
            'extended_fingers': 2,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'middle'],
            'fingers_crossed': True
        }
        
        # Letter S: Fist with thumb over fingers
        patterns['S'] = {
            'extended_fingers': 0,
            'thumb_position': 'over_fingers',
            'hand_closed': True,
            'fingers_touching': False
        }
        
        # Letter T: Thumb between index and middle
        patterns['T'] = {
            'extended_fingers': 0,
            'thumb_position': 'between_index_middle',
            'hand_closed': True,
            'fingers_touching': False
        }
        
        # Letter U: Index and middle fingers extended, others tucked
        patterns['U'] = {
            'extended_fingers': 2,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'middle']
        }
        
        # Letter V: Index and middle fingers extended apart
        patterns['V'] = {
            'extended_fingers': 2,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'middle'],
            'fingers_spread': True
        }
        
        # Letter W: Index, middle, and ring fingers extended
        patterns['W'] = {
            'extended_fingers': 3,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['index', 'middle', 'ring']
        }
        
        # Letter X: Index finger bent
        patterns['X'] = {
            'extended_fingers': 0,
            'thumb_position': 'tucked',
            'hand_closed': True,
            'fingers_touching': False,
            'index_bent': True
        }
        
        # Letter Y: Thumb and pinky extended
        patterns['Y'] = {
            'extended_fingers': 2,
            'thumb_position': 'extended',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_fingers_list': ['thumb', 'pinky']
        }
        
        # Letter Z: Index finger drawing Z shape
        patterns['Z'] = {
            'extended_fingers': 1,
            'thumb_position': 'tucked',
            'hand_closed': False,
            'fingers_touching': False,
            'extended_finger': 'index',
            'motion': 'Z_shape'
        }
        
        return patterns
    
    def detect_fingerspelling(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Detect fingerspelled letter from hand landmarks
        
        Args:
            landmarks: Array of hand landmarks (21, 3)
            
        Returns:
            Tuple of (detected_letter, confidence)
        """
        if landmarks.size == 0:
            return "No hand detected", 0.0
        
        # Analyze hand configuration
        hand_config = self._analyze_hand_configuration(landmarks)
        
        # Compare with known patterns
        best_match = "Unknown"
        best_confidence = 0.0
        
        for letter, pattern in self.letter_patterns.items():
            confidence = self._calculate_pattern_match(hand_config, pattern)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = letter
        
        return best_match, best_confidence
    
    def _analyze_hand_configuration(self, landmarks: np.ndarray) -> Dict:
        """
        Analyze hand configuration from landmarks
        
        Args:
            landmarks: Array of hand landmarks (21, 3)
            
        Returns:
            Dictionary describing hand configuration
        """
        config = {}
        
        # Define fingertip indices
        fingertips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        # Define joint indices for each finger
        finger_joints = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        # Check if fingers are extended
        extended_fingers = []
        for finger, tip_idx in fingertips.items():
            joints = finger_joints[finger]
            extended = self._is_finger_extended(landmarks, joints)
            if extended:
                extended_fingers.append(finger)
        
        config['extended_fingers'] = len(extended_fingers)
        config['extended_fingers_list'] = extended_fingers
        
        # Check specific finger extensions
        for finger in fingertips.keys():
            config[f"{finger}_extended"] = finger in extended_fingers
        
        # Check thumb position
        config['thumb_position'] = self._analyze_thumb_position(landmarks)
        
        # Check for special configurations
        config['fingers_touching'] = self._check_fingers_touching(landmarks)
        config['hand_curved'] = self._check_hand_curved(landmarks)
        config['fingers_crossed'] = self._check_fingers_crossed(landmarks)
        config['fingers_spread'] = self._check_fingers_spread(landmarks)
        config['hand_closed'] = len(extended_fingers) == 0
        
        # Check for specific finger configurations
        if 'index' in extended_fingers and 'thumb' in extended_fingers:
            config['thumb_index_touch'] = self._check_thumb_index_touch(landmarks)
        else:
            config['thumb_index_touch'] = False
        
        # Check for bent index finger (for X)
        config['index_bent'] = self._check_index_bent(landmarks)
        
        return config
    
    def _is_finger_extended(self, landmarks: np.ndarray, joints: List[int]) -> bool:
        """Check if a finger is extended based on joint angles"""
        if len(joints) < 3:
            return False
        
        # Calculate angles between joints
        angles = []
        for i in range(len(joints) - 2):
            p1 = landmarks[joints[i]]
            p2 = landmarks[joints[i + 1]]
            p3 = landmarks[joints[i + 2]]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # Finger is extended if angles are close to 180 degrees
        avg_angle = np.mean(angles)
        return avg_angle > 2.5  # Approximately 143 degrees
    
    def _analyze_thumb_position(self, landmarks: np.ndarray) -> str:
        """Analyze thumb position relative to other fingers"""
        thumb_tip = landmarks[4]
        wrist = landmarks[0]
        
        # Calculate thumb position relative to palm
        thumb_vector = thumb_tip - wrist
        
        # Simple heuristic for thumb position
        if thumb_vector[0] > 0.1:  # Thumb to the right
            return 'side'
        elif thumb_vector[1] < -0.1:  # Thumb up
            return 'up'
        else:
            return 'tucked'
    
    def _check_fingers_touching(self, landmarks: np.ndarray) -> bool:
        """Check if fingertips are touching"""
        fingertips = [4, 8, 12, 16, 20]
        min_distance = 0.05  # Threshold for touching
        
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
                if dist < min_distance:
                    return True
        return False
    
    def _check_hand_curved(self, landmarks: np.ndarray) -> bool:
        """Check if hand is curved (like letter C)"""
        # Check if fingertips form a curve
        fingertips = landmarks[[4, 8, 12, 16, 20]]
        wrist = landmarks[0]
        
        # Calculate distances from wrist to fingertips
        distances = [np.linalg.norm(tip - wrist) for tip in fingertips]
        
        # Hand is curved if distances are similar (fingers form arc)
        return np.std(distances) < 0.05
    
    def _check_fingers_crossed(self, landmarks: np.ndarray) -> bool:
        """Check if fingers are crossed (like letter R)"""
        # Check if index and middle fingers are crossed
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        
        # Calculate if fingers cross
        index_vector = index_tip - index_mcp
        middle_vector = middle_tip - middle_mcp
        
        cross_product = np.cross(index_vector, middle_vector)
        return abs(cross_product[2]) > 0.1  # Z-component indicates crossing
    
    def _check_fingers_spread(self, landmarks: np.ndarray) -> bool:
        """Check if fingers are spread apart"""
        fingertips = landmarks[[8, 12, 16, 20]]  # Index, middle, ring, pinky
        
        # Calculate distances between adjacent fingertips
        distances = []
        for i in range(len(fingertips) - 1):
            dist = np.linalg.norm(fingertips[i] - fingertips[i + 1])
            distances.append(dist)
        
        # Fingers are spread if distances are large
        return np.mean(distances) > 0.08
    
    def _check_thumb_index_touch(self, landmarks: np.ndarray) -> bool:
        """Check if thumb and index finger are touching"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance < 0.05  # Threshold for touching
    
    def _check_index_bent(self, landmarks: np.ndarray) -> bool:
        """Check if index finger is bent (for letter X)"""
        # Check angle between index finger joints
        index_joints = landmarks[[5, 6, 7, 8]]  # MCP, PIP, DIP, tip
        
        # Calculate angles between joints
        angles = []
        for i in range(len(index_joints) - 2):
            p1 = index_joints[i]
            p2 = index_joints[i + 1]
            p3 = index_joints[i + 2]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # Index is bent if angles are small (less than 90 degrees)
        return np.mean(angles) < 1.57  # 90 degrees in radians
    
    def _calculate_pattern_match(self, hand_config: Dict, pattern: Dict) -> float:
        """Calculate how well hand configuration matches a pattern"""
        matches = 0
        total_checks = 0
        weight = 1.0
        
        # Check extended fingers count (most important)
        if 'extended_fingers' in pattern:
            total_checks += 2  # Higher weight
            if hand_config.get('extended_fingers', 0) == pattern['extended_fingers']:
                matches += 2
            else:
                # Partial credit for close matches
                diff = abs(hand_config.get('extended_fingers', 0) - pattern['extended_fingers'])
                if diff == 1:
                    matches += 1
        
        # Check thumb position
        if 'thumb_position' in pattern:
            total_checks += 1
            if hand_config.get('thumb_position') == pattern['thumb_position']:
                matches += 1
        
        # Check hand closed/open
        if 'hand_closed' in pattern:
            total_checks += 1
            if hand_config.get('hand_closed', False) == pattern['hand_closed']:
                matches += 1
        
        # Check fingers touching
        if 'fingers_touching' in pattern:
            total_checks += 1
            if hand_config.get('fingers_touching', False) == pattern['fingers_touching']:
                matches += 1
        
        # Check specific finger extensions
        if 'extended_finger' in pattern:
            total_checks += 1
            if hand_config.get(f"{pattern['extended_finger']}_extended", False):
                matches += 1
        
        # Check extended fingers list
        if 'extended_fingers_list' in pattern:
            total_checks += 1
            expected_fingers = set(pattern['extended_fingers_list'])
            actual_fingers = set(hand_config.get('extended_fingers_list', []))
            if expected_fingers == actual_fingers:
                matches += 1
            elif len(expected_fingers.intersection(actual_fingers)) > 0:
                matches += 0.5  # Partial credit
        
        # Check special configurations
        special_configs = ['hand_curved', 'fingers_crossed', 'fingers_spread', 
                          'thumb_index_touch', 'index_bent']
        for config in special_configs:
            if config in pattern:
                total_checks += 1
                if hand_config.get(config, False) == pattern[config]:
                    matches += 1
        
        if total_checks == 0:
            return 0.0
        
        return matches / total_checks
