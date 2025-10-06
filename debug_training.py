#!/usr/bin/env python3
"""
Debug Training Pipeline
Test the training data loading and model prediction
"""

import json
import numpy as np
from asl_detector import ASLDetector
from hand_tracker import HandTracker

def debug_training_data():
    """Debug the training data structure"""
    print("=== Training Data Debug ===")
    
    try:
        with open('training_data.json', 'r') as f:
            data = json.load(f)
        
        features = data.get('features', [])
        labels = data.get('labels', [])
        
        print(f"Total samples: {len(features)}")
        print(f"Total labels: {len(labels)}")
        
        if features:
            print(f"First feature shape: {np.array(features[0]).shape}")
            print(f"Feature sample: {features[0][:5]}...")  # First 5 values
            
        if labels:
            print(f"Labels: {set(labels)}")
            
        # Check for consistency
        if len(features) != len(labels):
            print("ERROR: Features and labels length mismatch!")
            
    except Exception as e:
        print(f"Error reading training data: {e}")

def test_model_prediction():
    """Test model prediction with sample data"""
    print("\n=== Model Prediction Test ===")
    
    try:
        detector = ASLDetector()
        
        if not detector.is_trained:
            print("Model not trained!")
            return
            
        print(f"Model trained: {detector.is_trained}")
        print(f"Available signs: {list(detector.label_to_sign.values())}")
        
        # Try to predict with dummy features
        dummy_features = np.random.random(93)  # Typical feature size
        prediction, confidence = detector.predict(dummy_features)
        print(f"Dummy prediction: {prediction} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"Error testing model: {e}")

def test_training_pipeline():
    """Test the complete training pipeline"""
    print("\n=== Training Pipeline Test ===")
    
    try:
        detector = ASLDetector()
        
        # Check if training data was loaded
        print(f"Training features loaded: {len(detector.training_features)}")
        print(f"Training labels loaded: {len(detector.training_labels)}")
        
        if len(detector.training_features) > 0:
            # Test retraining
            print("Retraining model...")
            detector.train_model()
            print("Retrain completed")
            
            # Test prediction after retrain
            if len(detector.training_features) > 0:
                sample_features = np.array(detector.training_features[0])
                expected_label = detector.training_labels[0]
                prediction, confidence = detector.predict(sample_features)
                print(f"Sample test - Expected: {expected_label}, Got: {prediction}, Confidence: {confidence:.2f}")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")

def main():
    """Main debug function"""
    debug_training_data()
    test_model_prediction()
    test_training_pipeline()

if __name__ == "__main__":
    main()
