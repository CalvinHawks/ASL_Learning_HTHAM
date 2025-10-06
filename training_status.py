#!/usr/bin/env python3
"""
Training Status Checker
Check the current status of your ASL model training
"""

import json
import os
from asl_detector import ASLDetector

def check_training_status():
    """Check the current training status"""
    print("=== ASL Training Status ===")
    
    # Check if model files exist
    model_files = {
        'Model': 'asl_model.pkl',
        'Scaler': 'asl_scaler.pkl', 
        'Mappings': 'asl_mappings.json',
        'Training Data': 'training_data.json'
    }
    
    print("\nFile Status:")
    for name, file in model_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  {name}: [OK] Found ({size} bytes)")
        else:
            print(f"  {name}: [MISSING] Not found")
    
    # Load and check training data
    if os.path.exists('training_data.json'):
        try:
            with open('training_data.json', 'r') as f:
                data = json.load(f)
                features = data.get('features', [])
                labels = data.get('labels', [])
                
            print(f"\nTraining Data:")
            print(f"  Total samples: {len(features)}")
            print(f"  Unique signs: {len(set(labels))}")
            
            # Count samples per sign
            sign_counts = {}
            for label in labels:
                sign_counts[label] = sign_counts.get(label, 0) + 1
            
            print(f"\nSamples per sign:")
            for sign, count in sorted(sign_counts.items()):
                print(f"  {sign}: {count} samples")
                
        except Exception as e:
            print(f"Error reading training data: {e}")
    
    # Check model status
    try:
        detector = ASLDetector()
        if detector.is_trained:
            print(f"\nModel Status: [TRAINED]")
            print(f"  Signs in model: {len(detector.label_to_sign)}")
            print(f"  Available signs: {list(detector.label_to_sign.values())}")
        else:
            print(f"\nModel Status: [NOT TRAINED]")
    except Exception as e:
        print(f"Error checking model: {e}")

def main():
    """Main function"""
    check_training_status()
    
    print("\n=== Recommendations ===")
    print("1. If you have training data but no model, press 's' in the app to train")
    print("2. If you want to add more samples, press 't' while signing in the app")
    print("3. If you want to start fresh, delete training_data.json and restart")

if __name__ == "__main__":
    main()
