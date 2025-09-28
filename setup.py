#!/usr/bin/env python3
"""
Setup script for ASL Learning Application
This script helps users set up the application and check requirements.
"""

import subprocess
import sys
import os
import cv2

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"[ERROR] Python {version.major}.{version.minor} is not supported.")
        print("Please install Python 3.7 or higher.")
        return False
    else:
        print(f"[OK] Python {version.major}.{version.minor} is compatible.")
        return True

def check_camera():
    """Check if camera is available"""
    print("\nChecking camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[OK] Camera is available.")
        cap.release()
        return True
    else:
        print("[ERROR] No camera found.")
        print("Please connect a camera and try again.")
        return False

def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error installing requirements: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting imports...")
    modules = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("joblib", "Joblib")
    ]
    
    all_imports_successful = True
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"[OK] {name} imported successfully.")
        except ImportError as e:
            print(f"[ERROR] {name} import failed: {e}")
            all_imports_successful = False
    
    return all_imports_successful

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = ["models", "data", "saved_models"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[OK] Created directory: {directory}")
        else:
            print(f"[OK] Directory already exists: {directory}")

def run_basic_test():
    """Run a basic test of the application"""
    print("\nRunning basic test...")
    try:
        from hand_tracker import HandTracker
        from asl_detector import FingerspellingDetector
        from learning_interface import LearningInterface
        
        # Test initialization
        hand_tracker = HandTracker()
        fingerspelling_detector = FingerspellingDetector()
        learning_interface = LearningInterface()
        
        print("[OK] All modules initialized successfully.")
        
        # Cleanup
        hand_tracker.release()
        
        return True
    except Exception as e:
        print(f"[ERROR] Basic test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("ASL Learning Application - Setup")
    print("================================")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check camera
    if not check_camera():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Some modules failed to import. Please check the installation.")
        return False
    
    # Create directories
    create_directories()
    
    # Run basic test
    if not run_basic_test():
        print("\n[ERROR] Basic test failed. Please check the installation.")
        return False
    
    print("\n[SUCCESS] Setup completed successfully!")
    print("\nYou can now run the application with:")
    print("  python asl_learning_app.py")
    print("\nOr run examples with:")
    print("  python example_usage.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
