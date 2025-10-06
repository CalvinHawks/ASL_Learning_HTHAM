#!/usr/bin/env python3
"""
Camera Test Script
Test different camera backends and indices to diagnose camera issues
"""

import cv2
import sys

def test_camera_backends():
    """Test different camera backends and indices"""
    print("=== Camera Backend Test ===")
    
    backends = [
        ("Default", cv2.CAP_ANY),
        ("DirectShow", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("V4L2", cv2.CAP_V4L2),
    ]
    
    for backend_name, backend in backends:
        print(f"\nTesting {backend_name} backend:")
        for i in range(3):  # Test camera indices 0, 1, 2
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"  Camera {i}: [OK] Working (frame shape: {frame.shape})")
                    else:
                        print(f"  Camera {i}: [WARN] Opens but can't read frames")
                else:
                    print(f"  Camera {i}: [FAIL] Cannot open")
                cap.release()
            except Exception as e:
                print(f"  Camera {i}: [ERROR] {e}")

def test_simple_camera():
    """Simple camera test"""
    print("\n=== Simple Camera Test ===")
    
    # Test with DirectShow (Windows recommended)
    print("Testing with DirectShow backend...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("DirectShow failed, trying default...")
        cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("[OK] Camera opened successfully!")
        
        # Try to read a few frames
        for i in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK] Frame {i+1} read successfully (shape: {frame.shape})")
            else:
                print(f"[FAIL] Failed to read frame {i+1}")
                break
        
        cap.release()
        return True
    else:
        print("[FAIL] Could not open camera with any backend")
        return False

def main():
    """Main test function"""
    print("Camera Diagnostic Tool")
    print("=====================")
    
    # Test simple camera access
    if test_simple_camera():
        print("\n[OK] Camera is working! The issue might be in the main application.")
    else:
        print("\n[FAIL] Camera is not accessible. Possible issues:")
        print("1. Camera is being used by another application")
        print("2. Camera permissions not granted")
        print("3. Camera driver issues")
        print("4. Hardware problems")
    
    # Test different backends
    test_camera_backends()
    
    print("\n=== Recommendations ===")
    print("1. Close any applications using the camera (Zoom, Teams, etc.)")
    print("2. Check Windows camera permissions")
    print("3. Try running as administrator")
    print("4. Restart the computer if issues persist")

if __name__ == "__main__":
    main()
