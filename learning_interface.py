import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import json
import os

class LearningInterface:
    """User interface for ASL learning with prompts and feedback"""
    
    def __init__(self, window_name: str = "ASL Learning App"):
        """
        Initialize the learning interface
        
        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        self.current_lesson = None
        self.lessons = self._load_lessons()
        self.lesson_index = 0
        self.score = 0
        self.total_attempts = 0
        self.correct_attempts = 0
        self.current_prompt = ""
        self.feedback_message = ""
        self.feedback_color = (0, 255, 0)  # Green by default
        self.last_detection_time = 0
        self.detection_cooldown = 0.5  # Shorter cooldown for more responsive detection
        
    def _load_lessons(self) -> List[Dict]:
        """Load predefined ASL learning lessons"""
        lessons = [
            {
                "name": "Basic Fingerspelling",
                "type": "fingerspelling",
                "prompts": [
                    {"text": "Sign the letter A", "target": "A"},
                    {"text": "Sign the letter B", "target": "B"},
                    {"text": "Sign the letter C", "target": "C"},
                    {"text": "Sign the letter D", "target": "D"},
                    {"text": "Sign the letter E", "target": "E"},
                    {"text": "Sign the letter F", "target": "F"},
                    {"text": "Sign the letter G", "target": "G"},
                    {"text": "Sign the letter H", "target": "H"},
                    {"text": "Sign the letter I", "target": "I"},
                    {"text": "Sign the letter J", "target": "J"}
                ]
            },
            {
                "name": "Common Words",
                "type": "words",
                "prompts": [
                    {"text": "Sign HELLO", "target": "HELLO"},
                    {"text": "Sign THANK YOU", "target": "THANK_YOU"},
                    {"text": "Sign PLEASE", "target": "PLEASE"},
                    {"text": "Sign SORRY", "target": "SORRY"},
                    {"text": "Sign YES", "target": "YES"},
                    {"text": "Sign NO", "target": "NO"},
                    {"text": "Sign GOOD", "target": "GOOD"},
                    {"text": "Sign BAD", "target": "BAD"},
                    {"text": "Sign LOVE", "target": "LOVE"},
                    {"text": "Sign HELP", "target": "HELP"}
                ]
            },
            {
                "name": "Numbers 1-10",
                "type": "numbers",
                "prompts": [
                    {"text": "Sign the number 1", "target": "1"},
                    {"text": "Sign the number 2", "target": "2"},
                    {"text": "Sign the number 3", "target": "3"},
                    {"text": "Sign the number 4", "target": "4"},
                    {"text": "Sign the number 5", "target": "5"},
                    {"text": "Sign the number 6", "target": "6"},
                    {"text": "Sign the number 7", "target": "7"},
                    {"text": "Sign the number 8", "target": "8"},
                    {"text": "Sign the number 9", "target": "9"},
                    {"text": "Sign the number 10", "target": "10"}
                ]
            }
        ]
        return lessons
    
    def start_lesson(self, lesson_index: int = 0) -> None:
        """
        Start a specific lesson
        
        Args:
            lesson_index: Index of the lesson to start
        """
        if 0 <= lesson_index < len(self.lessons):
            self.current_lesson = self.lessons[lesson_index]
            self.lesson_index = lesson_index
            self.score = 0
            self.total_attempts = 0
            self.correct_attempts = 0
            self.current_prompt = self.current_lesson["prompts"][0]["text"]
            self.feedback_message = "Get ready to start!"
            self.feedback_color = (0, 255, 255)  # Yellow
            print(f"Started lesson: {self.current_lesson['name']}")
    
    def get_current_prompt(self) -> str:
        """Get the current learning prompt"""
        return self.current_prompt
    
    def get_current_target(self) -> str:
        """Get the current target sign/letter"""
        if self.current_lesson and self.current_lesson["prompts"]:
            prompt_index = min(self.score, len(self.current_lesson["prompts"]) - 1)
            return self.current_lesson["prompts"][prompt_index]["target"]
        return ""
    
    def check_answer(self, detected_sign: str, confidence: float) -> bool:
        """
        Check if the detected sign matches the target
        
        Args:
            detected_sign: The sign detected by the system
            confidence: Confidence level of the detection
            
        Returns:
            True if the answer is correct, False otherwise
        """
        current_time = time.time()
        
        # Check cooldown to prevent spam
        if current_time - self.last_detection_time < self.detection_cooldown:
            return False
        
        self.last_detection_time = current_time
        self.total_attempts += 1
        
        target = self.get_current_target()
        
        if not target:
            return False
        
        # Check if detection matches target
        is_correct = detected_sign.upper() == target.upper()
        
        if is_correct and confidence > 0.5:  # Lower confidence threshold for better detection
            self.correct_attempts += 1
            self.score += 1
            self.feedback_message = f"Correct! Great job! (Confidence: {confidence:.2f})"
            self.feedback_color = (0, 255, 0)  # Green
            
            # Move to next prompt
            self._next_prompt()
            
            # Add a small delay to show the success message
            time.sleep(0.5)
            
        else:
            if confidence <= 0.5:
                self.feedback_message = f"Try again! Make sure your hand is clearly visible. (Confidence: {confidence:.2f})"
            else:
                self.feedback_message = f"Not quite right. Try again! (Detected: {detected_sign}, Expected: {target})"
            self.feedback_color = (0, 0, 255)  # Red
        
        return is_correct
    
    def _next_prompt(self) -> None:
        """Move to the next prompt in the current lesson"""
        if not self.current_lesson:
            return
        
        # Check if lesson is complete
        if self.score >= len(self.current_lesson["prompts"]):
            # Lesson completed
            self.current_prompt = "Lesson completed! Great job!"
            self.feedback_message = f"Final Score: {self.correct_attempts}/{self.total_attempts} correct!"
            self.feedback_color = (0, 255, 0)  # Green
            print(f"Lesson completed! Score: {self.correct_attempts}/{self.total_attempts}")
        else:
            # Move to next prompt
            next_index = self.score
            if next_index < len(self.current_lesson["prompts"]):
                self.current_prompt = self.current_lesson["prompts"][next_index]["text"]
                print(f"Next prompt: {self.current_prompt}")
    
    def is_lesson_complete(self) -> bool:
        """Check if the current lesson is complete"""
        if not self.current_lesson:
            return True
        return self.score >= len(self.current_lesson["prompts"])
    
    def get_lesson_progress(self) -> Tuple[int, int]:
        """Get current lesson progress as (current, total)"""
        if not self.current_lesson:
            return 0, 0
        return self.score, len(self.current_lesson["prompts"])
    
    def get_accuracy(self) -> float:
        """Get current accuracy percentage"""
        if self.total_attempts == 0:
            return 0.0
        return (self.correct_attempts / self.total_attempts) * 100
    
    def draw_interface(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the learning interface on the frame
        
        Args:
            frame: Input frame to draw on
            
        Returns:
            Frame with interface elements drawn
        """
        height, width = frame.shape[:2]
        
        # Create overlay for better text visibility
        overlay = frame.copy()
        
        # Draw semi-transparent background for text areas
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, height - 100), (width - 10, height - 10), (0, 0, 0), -1)
        
        # Blend overlay with frame
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw lesson title
        if self.current_lesson:
            lesson_title = f"Lesson: {self.current_lesson['name']}"
            cv2.putText(frame, lesson_title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw current prompt
        prompt_text = f"Prompt: {self.current_prompt}"
        cv2.putText(frame, prompt_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw progress
        current, total = self.get_lesson_progress()
        progress_text = f"Progress: {current}/{total}"
        cv2.putText(frame, progress_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw feedback message
        if self.feedback_message:
            cv2.putText(frame, self.feedback_message, (20, height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.feedback_color, 2)
        
        # Draw accuracy
        accuracy = self.get_accuracy()
        accuracy_text = f"Accuracy: {accuracy:.1f}%"
        cv2.putText(frame, accuracy_text, (20, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instructions
        instructions = "Press 'n' for next lesson, 'r' to restart, 'q' to quit"
        cv2.putText(frame, instructions, (width - 500, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def handle_keypress(self, key: int) -> str:
        """
        Handle keyboard input
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            Action taken ('next_lesson', 'restart', 'quit', 'none')
        """
        if key == ord('n') or key == ord('N'):
            # Next lesson
            next_lesson = (self.lesson_index + 1) % len(self.lessons)
            self.start_lesson(next_lesson)
            return 'next_lesson'
        
        elif key == ord('r') or key == ord('R'):
            # Restart current lesson
            self.start_lesson(self.lesson_index)
            return 'restart'
        
        elif key == ord('q') or key == ord('Q') or key == 27:  # ESC key
            return 'quit'
        
        return 'none'
    
    def save_progress(self, filename: str = "learning_progress.json") -> None:
        """Save learning progress to file"""
        progress = {
            "current_lesson": self.lesson_index,
            "score": self.score,
            "total_attempts": self.total_attempts,
            "correct_attempts": self.correct_attempts,
            "accuracy": self.get_accuracy()
        }
        
        with open(filename, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self, filename: str = "learning_progress.json") -> None:
        """Load learning progress from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    progress = json.load(f)
                
                self.lesson_index = progress.get("current_lesson", 0)
                self.score = progress.get("score", 0)
                self.total_attempts = progress.get("total_attempts", 0)
                self.correct_attempts = progress.get("correct_attempts", 0)
                
                print(f"Progress loaded: {self.correct_attempts}/{self.total_attempts} correct")
            except Exception as e:
                print(f"Error loading progress: {e}")
    
    def reset_progress(self) -> None:
        """Reset all learning progress"""
        self.score = 0
        self.total_attempts = 0
        self.correct_attempts = 0
        self.lesson_index = 0
        self.feedback_message = "Progress reset!"
        self.feedback_color = (0, 255, 255)  # Yellow
