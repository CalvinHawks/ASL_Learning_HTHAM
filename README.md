# ASL Learning Application

A computer vision application for learning American Sign Language (ASL) using MediaPipe hand tracking. This application provides an interactive learning experience where users can practice ASL signs and fingerspelling with real-time feedback.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand landmark detection
- **Fingerspelling Recognition**: Recognizes individual ASL letters A-Z
- **ASL Sign Recognition**: Machine learning-based recognition of common ASL signs
- **Interactive Learning Interface**: Guided lessons with prompts and feedback
- **Progress Tracking**: Save and load learning progress
- **Multiple Learning Modes**: Fingerspelling, ASL signs, and numbers
- **Custom Training**: Train the model with your own ASL signs

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- A webcam or camera

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ASL_Learning_HTHAM
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Run the main application:
```bash
python asl_learning_app.py
```

2. The application will open your camera and display the learning interface.

3. Follow the on-screen prompts to practice ASL signs.

### Controls

- **'n'** - Next lesson
- **'r'** - Restart current lesson
- **'m'** - Toggle detection mode (fingerspelling/ASL/auto)
- **'t'** - Train ASL model with current hand position
- **'s'** - Save progress
- **'l'** - Load progress
- **'q' or ESC** - Quit

### Training Custom ASL Signs

1. Run the training script:
```bash
python train_asl_model.py
```

2. Follow the interactive prompts to train the model with your own ASL signs.

3. The trained model will be saved and can be used in the main application.

## Learning Modes

### 1. Fingerspelling Mode
- Practice individual ASL letters A-Z
- Real-time recognition of hand shapes
- Perfect for learning the ASL alphabet

### 2. ASL Signs Mode
- Learn common ASL words and phrases
- Requires training with custom data
- More complex sign recognition

### 3. Auto Mode
- Automatically switches between fingerspelling and ASL signs
- Best for mixed practice sessions

## File Structure

```
ASL_Learning_HTHAM/
├── asl_learning_app.py      # Main application
├── hand_tracker.py          # MediaPipe hand tracking
├── asl_detector.py          # ASL recognition system
├── learning_interface.py    # User interface
├── train_asl_model.py       # Model training script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks in real-time
2. **Feature Extraction**: Hand landmarks are converted to feature vectors
3. **Sign Recognition**: Machine learning models classify hand configurations
4. **Learning Interface**: Provides prompts and feedback to guide learning
5. **Progress Tracking**: Saves learning progress and statistics

## Troubleshooting

### Camera Issues
- Make sure your camera is connected and not being used by another application
- Try changing the camera index in the code if you have multiple cameras

### Recognition Issues
- Ensure good lighting conditions
- Keep your hand clearly visible in the camera frame
- Try adjusting the confidence threshold in the code

### Performance Issues
- Close other applications using the camera
- Reduce the camera resolution in the code if needed
- Make sure you have sufficient RAM available

## Contributing

Feel free to contribute to this project by:
- Adding new ASL signs to the recognition system
- Improving the learning interface
- Adding new learning modes
- Fixing bugs or improving performance


## Acknowledgments

- MediaPipe team for the excellent hand tracking solution
- The ASL community for inspiration and feedback
- OpenCV for computer vision capabilities
