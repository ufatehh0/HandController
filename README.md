# HandController (MediaPipe + PyQt5)

Control keyboard and mouse actions using hand gestures captured from the webcam.

## Requirements
- **Python 3.11 or below** 


## Installation
Install the required libraries:

 - pip install opencv-python mediapipe PyQt5 pynput watchdog

## Usage 
python gui.py

### Settings Explained
- **action** – The key or mouse action to trigger (e.g., `"w"`, `"Key.space"`, `"mouse_left"`).  
- **mode** – How the action is triggered:  
  - `"hold"` = press while gesture is active  
  - `"repeat"` = repeatedly tap while gesture is active  
- **repeat_hz** – Frequency (times per second) when using `"repeat"` mode.  
- **tap_ms** – Duration (milliseconds) of a single tap (press + release).  
- **mirror_view** – If `true`, flips the camera preview like a mirror.  
- **mirror_controls** – If `true`, swaps left and right hand controls.  
- **debug_draw** – If `true`, draws landmarks and hand skeleton on the preview.  
- **thresholds** – Numeric sensitivity values:  
  - **pinch_dist** – Distance threshold for pinch detection.  
  - **two_split_min** – Minimum horizontal distance for two-fingers (V) gesture.  
- **assignments** – Mapping of each gesture (`left_pinch`, `right_fist`, etc.) to its action, mode, and repeat settings.
