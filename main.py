import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # (optional) reduce TFLite warnings

import cv2
import mediapipe as mp
import math
import json
import time
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Keyboard & Mouse ---
keyboard = KeyboardController()
mouse = MouseController()

# --- Gesture Settings (loaded from JSON) ---
assignments = {}
settings = {
    "mirror_view": True,         # show the camera as a mirror
    "mirror_controls": False,    # swap Left/Right mapping for mirrored control logic
    "thresholds": {              # gesture thresholds
        "pinch_dist": 0.05,
        "two_split_min": 0.02
    },
    "debug_draw": True           # draw landmark connections
}

# --- JSON Watcher ---
class JSONHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("settings.json"):
            load_settings()

def load_settings():
    global assignments, settings
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # accept assignments (can be string or object)
        assignments = data.get("assignments", {})
        # general parameters
        settings.update({k: v for k, v in data.items() if k != "assignments"})
        print("Settings updated.")
    except Exception as e:
        print("Settings load error:", e)

# Initial JSON read
load_settings()

# Observer thread
observer = Observer()
observer.schedule(JSONHandler(), path=".", recursive=False)
observer.start()

# --- Mediapipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# --- State ---
pressed = {
    "left_pinch": False,
    "right_pinch": False,
    "left_fist": False,
    "right_fist": False,
    "left_two": False,
    "right_two": False
}

# last-fire timestamps for repeat mode
last_fire_ts = {
    "left_pinch": 0.0,
    "right_pinch": 0.0,
    "left_fist": 0.0,
    "right_fist": 0.0,
    "left_two": 0.0,
    "right_two": 0.0
}

# --- Helper: settings helpers ---
def get_threshold(name, default):
    return settings.get("thresholds", {}).get(name, default)

def resolve_hand_label(label):
    """
    Returns the actual MediaPipe hand label ("Left"/"Right").
    If mirror_controls=True, swap left/right for control logic.
    """
    if settings.get("mirror_controls", False):
        return "Right" if label == "Left" else "Left"
    return label

def get_assignment(gesture_key):
    """
    assignments[gesture_key] can be either a string or an object.
    Object format:
      {"action": "Key.space" or "w" or "mouse_left",
       "mode": "hold" | "repeat",
       "repeat_hz": 10,
       "tap_ms": 40}
    """
    item = assignments.get(gesture_key)
    if item is None:
        return None
    if isinstance(item, str):
        # backward compatibility: only action provided
        return {"action": item, "mode": "hold", "repeat_hz": 8, "tap_ms": 40}
    # fill missing fields with defaults
    return {
        "action": item.get("action"),
        "mode": item.get("mode", "hold"),
        "repeat_hz": float(item.get("repeat_hz", 8)),
        "tap_ms": int(item.get("tap_ms", 40))
    }

# --- Gesture Helpers ---
def is_pinch(hand):
    x1, y1 = hand.landmark[4].x, hand.landmark[4].y     # thumb tip
    x2, y2 = hand.landmark[8].x, hand.landmark[8].y     # index finger tip
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist < get_threshold("pinch_dist", 0.05)

def is_fist(hand):
    # if fingertip is below PIP (y is larger), treat as folded
    return (
        hand.landmark[8].y  > hand.landmark[5].y  and
        hand.landmark[12].y > hand.landmark[9].y  and
        hand.landmark[16].y > hand.landmark[13].y and
        hand.landmark[20].y > hand.landmark[17].y
    )

def finger_extended(hand, tip_id, pip_id):
    margin = 0.01
    return (hand.landmark[tip_id].y + margin) < hand.landmark[pip_id].y

def is_two_fingers_V(hand):
    index_ext  = finger_extended(hand, 8, 6)
    middle_ext = finger_extended(hand, 12, 10)
    ring_fold  = not finger_extended(hand, 16, 14)
    pinky_fold = not finger_extended(hand, 20, 18)
    min_split = get_threshold("two_split_min", 0.02)
    x_index  = hand.landmark[8].x
    x_middle = hand.landmark[12].x
    lateral_split_ok = abs(x_index - x_middle) > min_split
    return index_ext and middle_ext and ring_fold and pinky_fold and lateral_split_ok

def get_key(key_str):
    if isinstance(key_str, str) and key_str.startswith("Key."):
        return getattr(Key, key_str.split(".")[1])
    return key_str

def press_action(action, press=True):
    if not action:
        return
    if isinstance(action, str) and action.startswith("mouse_"):
        btn = {
            "mouse_left": Button.left,
            "mouse_right": Button.right,
            "mouse_middle": Button.middle
        }.get(action, Button.left)
        if press:
            mouse.press(btn)
        else:
            mouse.release(btn)
    else:
        key = get_key(action)
        try:
            if press:
                keyboard.press(key)
            else:
                keyboard.release(key)
        except Exception as e:
            print("None")

def tap_action(action, tap_ms=40):
    """Short tap (press then release after a brief delay)"""
    if not action:
        return
    press_action(action, True)
    # Tiny delay (keep ultra small to avoid blocking)
    t_end = time.perf_counter() + (tap_ms / 1000.0)
    while time.perf_counter() < t_end:
        pass
    press_action(action, False)

def handle_gesture(gesture_key, active_now):
    """
    Manages both HOLD and REPEAT logic for each gesture.
    """
    global pressed, last_fire_ts
    cfg = get_assignment(gesture_key)
    if cfg is None:
        # nothing mapped
        return

    mode = cfg["mode"]
    action = cfg["action"]

    # HOLD mode: on enter press, on exit release
    if mode == "hold":
        if active_now and not pressed[gesture_key]:
            press_action(action, True)
            pressed[gesture_key] = True
        elif (not active_now) and pressed[gesture_key]:
            press_action(action, False)
            pressed[gesture_key] = False

    # REPEAT mode: while active, send periodic "tap"
    elif mode == "repeat":
        if active_now:
            hz = max(1.0, float(cfg.get("repeat_hz", 8.0)))
            period = 1.0 / hz
            now = time.perf_counter()
            if now - last_fire_ts[gesture_key] >= period:
                tap_action(action, tap_ms=cfg.get("tap_ms", 40))
                last_fire_ts[gesture_key] = now
            pressed[gesture_key] = True
        else:
            # not active — reset state
            pressed[gesture_key] = False
            last_fire_ts[gesture_key] = 0.0

# --- Main Loop ---
try:
    while True:
        success, img = cap.read()
        if not success:
            print("[ERR] Failed to read from camera.")
            break

        # Mirror view (if enabled)
        if settings.get("mirror_view", True):
            img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for handLms, handType in zip(result.multi_hand_landmarks, result.multi_handedness):

                if settings.get("debug_draw", True):
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                # "Left"/"Right" (swap if mirror control is enabled)
                raw_label = handType.classification[0].label  # "Left" or "Right"
                label = resolve_hand_label(raw_label)

                # Compute gestures
                pinch = is_pinch(handLms)
                fist  = is_fist(handLms)
                two   = is_two_fingers_V(handLms)

                if label == "Left":
                    handle_gesture("left_pinch", pinch)
                    handle_gesture("left_fist",  fist)
                    handle_gesture("left_two",   two)

                elif label == "Right":
                    handle_gesture("right_pinch", pinch)
                    handle_gesture("right_fist",  fist)
                    handle_gesture("right_two",   two)

        cv2.imshow("Hand Control", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    observer.stop()
    observer.join()
    cap.release()
    cv2.destroyAllWindows()
