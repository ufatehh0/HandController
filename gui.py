import sys
import os
import json
import math
import time
from dataclasses import dataclass

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import mediapipe as mp

from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button

USE_JSON = True  ## Use JSON ?


# ---------- Helper dataclass for assignment rows ----------
@dataclass
class AssignmentRow:
    gesture: str
    action: str
    mode: str
    repeat_hz: float
    tap_ms: int

# ---------- Globals / Defaults ----------
DEFAULT_SETTINGS = {
    "mirror_view":False,
    "mirror_controls": False,
    "debug_draw": True,
    "thresholds": { "pinch_dist": 0.05, "two_split_min": 0.02 },
    "assignments": {
        "left_pinch":  { "action": "w",    "mode": "hold"},
        "right_pinch": { "action": "s",    "mode": "hold"},
        "left_fist":   { "action": "d",    "mode": "hold"},
        "right_fist":  { "action": "a",    "mode": "hold"},
        "left_two":    { "action": "none", "mode": "repeat", "repeat_hz": 5, "tap_ms": 40 },
        "right_two":   { "action": "none", "mode": "repeat", "repeat_hz": 5, "tap_ms": 40 }
    }
}

SETTINGS_FILE = "settings.json"

# ready-made action choices
ACTION_SUGGESTIONS = [
    # mouse
    "mouse_left", "mouse_right", "mouse_middle",
    # Key.* (pynput)
    "Key.space", "Key.enter", "Key.esc", "Key.tab",
    "Key.shift", "Key.ctrl", "Key.alt",
    "Key.up", "Key.down", "Key.left", "Key.right",
    
    
]

# ---------- Keyboard/Mouse controllers ----------
keyboard = KeyboardController()
mouse = MouseController()

# ---------- Video + Mediapipe worker in QThread ----------
class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    status_update = QtCore.pyqtSignal(str)
    settings_changed = QtCore.pyqtSignal(dict)

    def __init__(self, settings):
        super().__init__()
        self._running = False
        self.settings = settings.copy()
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = None

        self.pressed = {
            "left_pinch": False, "right_pinch": False,
            "left_fist": False,  "right_fist": False,
            "left_two": False,   "right_two": False
        }
        self.last_fire_ts = {k: 0.0 for k in self.pressed.keys()}

    def update_settings(self, new_settings):
        self.settings = new_settings.copy()
        self.settings_changed.emit(self.settings)

    def get_threshold(self, name, default):
        return self.settings.get("thresholds", {}).get(name, default)

    def resolve_hand_label(self, label):
        if self.settings.get("mirror_controls", False):
            return "Right" if label == "Left" else "Left"
        return label

    def get_assignment(self, gesture_key):
        assignments = self.settings.get("assignments", {})
        item = assignments.get(gesture_key)
        if item is None:
            return None
        if isinstance(item, str):
            return {"action": item, "mode": "hold", "repeat_hz": 8.0, "tap_ms": 40}
        return {
            "action": item.get("action"),
            "mode": item.get("mode", "hold"),
            "repeat_hz": float(item.get("repeat_hz", 8.0)),
            "tap_ms": int(item.get("tap_ms", 40))
        }

    def get_key(self, key_str):
        if isinstance(key_str, str) and key_str.startswith("Key."):
            return getattr(Key, key_str.split(".")[1])
        return key_str

    def press_action(self, action, press=True):
        if not action or action == "none":
            return
        if isinstance(action, str) and action.startswith("mouse_"):
            btn = {
                "mouse_left": Button.left,
                "mouse_right": Button.right,
                "mouse_middle": Button.middle
            }.get(action, Button.left)
            (mouse.press if press else mouse.release)(btn)
        else:
            key = self.get_key(action)
            try:
                (keyboard.press if press else keyboard.release)(key)
            except Exception:
                pass

    def tap_action(self, action, tap_ms=40):
        if not action or action == "none":
            return
        self.press_action(action, True)
        t_end = time.perf_counter() + (tap_ms / 1000.0)
        while time.perf_counter() < t_end:
            pass
        self.press_action(action, False)

    def is_pinch(self, hand):
        x1, y1 = hand.landmark[4].x, hand.landmark[4].y
        x2, y2 = hand.landmark[8].x, hand.landmark[8].y
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist < self.get_threshold("pinch_dist", 0.05)

    def is_fist(self, hand):
        return (
            hand.landmark[8].y  > hand.landmark[5].y  and
            hand.landmark[12].y > hand.landmark[9].y  and
            hand.landmark[16].y > hand.landmark[13].y and
            hand.landmark[20].y > hand.landmark[17].y
        )

    def finger_extended(self, hand, tip_id, pip_id):
        margin = 0.01
        return (hand.landmark[tip_id].y + margin) < hand.landmark[pip_id].y

    def is_two_fingers_V(self, hand):
        index_ext  = self.finger_extended(hand, 8, 6)
        middle_ext = self.finger_extended(hand, 12, 10)
        ring_fold  = not self.finger_extended(hand, 16, 14)
        pinky_fold = not self.finger_extended(hand, 20, 18)
        min_split = self.get_threshold("two_split_min", 0.02)
        x_index  = hand.landmark[8].x
        x_middle = hand.landmark[12].x
        lateral_split_ok = abs(x_index - x_middle) > min_split
        return index_ext and middle_ext and ring_fold and pinky_fold and lateral_split_ok

    def handle_gesture(self, gesture_key, active_now):
        cfg = self.get_assignment(gesture_key)
        if cfg is None:
            return
        mode = cfg["mode"]
        action = cfg["action"]
        if mode == "hold":
            if active_now and not self.pressed[gesture_key]:
                self.press_action(action, True)
                self.pressed[gesture_key] = True
            elif (not active_now) and self.pressed[gesture_key]:
                self.press_action(action, False)
                self.pressed[gesture_key] = False
        elif mode == "repeat":
            if active_now:
                hz = max(1.0, float(cfg.get("repeat_hz", 8.0)))
                period = 1.0 / hz
                now = time.perf_counter()
                if now - self.last_fire_ts[gesture_key] >= period:
                    self.tap_action(action, tap_ms=cfg.get("tap_ms", 40))
                    self.last_fire_ts[gesture_key] = now
                self.pressed[gesture_key] = True
            else:
                self.pressed[gesture_key] = False
                self.last_fire_ts[gesture_key] = 0.0

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7,
                                         min_tracking_confidence=0.7,
                                         max_num_hands=2)
        self._running = True
        last_status = ""

        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            if self.settings.get("mirror_view", True):
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            if res.multi_hand_landmarks and res.multi_handedness:
                parts = []
                for handLms, handType in zip(res.multi_hand_landmarks, res.multi_handedness):
                    if self.settings.get("debug_draw", True):
                        self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)

                    raw_label = handType.classification[0].label
                    label = self.resolve_hand_label(raw_label)

                    pinch = self.is_pinch(handLms)
                    fist  = self.is_fist(handLms)
                    two   = self.is_two_fingers_V(handLms)

                    if label == "Left":
                        self.handle_gesture("left_pinch", pinch)
                        self.handle_gesture("left_fist",  fist)
                        self.handle_gesture("left_two",   two)
                    else:
                        self.handle_gesture("right_pinch", pinch)
                        self.handle_gesture("right_fist",  fist)
                        self.handle_gesture("right_two",   two)

                    tags = []
                    if pinch: tags.append("Pinch")
                    if fist:  tags.append("Fist")
                    if two:   tags.append("Two")
                    parts.append(f"{label}: {','.join(tags) if tags else 'Idle'}")

                status = " | ".join(parts)
            else:
                status = "No hands"
                for k in list(self.pressed.keys()):
                    if self.pressed[k]:
                        cfg = self.get_assignment(k)
                        if cfg: self.press_action(cfg["action"], False)
                        self.pressed[k] = False

            if status != last_status:
                self.status_update.emit(status)
                last_status = status

            h, w, ch = frame.shape
            qimg = QtGui.QImage(frame.data, w, h, ch*w, QtGui.QImage.Format_BGR888)
            self.frame_ready.emit(qimg)

            time.sleep(0.01)

        try:
            if self.cap: self.cap.release()
            if self.hands: self.hands.close()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self.wait()


# ---------- Main Window ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Control — GUI")
        
        self.resize(1150, 740)

        # modern look
        QtWidgets.QApplication.setStyle("Fusion")
        self.apply_modern_qss()

        self.settings = self.load_settings_from_file()

        # central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)
        main.setContentsMargins(12,12,12,12)
        main.setSpacing(12)

        # ---- LEFT: video + status + start/stop
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setFixedSize(960, 540)  # 16:9
        self.video_label.setStyleSheet("background:#000; border-radius:12px;")
        self._blacken_video()
        left.addWidget(self.video_label)

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight:600; color:#c7a4ff;")
        
        left.addWidget(self.status_label)

        buttons = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn  = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        for b in (self.start_btn, self.stop_btn):
            b.setMinimumHeight(36)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        left.addLayout(buttons)

        main.addLayout(left, 2)

        # ---- RIGHT: Splitter (top: settings, bottom: assignments wide)
        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_split.setHandleWidth(10)

        # top card: settings
        top_card = QtWidgets.QGroupBox("Settings")
        top_card.setStyleSheet("QGroupBox{font-weight:600;}")
        top_layout = QtWidgets.QVBoxLayout(top_card)
        top_layout.setSpacing(8)

        self.chk_mirror_view = QtWidgets.QCheckBox("Mirror View (Preview)")
        self.chk_mirror_controls = QtWidgets.QCheckBox("Mirror Controls (Left/Right swap)")
        self.chk_debug_draw = QtWidgets.QCheckBox("Debug draw landmarks")
        self.chk_mirror_view.setChecked(self.settings.get("mirror_view", True))
        self.chk_mirror_controls.setChecked(self.settings.get("mirror_controls", False))
        self.chk_debug_draw.setChecked(self.settings.get("debug_draw", True))
        top_layout.addWidget(self.chk_mirror_view)
        top_layout.addWidget(self.chk_mirror_controls)
        top_layout.addWidget(self.chk_debug_draw)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12); grid.setVerticalSpacing(8)
        lbl1 = QtWidgets.QLabel("Pinch distance")
        lbl2 = QtWidgets.QLabel("Two-fingers split")
        self.sld_pinch = QtWidgets.QDoubleSpinBox()
        self.sld_pinch.setDecimals(3); self.sld_pinch.setRange(0.001, 0.2); self.sld_pinch.setSingleStep(0.005)
        self.sld_pinch.setValue(self.settings.get("thresholds", {}).get("pinch_dist", 0.05))
        self.sld_two   = QtWidgets.QDoubleSpinBox()
        self.sld_two.setDecimals(3); self.sld_two.setRange(0.001, 0.2); self.sld_two.setSingleStep(0.005)
        self.sld_two.setValue(self.settings.get("thresholds", {}).get("two_split_min", 0.02))
        grid.addWidget(lbl1,0,0); grid.addWidget(self.sld_pinch,0,1)
        grid.addWidget(lbl2,1,0); grid.addWidget(self.sld_two,1,1)
        top_layout.addLayout(grid)

        right_split.addWidget(top_card)

        # bottom card: assignments (JSON editing area) — wide
        bottom_card = QtWidgets.QGroupBox("Gesture Assignments (JSON Edit Area)")
        bottom_card.setStyleSheet("QGroupBox{font-weight:600;}")
        bottom_layout = QtWidgets.QVBoxLayout(bottom_card)
        bottom_layout.setSpacing(8)

        self.tbl = QtWidgets.QTableWidget(6, 5)
        self.tbl.setHorizontalHeaderLabels(["Gesture", "Action", "Mode", "Repeat Hz", "Tap ms"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.tbl.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gesture_keys = ["left_pinch","right_pinch","left_fist","right_fist","left_two","right_two"]
        self.populate_assignments_table()
        bottom_layout.addWidget(self.tbl, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Reload settings.json")
        self.save_btn = QtWidgets.QPushButton("Save to settings.json")
        self.reset_btn = QtWidgets.QPushButton("Reset to defaults")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.reset_btn)
        bottom_layout.addLayout(btn_row)

        right_split.addWidget(bottom_card)

        # splitter sizes: make the bottom part larger
        right_split.setStretchFactor(0, 1)  # top
        right_split.setStretchFactor(1, 3)  # bottom (larger)

        main.addWidget(right_split, 3)

        # connections
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)
        self.load_btn.clicked.connect(self.reload_from_file)
        self.save_btn.clicked.connect(self.save_to_file)
        self.reset_btn.clicked.connect(self.reset_defaults)

        for w in (self.chk_mirror_view, self.chk_mirror_controls, self.chk_debug_draw):
            w.stateChanged.connect(self.apply_controls_to_settings)
        self.sld_pinch.valueChanged.connect(self.apply_controls_to_settings)
        self.sld_two.valueChanged.connect(self.apply_controls_to_settings)

        self.file_watcher = QtCore.QFileSystemWatcher([SETTINGS_FILE] if os.path.exists(SETTINGS_FILE) else [])
        self.file_watcher.fileChanged.connect(self.on_external_settings_changed)

        self.worker = None

        if not USE_JSON:  # Turn off JSON
            self.load_btn.hide()
            self.save_btn.hide()

    # ---------- Style ----------
    def apply_modern_qss(self):
        # Color palette
        PRIMARY_TXT   = "#c7a4ff"   # primary text (lavender)
        TITLE_TXT     = "#b388ff"   # headings
        SUBTLE_TXT    = "#a78bfa"   # secondary
        BORDER_COL    = "#2a2f3a"
        BG_MAIN       = "#0f1115"
        BG_CARD       = "#141824"
        PURPLE_FILL   = "#7c3aed"   # checked background + scrollbar handle
        PURPLE_EDGE   = "#a78bfa"   # checked border

        self.setStyleSheet(f"""
            QMainWindow {{
                background:{BG_MAIN};
                color:{PRIMARY_TXT};
            }}
            QLabel {{
                color:{PRIMARY_TXT};
            }}
            QGroupBox {{
                border:1px solid {BORDER_COL};
                border-radius:12px;
                padding:10px;
                margin-top:8px;
                color:{PRIMARY_TXT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 4px;
                color:{TITLE_TXT};
                font-weight:600;
            }}

            QPushButton {{
                background:#1b1f2a;
                border:1px solid {BORDER_COL};
                border-radius:10px;
                padding:8px 14px;
                color:{PRIMARY_TXT};
            }}
            QPushButton:hover {{ background:#222736; }}
            QPushButton:disabled {{
                color:#7e62b8;
                border-color:#333;
                background:#151821;
            }}

            /* CHECKBOX / RADIO */
            QCheckBox, QRadioButton {{
                color:{PRIMARY_TXT};
                spacing:8px;
            }}
            QCheckBox::indicator, QRadioButton::indicator {{
                width:18px; height:18px;
                border:1px solid {BORDER_COL};
                border-radius:4px;
                background:{BG_CARD};
                margin-right:6px;
            }}
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
                border:1px solid {PURPLE_EDGE};
                background:{PURPLE_FILL};
                image: url(:/qt-project.org/styles/commonstyle/images/check.png);
            }}
            QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
                border-color:{PURPLE_EDGE};
            }}

            /* INPUTS & TABLE */
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QTableWidget {{
                background:{BG_CARD};
                color:{PRIMARY_TXT};
                border:1px solid {BORDER_COL};
                border-radius:8px;
                selection-background-color:#322a44;
                selection-color:{PRIMARY_TXT};
            }}

            QHeaderView::section {{
                background:{BG_CARD};
                color:{SUBTLE_TXT};
                border:none;
                padding:6px;
            }}

            QTableWidget {{
                gridline-color:{BORDER_COL};
                alternate-background-color:#121520;
            }}

            /* SCROLLBAR — black background + purple handle */
            QScrollBar:vertical, QScrollBar:horizontal {{
                background:#000000;
                border:none;
                margin:0px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background:{PURPLE_FILL};
                border-radius:6px;
                min-height:24px;
                min-width:24px;
            }}
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
                background:#9f67ff;
            }}
            QScrollBar::handle:vertical:pressed, QScrollBar::handle:horizontal:pressed {{
                background:#5b21b6;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                background:none;
                border:none;
                width:0px;
                height:0px;
            }}
    """)

    # ---------- Table widgets ----------
    def _make_action_combo(self, current_text: str) -> QtWidgets.QComboBox:
        cmb = QtWidgets.QComboBox()
        cmb.setEditable(True)
        cmb.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        for s in ACTION_SUGGESTIONS:
            cmb.addItem(s)
        if current_text and current_text not in ACTION_SUGGESTIONS:
            cmb.insertItem(0, current_text)
            cmb.setCurrentIndex(0)
        else:
            cmb.setCurrentText(current_text if current_text else "none")
        comp = QtWidgets.QCompleter(ACTION_SUGGESTIONS, cmb)
        comp.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        cmb.setCompleter(comp)
        cmb.currentTextChanged.connect(self.apply_controls_to_settings)
        return cmb

    def _make_mode_combo(self, mode_text: str, row: int) -> QtWidgets.QComboBox:
        cmb = QtWidgets.QComboBox()
        cmb.addItems(["hold", "repeat"])
        cmb.setCurrentText(mode_text or "hold")
        cmb.currentTextChanged.connect(lambda _m: self._on_mode_changed(row))
        return cmb

    def _make_hz_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(1.0, 60.0); sp.setDecimals(1); sp.setSingleStep(0.5)
        sp.setValue(float(value) if value else 8.0)
        sp.setAlignment(QtCore.Qt.AlignCenter)
        sp.valueChanged.connect(lambda *_: self.apply_controls_to_settings())
        return sp

    def _make_ms_spin(self, value: int) -> QtWidgets.QSpinBox:
        sp = QtWidgets.QSpinBox()
        sp.setRange(10, 300); sp.setSingleStep(10); sp.setValue(int(value) if value else 40)
        sp.setAlignment(QtCore.Qt.AlignCenter)
        sp.valueChanged.connect(lambda *_: self.apply_controls_to_settings())
        return sp

    def _set_repeat_enabled(self, row: int, enabled: bool):
        for col in (3, 4):
            w = self.tbl.cellWidget(row, col)
            if w:
                w.setEnabled(enabled)
                w.setStyleSheet("" if enabled else "color:#888;")

    def _on_mode_changed(self, row: int):
        mode_w = self.tbl.cellWidget(row, 2)
        enabled = (mode_w.currentText() == "repeat") if isinstance(mode_w, QtWidgets.QComboBox) else False
        self._set_repeat_enabled(row, enabled)
        self.apply_controls_to_settings()

    def populate_assignments_table(self):
        self.tbl.blockSignals(True)
        self.tbl.clearContents()
        assignments = self.settings.get("assignments", {})
        for row, g in enumerate(self.gesture_keys):
            item_g = QtWidgets.QTableWidgetItem(g)
            item_g.setFlags(item_g.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(row, 0, item_g)

            cfg = assignments.get(g, {})
            if isinstance(cfg, str):
                cfg = {"action": cfg, "mode": "hold"}

            action = (cfg or {}).get("action", "none")
            cmb_action = self._make_action_combo(action)
            self.tbl.setCellWidget(row, 1, cmb_action)

            mode = (cfg or {}).get("mode", "hold")
            cmb_mode = self._make_mode_combo(mode, row)
            self.tbl.setCellWidget(row, 2, cmb_mode)

            hz = (cfg or {}).get("repeat_hz", 8.0)
            sp_hz = self._make_hz_spin(hz)
            self.tbl.setCellWidget(row, 3, sp_hz)

            ms = (cfg or {}).get("tap_ms", 40)
            sp_ms = self._make_ms_spin(ms)
            self.tbl.setCellWidget(row, 4, sp_ms)

            self._set_repeat_enabled(row, mode == "repeat")

        self.tbl.blockSignals(False)

    def read_table_into_settings(self):
        if "assignments" not in self.settings:
            self.settings["assignments"] = {}
        for row, g in enumerate(self.gesture_keys):
            action_w = self.tbl.cellWidget(row, 1)
            mode_w   = self.tbl.cellWidget(row, 2)
            hz_w     = self.tbl.cellWidget(row, 3)
            ms_w     = self.tbl.cellWidget(row, 4)

            action = action_w.currentText().strip() if isinstance(action_w, QtWidgets.QComboBox) else "none"
            mode   = mode_w.currentText().strip() if isinstance(mode_w, QtWidgets.QComboBox) else "hold"

            entry = {"action": action, "mode": mode}
            if mode == "repeat":
                entry["repeat_hz"] = float(hz_w.value()) if isinstance(hz_w, QtWidgets.QDoubleSpinBox) else 8.0
                entry["tap_ms"]    = int(ms_w.value())  if isinstance(ms_w, QtWidgets.QSpinBox)       else 40
            self.settings["assignments"][g] = entry

    # ---------- settings I/O ----------
    def load_settings_from_file(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged = DEFAULT_SETTINGS.copy()
                merged.update(data)
                if "thresholds" in data:
                    merged["thresholds"].update(data["thresholds"])
                return merged
            except Exception as e:
                print("Error reading settings.json:", e)
                return DEFAULT_SETTINGS.copy()
        else:
            return DEFAULT_SETTINGS.copy()

    def save_settings_to_file(self):
        self.read_table_into_settings()
        self.settings["thresholds"] = {
            "pinch_dist": float(self.sld_pinch.value()),
            "two_split_min": float(self.sld_two.value())
        }
        self.settings["mirror_view"] = bool(self.chk_mirror_view.isChecked())
        self.settings["mirror_controls"] = bool(self.chk_mirror_controls.isChecked())
        self.settings["debug_draw"] = bool(self.chk_debug_draw.isChecked())
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Saved", "settings.json saved.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not save settings.json:\n{e}")

    # ---------- UI handlers ----------
    def apply_controls_to_settings(self):
        self.settings["mirror_view"] = self.chk_mirror_view.isChecked()
        self.settings["mirror_controls"] = self.chk_mirror_controls.isChecked()
        self.settings["debug_draw"] = self.chk_debug_draw.isChecked()
        self.settings.setdefault("thresholds", {})
        self.settings["thresholds"]["pinch_dist"] = float(self.sld_pinch.value())
        self.settings["thresholds"]["two_split_min"] = float(self.sld_two.value())

        if self.worker and self.worker.isRunning():
            self.read_table_into_settings()
            self.worker.update_settings(self.settings)

    def reset_defaults(self):
        self.settings = DEFAULT_SETTINGS.copy()
        self.settings["thresholds"] = DEFAULT_SETTINGS["thresholds"].copy()
        self.chk_mirror_view.setChecked(self.settings["mirror_view"])
        self.chk_mirror_controls.setChecked(self.settings["mirror_controls"])
        self.chk_debug_draw.setChecked(self.settings["debug_draw"])
        self.sld_pinch.setValue(self.settings["thresholds"]["pinch_dist"])
        self.sld_two.setValue(self.settings["thresholds"]["two_split_min"])
        self.populate_assignments_table()
        self.apply_controls_to_settings()

    def reload_from_file(self):
        self.settings = self.load_settings_from_file()
        self.chk_mirror_view.setChecked(self.settings.get("mirror_view", True))
        self.chk_mirror_controls.setChecked(self.settings.get("mirror_controls", False))
        self.chk_debug_draw.setChecked(self.settings.get("debug_draw", True))
        self.sld_pinch.setValue(self.settings.get("thresholds", {}).get("pinch_dist", 0.05))
        self.sld_two.setValue(self.settings.get("thresholds", {}).get("two_split_min", 0.02))
        self.populate_assignments_table()
        self.apply_controls_to_settings()

    def save_to_file(self):
        self.save_settings_to_file()

    def on_external_settings_changed(self, path):
        QtWidgets.QMessageBox.information(self, "settings.json", "External change detected. Reloading.")
        self.reload_from_file()
        if self.worker and self.worker.isRunning():
            self.worker.update_settings(self.settings)

    # ---------- start / stop worker ----------
    def start_worker(self):
        if self.worker is None or not self.worker.isRunning():
            self.read_table_into_settings()
            self.settings["thresholds"]["pinch_dist"] = float(self.sld_pinch.value())
            self.settings["thresholds"]["two_split_min"] = float(self.sld_two.value())

            self.worker = VideoWorker(self.settings)
            self.worker.frame_ready.connect(self.on_frame)
            self.worker.status_update.connect(self.on_status)
            self.worker.settings_changed.connect(lambda s: print("Worker updated settings"))
            self.worker.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._blacken_video()

    # ---------- video frame display ----------
    def on_frame(self, qimg):
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def on_status(self, text):
        self.status_label.setText("Status: " + text)

    def _blacken_video(self):
        pm = QtGui.QPixmap(self.video_label.size())
        pm.fill(QtGui.QColor("#000000"))
        self.video_label.setPixmap(pm)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()


# ---------- Run ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
