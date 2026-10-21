# ==============================================================================
#  ALZHEIMER MEDICATION MONITORING ASSISTANT — MASTER APP  (app.py)
#  Integrates: MediaPipe Holistic + YOLOv8 + Pill Classifier +
#              Twilio SMS + Firebase + Biometric Analytics
# ==============================================================================

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
import threading
import pyttsx3
import tensorflow as tf

# YOLO Object Detection
from ultralytics import YOLO

# MediaPipe Tasks API
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils

# Custom Mathematical Analysis
from distance_calculator import analyze_mouth_and_fingertips, analyze_object_interaction

# ── NEW: Import All Extension Modules ───────────────────────────────────────
from twilio_module   import (send_medication_taken_sms, send_missed_medication_sms,
                              send_no_activity_alert, send_daily_summary_sms,
                              get_active_slot, MED_SLOTS)
from firebase_module import MedicationVideoRecorder, upload_graph_to_firebase
from analytics_module import BiometricAnalytics
# from face_detect import FaceVerifier   # ← DISABLED for testing

# Pill Labels from MobileNet
PILL_LABELS = [
    'Amoxicillin 500 mg', 'Apixaban 2.5 mg', 'Aprepitant 80 mg', 'Atomoxetine 25 mg',
    'Calcitriol 0.00025', 'Prasugrel 10 MG', 'Ramipril 5 MG', 'Saxagliptin 5 MG',
    'Sitagliptin 50 MG', 'Tadalafil 5 MG', 'carvedilol 3.125', 'celecoxib 200',
    'duloxetine 30', 'eltrombopag 25', 'metformin_500', 'montelukast-10',
    'mycophenolate-250', 'omeprazole_40', 'oseltamivir-45', 'pantaprazole-40',
    'pitavastatin_1', 'prednisone_5', 'sertraline_25'
]

# ── CONFIGURATION ───────────────────────────────────────────────────────────
HOLISTIC_MODEL_PATH = "holistic_landmarker.task"
YOLO_MODEL_PATH     = "yolov8n.pt"

IMPORTANT_OBJECTS = ["bottle", "cup", "cell phone", "book", "laptop", "keyboard", "mouse",
                     "medicine", "pill", "tablet"]

# Smart confirmation thresholds (Module 6)
PILL_PIXEL_GAP_MIN  = 15      # fingers must be > 15 px apart (holding something)
PILL_PIXEL_GAP_MAX  = 80      # and < 80 px apart (small object, not a fist)
MOUTH_OPEN_MM_MIN   = 2.0     # mouth must be open ≥ 2 mm (relaxed for demo)
HAND_MOUTH_MM_MAX   = 80.0    # hand must be within 80 mm of mouth (relaxed for demo)
CONFIRM_FRAMES      = 5       # (legacy — kept for reference)
PINCH_CONFIRM_SECS  = 5.0    # CONTINUOUS seconds required to confirm a dose
                              # (the streak resets to 0 if pinch breaks even for one frame)
PINCH_RATIO_THRESH  = 0.35   # 65%-closed threshold so a loose pill-grip also fires

# Idle alert: SMS if no movement for this many seconds
IDLE_ALERT_SECONDS  = 300     # 5 minutes

# Premium UI Palette
COLOR_FACE_MESH   = (121, 22, 76)
COLOR_POSE_JOINTS = (245, 117, 66)
COLOR_POSE_LINES  = (245, 66, 230)
COLOR_HAND_JOINTS = (121, 22, 76)
COLOR_HAND_LINES  = (250, 44, 250)
COLOR_TEXT        = (0, 255, 255)
COLOR_BG          = (20, 20, 20)
COLOR_PILL_BOX    = (0, 0, 255)
COLOR_DRINK_BOX   = (0, 255, 0)
COLOR_WARN        = (0, 165, 255)

POSE_CONNECTIONS = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


def get_spec(color, thickness=2, radius=2):
    return drawing_utils.DrawingSpec(color=color, thickness=thickness, circle_radius=radius)


def get_hand_center(hand_landmarks):
    if hand_landmarks and len(hand_landmarks) > 0:
        coords = [(lm.x, lm.y) for lm in hand_landmarks]
        return np.mean(coords, axis=0)
    return None


def bbox_center_norm(box, w, h):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / (2 * w), (y1 + y2) / (2 * h)])


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def log_event(event: str):
    with open("activity_log.txt", "a") as f:
        f.write(f"{datetime.now()} — {event}\n")


# ==============================================================================
# ==============================================================================
#  STANDALONE PILL CLASSIFIER  (ported from PillDetection/flask-app — no Flask)
#  Mirrors the exact predict() logic from the Pill-Detection-System:
#    1. Crop the palm region from the live camera frame
#    2. Resize to 224×224 and normalise (/255)
#    3. Run MobileNet → top-3 predictions with confidence
#    4. Draw a bounding box + label directly on the frame
# ==============================================================================

class PillClassifier:
    """Drop-in replacement for the Flask pill identifier, runs entirely in-process."""

    LABELS = PILL_LABELS   # shared list at top of file

    def __init__(self, model):
        self.model = model

    def predict_crop(self, frame, x1: int, y1: int, x2: int, y2: int):
        """
        Given a bounding-box (x1,y1)-(x2,y2) in pixel coords, crop, resize, run
        MobileNet and return (pill_name, confidence, top3_list).

        top3_list  →  [(name, conf_pct), ...] in descending order
        """
        if self.model is None:
            return None, 0.0, []
        if x2 <= x1 or y2 <= y1:
            return None, 0.0, []
        try:
            crop   = frame[y1:y2, x1:x2]
            crop_r = cv2.resize(crop, (224, 224))
            inp    = np.expand_dims(crop_r, axis=0) / 255.0
            pred   = self.model.predict(inp, verbose=0)[0]          # shape (23,)

            # mirror the Flask /results sort: top-3 indices descending
            top3_idx = pred.argsort()[-3:][::-1]
            top3     = [(self.LABELS[i], float(pred[i]) * 100) for i in top3_idx]
            best_name, best_conf_pct = top3[0]
            return best_name, best_conf_pct / 100.0, top3
        except Exception as ex:
            print(f"[PillClassifier] {ex}")
            return None, 0.0, []

    def draw_pill_box(self, frame, x1, y1, x2, y2, pill_name, conf, top3):
        """Draw bounding box + top-3 labels on frame in-place."""
        # Main box — blue (same as COLOR_PILL_BOX)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Primary label above box
        label_str = f"PILL: {pill_name}  ({conf*100:.0f}%)"
        cv2.putText(frame, label_str,
                    (x1, max(y1 - 12, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Top-3 mini list inside bottom of box
        for rank, (name, pct) in enumerate(top3):
            yy = y2 - 8 - rank * 18
            if yy < y1:
                break
            cv2.putText(frame, f"{rank+1}. {name}: {pct:.1f}%",
                        (x1 + 4, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 255), 1)


# ==============================================================================
#  MAIN APPLICATION CLASS
# ==============================================================================

class ComprehensiveApp:

    def __init__(self):
        print("🔄 Loading AI Models…")

        # 1. YOLO
        self.yolo_model = YOLO(YOLO_MODEL_PATH)

        # 2. MediaPipe Holistic
        if not os.path.exists(HOLISTIC_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {HOLISTIC_MODEL_PATH}")
        base_options = mp_python.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self.holistic_detector = mp_vision.HolisticLandmarker.create_from_options(options)

        # 3. Local Pill Image Classifier
        try:
            self.pill_model = tf.keras.models.load_model("MobileNet.keras")
            print("✅ Local Pill Model loaded!")
        except Exception as e:
            self.pill_model = None
            print(f"⚠️  Pill model unavailable: {e}")

        # ── State ─────────────────────────────────────────────────────────────
        self.taken_pills:    set  = set()
        self.pill_log:       list = []           # (timestamp, pill_name) tuples
        self.pill_cooldown:  float = 0.0

        self.last_action:      str   = "Idle"
        self.last_logged_time: float = 0.0
        self.prev_time:        float = 0.0

        # ── Cumulative pinch banking (replaces consecutive-frame counter) ──────────────
        # Pinch time accumulates across the whole 50-second session.
        # 2 seconds total (even with gaps) = dose confirmed.
        self.med_pinch_accum:   float = 0.0    # total seconds of valid pinch so far
        self.med_last_pinch_ts: float = 0.0    # wall-clock time of the previous pinch frame
        self.drink_confirm_count: int  = 0      # (unused — kept for legacy compat)

        # Idle / no-activity tracking
        self.last_motion_time:  float = time.time()
        self.idle_alert_sent:   bool  = False

        # Medication slot tracking
        self.slot_status: dict = {slot: None for slot in MED_SLOTS}  # None / "taken" / "missed"
        self.slot_sms_sent: set = set()

        # Extension modules
        self.recorder    = MedicationVideoRecorder()
        self.analytics   = BiometricAnalytics()
        self.pill_clf    = PillClassifier(self.pill_model)  # standalone Flask-free classifier
        # self.face_verifier = FaceVerifier()  # ← DISABLED for testing

        print("✅ All systems online.\n")

    # ── VOICE ─────────────────────────────────────────────────────────────────

    def speak(self, text: str):
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[Voice] {e}")
        threading.Thread(target=_speak, daemon=True).start()

    # ── MEDICATION EVENT ─────────────────────────────────────────────────────

    def _on_medication_confirmed(self, pill_name: str, confidence: float):
        """
        Called exactly once when the system has high confidence that
        the patient just took their medication.
        """
        now_str = datetime.now().strftime("%I:%M %p")
        log_entry = f"{now_str} — {pill_name}"
        self.pill_log.append(log_entry)

        if pill_name in self.taken_pills:
            # ── REPETITION ALARM ────────────────────────────────────────────
            msg = f"There is a repetition. Please do not take {pill_name} again."
            self.speak(msg)
            log_event(f"ALARM: Repeated dose — {pill_name}")
            print(f"🚨 ALARM: Repetition — {pill_name}")
        else:
            # ── FIRST DOSE: record it ────────────────────────────────────────
            self.taken_pills.add(pill_name)
            self.speak(f"{pill_name} detected. Dose recorded successfully.")
            log_event(f"RECORDED: {pill_name} ({confidence*100:.0f}% confidence)")
            print(f"✅ RECORDED: {pill_name}")

            # (SMS sent by run.py based on mode — not here)
            log_event(f"RECORDED: {pill_name} ({confidence*100:.0f}% confidence)")

            # Mark active slot as taken
            active = get_active_slot()
            if active:
                self.slot_status[active] = "taken"

        # Analytics: stop recording clip and upload
        self.analytics_session_end(success=True, pill_name=pill_name)

    def analytics_session_end(self, success: bool, pill_name: str = ""):
        summary    = self.analytics.get_summary()
        confidence = summary["completion_confidence"]
        log_event(f"SESSION END | {pill_name} | confidence={confidence}% | {summary}")

        # Stop video recording and upload to Firebase
        self.recorder.stop_and_upload(success=success)

        # Generate graphs and upload to Firebase
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_paths = self.analytics.generate_graphs(prefix=prefix)
        for gp in graph_paths:
            threading.Thread(
                target=upload_graph_to_firebase, args=(gp,), daemon=False
            ).start()

        self.analytics.reset()

    # ── YOLO ACTION INFERENCE ────────────────────────────────────────────────

    def infer_yolo_action(self, holistic_results, objects_with_pos_norm):
        right_hand = get_hand_center(holistic_results.right_hand_landmarks)
        left_hand  = get_hand_center(holistic_results.left_hand_landmarks)
        hands      = [h for h in [right_hand, left_hand] if h is not None]

        pose_landmarks = None
        if holistic_results.pose_landmarks and len(holistic_results.pose_landmarks) > 0:
            # Pass the entire list of landmarks, not just the 0th element (which would be the nose)
            pose_landmarks = holistic_results.pose_landmarks

        for obj_label, obj_center_norm in objects_with_pos_norm:
            # Object interaction analysis using shoulder/eye distances
            if obj_label in ["bottle", "cup", "wine glass", "glass"] and pose_landmarks:
                metrics = analyze_object_interaction(
                    obj_center_norm[0], obj_center_norm[1], pose_landmarks, obj_label
                )
                if metrics.get("action"):
                    return metrics["action"]

            # Fallback: hand proximity check
            for hand in hands:
                if distance(hand, obj_center_norm) < 0.1:
                    if obj_label in ["bottle", "cup", "glass", "wine glass"]:
                        return f"Holding {obj_label}"
                    if obj_label == "cell phone":
                        return "Using Phone"

            if obj_label == "book":      return "Reading Book"
            if obj_label == "laptop":    return "Working on Computer"
            if obj_label == "keyboard":  return "Typing"

        return None

    # ── UI OVERLAY ────────────────────────────────────────────────────────────

    def draw_ui_overlay(self, frame, fps, holistic_metrics, final_action):
        h_f, w_f = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_f, 155), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Title bar
        cv2.putText(frame, "🧠 ALZHEIMER MEDICATION MONITOR",
                    (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (w_f - 120, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_TEXT, 1)

        # Action status
        act_color = (0, 255, 0) if "Idle" in final_action else (0, 165, 255)
        cv2.putText(frame, f"STATUS: {final_action.upper()}",
                    (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, act_color, 1)

        if holistic_metrics:
            cv2.putText(frame,
                f"Mouth: {holistic_metrics['mouth_gap_mm']:.1f}mm  "
                f"({holistic_metrics['mouth_state']})  |  "
                f"Finger→Mouth: {holistic_metrics['finger_dist_mm']:.1f}mm",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

            p_color = (0, 255, 255) if holistic_metrics.get("is_pinching") else (180, 180, 180)
            cv2.putText(frame,
                f"Medicine Hold: {'ACTIVE — Holding Medication' if holistic_metrics.get('is_pinching') else 'None'}",
                (20, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.48, p_color, 1)


        # Continuous-streak progress bar (replaces old confirm-frame bar)
        if self.med_pinch_accum > 0:
            bar_w = int((self.med_pinch_accum / PINCH_CONFIRM_SECS) * 300)
            cv2.rectangle(frame, (20, 118), (20 + 300, 130), (60, 60, 60), -1)
            cv2.rectangle(frame, (20, 118), (20 + bar_w, 130), (0, 200, 255), -1)
            cv2.putText(frame,
                        f"Pinch streak: {self.med_pinch_accum:.1f}s / {PINCH_CONFIRM_SECS:.0f}s — hold steady!",
                        (20, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Slot status indicators (top-right)
        slot_labels = ["10:00 AM", "2:00 PM", "8:00 PM"]
        for idx, (slot, label) in enumerate(zip(MED_SLOTS, slot_labels)):
            state = self.slot_status.get(slot)
            icon  = "✅" if state == "taken" else "❌" if state == "missed" else "🕐"
            cv2.putText(frame, f"{icon} {label}",
                        (w_f - 165, 60 + idx * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # ── MAIN FRAME PROCESSOR ─────────────────────────────────────────────────

    def process_frame(self, frame):
        h, w, _ = frame.shape
        curr_time = time.time()
        fps = 1 / max(curr_time - self.prev_time, 0.001) if self.prev_time > 0 else 0
        self.prev_time = curr_time

        # ── 0. FACE VERIFICATION — DISABLED FOR TESTING ─────────────────────────
        # self.face_verifier.update(frame)
        # frame = self.face_verifier.draw_overlay(frame)
        # face_ok = self.face_verifier.is_verified or not self.face_verifier.face_images
        # if self.face_verifier.is_stranger:
        #     self.speak("Warning! Unrecognized person detected. Medication session paused.")
        face_ok = True  # Always True during testing

        # ── 1. YOLO DETECTION ─────────────────────────────────────────
        yolo_results = self.yolo_model(frame, verbose=False)
        objects_with_pos = []

        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = self.yolo_model.names[cls_id]
                if label in IMPORTANT_OBJECTS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_norm = bbox_center_norm((x1, y1, x2, y2), w, h)
                    objects_with_pos.append((label, center_norm))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DRINK_BOX, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DRINK_BOX, 2)

        # ── 2. HOLISTIC DETECTION ─────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        holistic_results = self.holistic_detector.detect(mp_image)

        if holistic_results.face_landmarks:
            drawing_utils.draw_landmarks(
                frame, holistic_results.face_landmarks, None,
                get_spec(COLOR_FACE_MESH, 1, 1), get_spec(COLOR_FACE_MESH, 1, 1))
        if holistic_results.left_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame, holistic_results.left_hand_landmarks, HAND_CONNECTIONS,
                get_spec(COLOR_HAND_JOINTS, 2, 4), get_spec(COLOR_HAND_LINES, 2, 2))
        if holistic_results.right_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame, holistic_results.right_hand_landmarks, HAND_CONNECTIONS,
                get_spec(COLOR_HAND_JOINTS, 2, 4), get_spec(COLOR_HAND_LINES, 2, 2))
        if holistic_results.pose_landmarks:
            drawing_utils.draw_landmarks(
                frame, holistic_results.pose_landmarks, POSE_CONNECTIONS,
                get_spec(COLOR_POSE_JOINTS, 2, 4), get_spec(COLOR_POSE_LINES, 2, 2))

        # ── 3. SPATIAL ANALYSIS ───────────────────────────────────────────────
        holistic_metrics = None
        holistic_action  = "Idle"

        if holistic_results.face_landmarks:
            face       = holistic_results.face_landmarks
            left_hand  = holistic_results.left_hand_landmarks
            right_hand = holistic_results.right_hand_landmarks

            holistic_metrics = analyze_mouth_and_fingertips(face, left_hand, right_hand)
            holistic_action  = holistic_metrics["action"]

            # Compute mm_per_unit for analytics module
            import math
            left_edge   = face[234]
            right_edge  = face[454]
            face_w_norm = math.sqrt((right_edge.x - left_edge.x)**2 + (right_edge.y - left_edge.y)**2)
            mm_per_unit = 150.0 / max(face_w_norm, 0.001)

            # Nearest object center for analytics
            nearest_obj = objects_with_pos[0][1] if objects_with_pos else None

            # Update biometric analytics every frame
            self.analytics.update(
                face_landmarks    = face,
                left_hand         = left_hand,
                right_hand        = right_hand,
                mm_per_unit       = mm_per_unit,
                object_center_xy  = nearest_obj,
            )

        # ── MODULE 6: SMART PILL DETECTION (3D Normalised Pinch — Camera-Invariant) ─
        # Instead of 2-D pixels (which break when the hand is far from the camera),
        # we use the full 3-D Euclidean formula from VPython: dist = sqrt(Δx²+Δy²+Δz²)
        # and compare it to the wrist→middle-tip hand span so the threshold scales
        # automatically with camera distance.
        # Pinch fires when:  pinch_3d / hand_size_3d  < 0.20  (≡ 80% fingers closed)
        if holistic_results.face_landmarks and self.pill_model:
            raw_hand = (holistic_results.right_hand_landmarks
                        or holistic_results.left_hand_landmarks)

            # MediaPipe Tasks API returns a list of hands. Extract the first hand’s landmarks.
            hand = None
            if raw_hand and len(raw_hand) > 0:
                hand = raw_hand[0] if isinstance(raw_hand[0], list) else raw_hand

            if hand and len(hand) > 12 and holistic_metrics:
                thumb  = hand[4]    # thumb tip
                index  = hand[8]    # index fingertip
                wrist  = hand[0]    # wrist
                middle = hand[12]   # middle-finger tip (hand-size reference)

                # — 3-D Euclidean distances in normalised landmark space —
                # (same formula as VPython mag(d) = sqrt(Δx²+Δy²+Δz²))
                pinch_3d = float(np.sqrt(
                    (thumb.x - index.x)**2 +
                    (thumb.y - index.y)**2 +
                    (thumb.z - index.z)**2
                ))
                hand_size_3d = float(np.sqrt(
                    (wrist.x - middle.x)**2 +
                    (wrist.y - middle.y)**2 +
                    (wrist.z - middle.z)**2
                ))

                # Dynamic ratio: 0.0 = fully closed, 1.0 = fully open
                pinch_ratio = (pinch_3d / hand_size_3d) if hand_size_3d > 1e-6 else 1.0
                pinch_pct   = pinch_ratio * 100.0          # % of hand span
                closeness   = max(0.0, 1.0 - pinch_ratio)  # 1.0 = fully closed

                # 65%-closed threshold (PINCH_RATIO_THRESH = 0.35 from constants above)
                cond_pinch = (pinch_ratio < PINCH_RATIO_THRESH)

                # Pixel coords for drawing (2-D only, display purposes)
                thumb_px = (int(thumb.x * w), int(thumb.y * h))
                index_px = (int(index.x * w), int(index.y * h))

                finger_dist = holistic_metrics.get("finger_dist_mm", 999.0)
                mouth_open  = holistic_metrics.get("mouth_gap_mm",   0.0)

                # ── Live debug HUD (bottom-left) ─────────────────────────────────
                debug_y = h - 160
                hud_lines = [
                    (f"Medicine Hold: {pinch_pct:.1f}%  "
                     f"{'ACTIVE ✔' if cond_pinch else 'not detected'}",
                     cond_pinch),
                    (f"Hold streak: {self.med_pinch_accum:.2f}s / {PINCH_CONFIRM_SECS:.1f}s",
                     self.med_pinch_accum > 0),
                    (f"Mouth: {mouth_open:.1f}mm  ({'open' if mouth_open >= MOUTH_OPEN_MM_MIN else 'closed'})",
                     mouth_open >= MOUTH_OPEN_MM_MIN),
                ]
                for hud_label, good in hud_lines:
                    cv2.putText(frame, hud_label, (10, debug_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (0, 255, 80) if good else (0, 80, 255), 2)
                    debug_y += 28

                # Draw live thumb↔index line (green = medicine hold active)
                line_color = (0, 255, 80) if cond_pinch else (0, 200, 255)
                cv2.line(frame, thumb_px, index_px, line_color, 2)
                cv2.circle(frame, thumb_px, 7, (0, 200, 255), -1)
                cv2.circle(frame, index_px, 7, (255, 200, 0), -1)

                # Medicine Hold streak progress bar
                bar_pct   = min(1.0, self.med_pinch_accum / PINCH_CONFIRM_SECS)
                bar_w     = int(bar_pct * 300)
                bar_color = (0, 255, 80) if bar_pct >= 1.0 else (0, 200, 255)
                cv2.rectangle(frame, (10, h - 22), (310, h - 8), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, h - 22), (10 + bar_w, h - 8), bar_color, -1)
                cv2.putText(frame,
                            f"Medicine Hold: {self.med_pinch_accum:.1f}s / {PINCH_CONFIRM_SECS:.1f}s",
                            (10, h - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

                # Bypass mouth/distance checks (testing mode)
                cond_mouth = True
                cond_close = True

                if cond_pinch and cond_mouth and cond_close and face_ok:
                    # ── PALM BOUNDING BOX (all 21 landmarks) ─────────────────────
                    all_xs = [int(lm.x * w) for lm in hand]
                    all_ys = [int(lm.y * h) for lm in hand]
                    pad    = 20
                    bx1 = max(0, min(all_xs) - pad)
                    by1 = max(0, min(all_ys) - pad)
                    bx2 = min(w, max(all_xs) + pad)
                    by2 = min(h, max(all_ys) + pad)

                    # ── RUN PILL CLASSIFIER ───────────────────────────────────────
                    pill, conf, top3 = self.pill_clf.predict_crop(frame, bx1, by1, bx2, by2)

                    if pill is not None:
                        # Draw bounding box + top-3 list
                        self.pill_clf.draw_pill_box(frame, bx1, by1, bx2, by2, pill, conf, top3)

                        # ── STRICT CONTINUOUS STREAK ──────────────────────────────
                        if self.med_last_pinch_ts > 0:
                            self.med_pinch_accum += curr_time - self.med_last_pinch_ts
                        self.med_last_pinch_ts = curr_time

                        if not self.recorder.is_recording:
                            self.recorder.start_recording(frame.shape, "Medication")
                            self.analytics.reset()
                        self.recorder.write_frame(frame, conf)

                        # ── FIRE when 5 continuous seconds reached ─────────────────
                        if self.med_pinch_accum >= PINCH_CONFIRM_SECS:
                            if curr_time - self.pill_cooldown > 5.0:
                                self._on_medication_confirmed(pill, conf)
                                self.pill_cooldown = curr_time
                            self.med_pinch_accum   = 0.0
                            self.med_last_pinch_ts = 0.0

                    else:
                        # Pill not classified — HARD RESET (strict)
                        self.med_pinch_accum   = 0.0
                        self.med_last_pinch_ts = 0.0
                else:
                    # Gesture broke — HARD RESET
                    self.med_pinch_accum   = 0.0
                    self.med_last_pinch_ts = 0.0




        # ── 4. YOLO-BASED ACTION INFERENCE ────────────────────────────────────
        yolo_action  = self.infer_yolo_action(holistic_results, objects_with_pos)
        final_action = yolo_action or (holistic_action if holistic_action != "Idle" else "Idle")

        if final_action != self.last_action and (curr_time - self.last_logged_time > 3):
            log_event(final_action)
            self.last_action      = final_action
            self.last_logged_time = curr_time
            self.last_motion_time = curr_time

        # ── MODULE 6: IDLE DETECTION ──────────────────────────────────────────
        idle_secs = int(curr_time - self.last_motion_time)
        if idle_secs >= IDLE_ALERT_SECONDS and not self.idle_alert_sent:
            # (no-activity SMS removed — SMS is mode-based in run.py)
            self.speak("No activity detected for several minutes. Please check on the patient.")
            self.idle_alert_sent = True
        elif idle_secs < 60:
            self.idle_alert_sent = False

        # ── MODULE 1: MISSED MEDICATION CHECK ─────────────────────────────────
        self._check_missed_slots()

        # ── 5. RENDER OVERLAY ─────────────────────────────────────────────────
        self.draw_ui_overlay(frame, fps, holistic_metrics, final_action)
        return frame

    # ── SLOT MISS CHECKER ─────────────────────────────────────────────────────

    def _check_missed_slots(self):
        """After the ±20-min window, mark slot as missed and send SMS.
        GUARD: never fires if medication has already been confirmed OR sim_mode is active."""
        if self.taken_pills or getattr(self, 'sim_mode', False):
            return
        from twilio_module import WINDOW_MINUTES
        from datetime import datetime, time as dtime

        now  = datetime.now()
        now_t = now.time()

        # Prevent alerting for slots that were ALREADY missed before the app was even turned on.
        if not hasattr(self, "_app_start_time"):
            self._app_start_time = now

        for slot in MED_SLOTS:
            if slot in self.slot_sms_sent:
                continue
            slot_dt  = datetime.combine(now.date(), slot)
            
            # If this slot's window ended BEFORE we even launched the app, skip it silently
            if slot_dt < self._app_start_time and (self._app_start_time - slot_dt).total_seconds() / 60 > WINDOW_MINUTES:
                self.slot_status[slot] = "missed"
                self.slot_sms_sent.add(slot)
                continue

            delta_m  = (now - slot_dt).total_seconds() / 60

            # Past the window and not yet taken
            if WINDOW_MINUTES < delta_m < 60 and self.slot_status[slot] is None:
                self.slot_status[slot] = "missed"
                self.slot_sms_sent.add(slot)
                # (SMS sent by run.py based on mode — not here)
                self.speak("Medication was missed. Caregiver will be notified at session end.")
                log_event(f"MISSED: slot {slot}")

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────

    def run(self):
        # ── SIMULATION MODE FLAG ──────────────────────────────────────────────
        # Set by run.py (mode S) via a .sim_mode flag file.
        # When True: the full monitoring process runs normally, but every exit
        # path (timeout / Q / KeyboardInterrupt) sends a SUCCESS SMS instead
        # of a missed-dose alert.
        self.sim_mode = os.path.exists(".sim_mode")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ── PHASE 0: FACE VERIFICATION ────────────────────────────────────────
        # Runs before the 50s window. Only when patient's face is confirmed
        # (LBPH match held for 2s) does medication monitoring begin.
        try:
            from patient_verification import PatientVerifier
            _root = os.path.dirname(os.path.abspath(__file__))

            # Get patient name written by run.py
            patient_name = ""
            cfg = os.path.join(_root, ".current_patient.txt")
            if os.path.exists(cfg):
                patient_name = open(cfg, encoding="utf-8").read().strip()

            # Auto-find first patient folder with images if name not set
            if not patient_name:
                base = os.path.join(_root, "assets", "patient")
                if os.path.isdir(base):
                    for e in sorted(os.scandir(base), key=lambda x: x.name):
                        if e.is_dir() and any(
                            f.lower().endswith((".png", ".jpg", ".jpeg"))
                            for f in os.listdir(e.path)
                        ):
                            patient_name = e.name
                            break

            patient_dir = os.path.join(_root, "assets", "patient", patient_name)
            verifier    = PatientVerifier(patient_dir)
            has_refs    = verifier.load_references()

            if has_refs:
                while cap.isOpened() and not verifier.is_verified:
                    ret, vframe = cap.read()
                    if not ret:
                        break
                    verifier.update(vframe, time.time())
                    cv2.imshow("Alzheimer Medication Monitor", vframe)
                    if cv2.waitKey(5) & 0xFF == ord("q"):
                        break
            else:
                print("[FaceVerify] No patient photos — skipping verification.\n")
        except Exception as _fv_err:
            print(f"[FaceVerify] Skipping — {_fv_err}\n")

        # ── PHASE 1: MEDICATION MONITORING (50-second window) ─────────────────
        SESSION_TIMEOUT   = 50
        POST_DETECT_WAIT  = 3
        session_start     = time.time()
        medication_done   = False
        done_time         = None


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            # ── 0. GLOBAL 2 FPS FRAME CAPTURE ────────────────────────────────
            # Capture exactly 2 frames per second globally during the session
            if not hasattr(self, '_global_cap'):
                self._global_cap = {
                    'last_ts': 0.0,
                    'idx': 0,
                    'prefix': datetime.now().strftime('%Y%m%d_%H%M%S')
                }
            if now - self._global_cap['last_ts'] >= (1.0 / 2.0):
                self._global_cap['last_ts'] = now
                try:
                    from firebase_module import save_captured_frame
                    # Pass a copy to avoid threading/modification issues
                    save_captured_frame(frame.copy(), self._global_cap['prefix'], self._global_cap['idx'])
                    self._global_cap['idx'] += 1
                except Exception as e:
                    print(f"Capture error: {e}")

            # ── 1. Check session timeout (180 s) ─────────────────────────────
            elapsed = now - session_start
            remaining = max(0, int(SESSION_TIMEOUT - elapsed))

            if not medication_done and elapsed > SESSION_TIMEOUT:
                if self.sim_mode:
                    self.speak("Simulation session complete. Medication marked as taken.")
                else:
                    self.speak("Medication monitoring session ended. Caregiver will be notified.")
                self._finalize_session(success=self.sim_mode)
                break

            # ── 2. Check post-detection 5-second wait ────────────────────────
            if medication_done and done_time and (now - done_time) >= POST_DETECT_WAIT:
                print(f"[Session] {POST_DETECT_WAIT}s post-detection wait complete. Finalizing…")
                self._finalize_session(success=True)
                break

            # ── 3. Process frame ─────────────────────────────────────────────
            processed = self.process_frame(frame)

            # Show session timer countdown on frame
            h_f, w_f = processed.shape[:2]
            timer_color = (0, 255, 80) if remaining > 60 else (0, 165, 255) if remaining > 30 else (0, 0, 255)
            cv2.putText(processed, f"Session: {remaining}s remaining",
                        (w_f - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

            # If medication was just confirmed, start the countdown
            if len(self.taken_pills) > 0 and not medication_done:
                medication_done = True
                done_time = now
                print(f"[Session] Medication confirmed! Waiting {POST_DETECT_WAIT}s then finalizing…")
                cv2.putText(processed, f"✓ DOSE CONFIRMED — stopping in {POST_DETECT_WAIT}s",
                            (10, h_f - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2)

            cv2.imshow("Alzheimer Medication Monitor", processed)

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q") or key == ord("g"):
                force_success = self.sim_mode or medication_done
                self._finalize_session(success=force_success)
                break

        cap.release()
        cv2.destroyAllWindows()

    # ── SESSION FINALIZER ─────────────────────────────────────────────────────

    def _finalize_session(self, success: bool):
        if self.recorder.is_recording:
            self.recorder.stop_and_upload(success=success)

        # 2. Generate graphs (saved locally to graphs/ folder)
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_paths = self.analytics.generate_graphs(prefix=prefix)

        upload_threads = []
        for gp in graph_paths:
            remote = f"graphs/{os.path.basename(gp)}"
            t = threading.Thread(
                target=upload_graph_to_firebase,
                args=(gp, remote),
                daemon=False
            )
            t.start()
            upload_threads.append(t)

        for t in upload_threads:
            t.join(timeout=30)

        taken    = len(self.taken_pills)
        pill_log = self.pill_log
        total    = len(MED_SLOTS)
        log_event(f"SESSION SUMMARY | taken={taken}/{total} | pills={pill_log}")

    def close(self):
        self.holistic_detector.close()


if __name__ == "__main__":
    app = None
    try:
        app = ComprehensiveApp()
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down…")
        if app:
            sim = getattr(app, "sim_mode", False)
            app._finalize_session(success=sim or len(app.taken_pills) > 0)
    except Exception as e:
        import traceback
        print(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        if app:
            app.close()
        # Always clean up the sim_mode flag
        if os.path.exists(".sim_mode"):
            os.remove(".sim_mode")
