# ==============================================================================
#  PREMIUM HOLISTIC ASSISTANT (Classy Edition)
# ==============================================================================
#  Features:
#    - Face Mesh (Tesselation)
#    - Hand Tracking (Left & Right)
#    - Pose Estimation (Body)
#    - Real-time Performance (FPS counter)
#    - Action Inference Status
# ==============================================================================

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

# MediaPipe Tasks API
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils
from distance_calculator import analyze_mouth_and_fingertips

# ── CONFIGURATION ───────────────────────────────────────────────────────────
MODEL_PATH = "holistic_landmarker.task"

# Modern Color Palette (Aesthetics)
COLOR_FACE_TESSELATION = (121, 22, 76)   # Deep Rose
COLOR_POSE_JOINTS      = (245, 117, 66)  # Orange
COLOR_POSE_LINES       = (245, 66, 230)  # Pink
COLOR_HAND_JOINTS      = (121, 22, 76)   # Deep Rose
COLOR_HAND_LINES       = (250, 44, 250)  # Magenta
COLOR_TEXT             = (0, 255, 255)   # Cyan
COLOR_BG_OVERLAY       = (20, 20, 20)    # Dark Gray

# Drawing Specs
def get_spec(color, thickness=2, radius=2):
    return drawing_utils.DrawingSpec(color=color, thickness=thickness, circle_radius=radius)

# ── CONNECTIONS ─────────────────────────────────────────────────────────────
POSE_CONNECTIONS = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS

# ── ASSISTANT LOGIC ──────────────────────────────────────────────────────────

def infer_simple_action(results):
    """
    Infers state using Face Mesh and exact Hand Fingertips.
    """
    # Ensure face is detected
    if results.face_landmarks:
        face = results.face_landmarks
        
        # Safely extract hands if they are visible
        left_hand = results.left_hand_landmarks if results.left_hand_landmarks else None
        right_hand = results.right_hand_landmarks if results.right_hand_landmarks else None
        
        # Run math from distance_calculator
        metrics = analyze_mouth_and_fingertips(face, left_hand, right_hand)
        return metrics
            
    return {
        "action": "Idle (No Face)",
        "mouth_state": "N/A",
        "mouth_gap_mm": 0.0,
        "mouth_gap_pct": 0.0,
        "finger_dist_mm": 999.0,
        "finger_dist_pct": 100.0
    }

# ── CORE CLASS ──────────────────────────────────────────────────────────────

class ClassyHolistic:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5
        )
        self.detector = mp_vision.HolisticLandmarker.create_from_options(options)
        self.prev_time = 0

    def draw_ui_overlay(self, frame, metrics, fps):
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), COLOR_BG_OVERLAY, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # UI Text
        cv2.putText(frame, "PREMIUM HOLISTIC ASSISTANT", (20, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_TEXT, 1)
        
        action = metrics["action"]
        status_color = (0, 255, 0) if "Idle" in action else (0, 165, 255)
        cv2.putText(frame, f"STATUS: {action.upper()}", (20, 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 1)
                    
        # Add new mm and percentage stats
        stats_text1 = f"Mouth Gap: {metrics['mouth_gap_mm']:.1f}mm ({metrics['mouth_gap_pct']:.1f}%) | State: {metrics['mouth_state']}"
        cv2.putText(frame, stats_text1, (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        dist_str = f"{metrics['finger_dist_mm']:.1f}mm" if metrics['finger_dist_mm'] != 999.0 else "N/A"
        stats_text2 = f"Finger to Mouth: {dist_str} ({metrics['finger_dist_pct']:.1f}%)"
        cv2.putText(frame, stats_text2, (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def process_frame(self, frame):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        # Drawing
        
        # Face
        if results.face_landmarks:
            drawing_utils.draw_landmarks(
                frame, results.face_landmarks, None,
                get_spec(COLOR_FACE_TESSELATION, 1, 1),
                get_spec(COLOR_FACE_TESSELATION, 1, 1)
            )

        # Hands
        if results.left_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, HAND_CONNECTIONS,
                get_spec(COLOR_HAND_JOINTS, 2, 4),
                get_spec(COLOR_HAND_LINES, 2, 2)
            )
        if results.right_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, HAND_CONNECTIONS,
                get_spec(COLOR_HAND_JOINTS, 2, 4),
                get_spec(COLOR_HAND_LINES, 2, 2)
            )

        # Pose
        if results.pose_landmarks:
            drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, POSE_CONNECTIONS,
                get_spec(COLOR_POSE_JOINTS, 2, 4),
                get_spec(COLOR_POSE_LINES, 2, 2)
            )

        metrics = infer_simple_action(results)
        self.draw_ui_overlay(frame, metrics, fps)
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Starting Classy Holistic... Press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Premium Holistic Assistant', processed_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    def close(self):
        self.detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    assistant = None
    try:
        assistant = ClassyHolistic()
        assistant.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if assistant:
            assistant.close()
