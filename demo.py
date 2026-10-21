import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os

# MediaPipe Tasks API imports
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils

# ---------------- YOLO ----------------
yolo_model = YOLO("yolov8n.pt")

IMPORTANT_OBJECTS = ["bottle", "cup", "cell phone"]

# ---------------- MEDIAPIPE SETUP ----------------
MODEL_PATH = "holistic_landmarker.task"

# Create a detector
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5
)
detector = mp_vision.HolisticLandmarker.create_from_options(options)

# Drawing specs
POSE_CONNECTIONS = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS

# ---------------- HELPERS ----------------

def get_hand_center(hand_landmarks):
    """
    In Holistic Tasks API, hand_landmarks is a List[NormalizedLandmark] for a single person.
    """
    if hand_landmarks and len(hand_landmarks) > 0:
        coords = [(lm.x, lm.y) for lm in hand_landmarks]
        return np.mean(coords, axis=0)
    return None

def bbox_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1+x2)/2, (y1+y2)/2])

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def log_event(event):
    with open("activity_log.txt", "a") as f:
        f.write(f"{datetime.now()} - {event}\n")

# ---------------- ACTION LOGIC ----------------

def infer_action(results, objects_with_pos):
    action = "idle"

    # Tasks API for Holistic: results.right_hand_landmarks is List[NormalizedLandmark]
    right_hand = get_hand_center(results.right_hand_landmarks)
    left_hand  = get_hand_center(results.left_hand_landmarks)

    hands = [h for h in [right_hand, left_hand] if h is not None]

    for obj, center in objects_with_pos:
        for hand in hands:
            if distance(hand, center) < 0.1:  # threshold
                if obj in ["bottle", "cup"]:
                    return "drinking"
                if obj == "cell phone":
                    return "using phone"

    return action

# ---------------- MAIN ----------------

cap = cv2.VideoCapture(0)

last_action = ""
last_logged_time = 0

print("Alzheimer Assistant Started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ---------------- YOLO ----------------
    yolo_results = yolo_model(frame, verbose=False)

    objects_with_pos = []

    for r in yolo_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            if label in IMPORTANT_OBJECTS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center = bbox_center((x1/w, y1/h, x2/w, y2/h))
                objects_with_pos.append((label, center))

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0), 2)

    # ---------------- MEDIAPIPE ----------------
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect
    results = detector.detect(mp_image)

    # Draw pose landmarks if detected
    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            POSE_CONNECTIONS
        )

    # ---------------- ACTION ----------------
    action = infer_action(results, objects_with_pos)

    # ---------------- LOGGING ----------------
    current_time = time.time()

    if action != last_action and current_time - last_logged_time > 3:
        log_event(action)
        last_logged_time = current_time
        last_action = action

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, f"Action: {action}",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,255), 2)

    cv2.imshow("Alzheimer Assistant", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()