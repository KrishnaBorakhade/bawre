# ==============================================================================
#  FACE DETECTION MODULE — face_detect.py
#
#  Uses DeepFace to continuously verify that the person in front of the camera
#  is the registered patient (≥ 40% confidence).
#
#  How it works:
#    1. On startup, it loads all face images from assets/patient/<name>/
#    2. Every N frames, it grabs a snapshot and runs DeepFace.verify()
#       against each stored face image.
#    3. It returns a FaceVerificationResult with:
#         - is_verified  : True if ≥ 1 image matched above threshold
#         - confidence   : best match confidence (0–100%)
#         - patient_name : from patient_info.txt
#         - is_stranger  : True if confidence < 40% for too long
# ==============================================================================

import os
import cv2
import time
import glob
import threading
import numpy as np

# DeepFace import with fallback warning
try:
    from deepface import DeepFace
    _DEEPFACE_AVAILABLE = True
except ImportError:
    print("[FaceDetect] WARNING: DeepFace not installed. Run: pip install deepface")
    _DEEPFACE_AVAILABLE = False

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
ASSETS_DIR         = "assets"
PATIENT_BASE_DIR   = os.path.join(ASSETS_DIR, "patient")
VERIFY_EVERY_N     = 30          # Check face every N frames (keeps camera fast)
MIN_CONFIDENCE_PCT = 40.0        # Minimum match confidence to accept as verified
STRANGER_ALERT_SEC = 10          # Alert if unverified for this many consecutive seconds
DEEPFACE_MODEL     = "VGG-Face"  # Options: VGG-Face, Facenet, Facenet512, ArcFace
DEEPFACE_METRIC    = "cosine"    # Distance metric


# ── PATIENT INFO LOADER ───────────────────────────────────────────────────────

def load_patient_info(patient_dir: str) -> dict:
    """Load patient details from patient_info.txt."""
    info = {"name": os.path.basename(patient_dir), "age": "?", "pills": [], "dosage": "?"}
    info_file = os.path.join(patient_dir, "patient_info.txt")
    if not os.path.exists(info_file):
        return info

    with open(info_file, encoding="utf-8") as f:
        lines = f.readlines()

    current_list = None
    for line in lines:
        line = line.strip()
        if line.startswith("name:"):
            info["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("age:"):
            info["age"] = line.split(":", 1)[1].strip()
        elif line.startswith("current_dosage:"):
            info["dosage"] = line.split(":", 1)[1].strip()
        elif line.startswith("pills:"):
            current_list = "pills"
        elif line.startswith("  - ") and current_list == "pills":
            info["pills"].append(line[4:].strip())

    return info


# ── MAIN VERIFIER CLASS ───────────────────────────────────────────────────────

class FaceVerifier:
    """
    Wraps DeepFace verification and runs it on a background thread
    so it never blocks the 30fps camera loop.
    """

    def __init__(self):
        self.patient_dir   = self._find_patient_dir()
        self.patient_info  = {}
        self.face_images   = []
        self.frame_count   = 0

        # Shared state updated by background thread
        self._confidence   = 0.0
        self._is_verified  = False
        self._is_stranger  = False
        self._last_good_ts = time.time()
        self._busy         = False   # True while DeepFace is running
        self._lock         = threading.Lock()

        if self.patient_dir:
            self.patient_info = load_patient_info(self.patient_dir)
            self.face_images  = sorted(
                glob.glob(os.path.join(self.patient_dir, "face_*.jpg"))
            )
            print(f"[FaceDetect] Loaded {len(self.face_images)} reference images "
                  f"for patient: {self.patient_info.get('name', '?')}")
        else:
            print("[FaceDetect] WARNING: No patient directory found. "
                  "Run capture_patient.py first.")

    # ── INTERNAL ─────────────────────────────────────────────────────────────

    @staticmethod
    def _find_patient_dir() -> str | None:
        """Auto-finds the first patient sub-folder under assets/patient/."""
        if not os.path.isdir(PATIENT_BASE_DIR):
            return None
        dirs = [
            os.path.join(PATIENT_BASE_DIR, d)
            for d in os.listdir(PATIENT_BASE_DIR)
            if os.path.isdir(os.path.join(PATIENT_BASE_DIR, d))
        ]
        return dirs[0] if dirs else None

    def _verify_in_background(self, frame_snapshot: np.ndarray):
        """Runs DeepFace on a copy of the current frame (background thread)."""
        if not _DEEPFACE_AVAILABLE or not self.face_images:
            return

        best_conf = 0.0
        verified  = False

        # Save temp image for DeepFace (it needs a file path)
        tmp_path = os.path.join(ASSETS_DIR, "_tmp_verify.jpg")
        cv2.imwrite(tmp_path, frame_snapshot)

        for ref_img in self.face_images[:5]:   # Check first 5 references (fast)
            try:
                result = DeepFace.verify(
                    img1_path   = tmp_path,
                    img2_path   = ref_img,
                    model_name  = DEEPFACE_MODEL,
                    distance_metric = DEEPFACE_METRIC,
                    enforce_detection = False,
                )
                # Convert distance to confidence %
                # cosine distance: 0 = perfect match, 1 = no match
                dist       = result.get("distance", 1.0)
                confidence = max(0.0, (1.0 - dist) * 100.0)
                if confidence > best_conf:
                    best_conf = confidence
                    verified  = result.get("verified", False) and confidence >= MIN_CONFIDENCE_PCT
            except Exception:
                pass   # Face not detected in frame — skip silently

        try:
            os.remove(tmp_path)
        except OSError:
            pass

        with self._lock:
            self._confidence  = round(best_conf, 1)
            self._is_verified = verified
            if verified:
                self._last_good_ts = time.time()
            else:
                elapsed = time.time() - self._last_good_ts
                self._is_stranger = elapsed >= STRANGER_ALERT_SEC
            self._busy = False

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray):
        """
        Call once per frame from app.py.
        Runs DeepFace every VERIFY_EVERY_N frames in the background.
        """
        self.frame_count += 1
        if not self._busy and self.face_images and (self.frame_count % VERIFY_EVERY_N == 0):
            snapshot = frame.copy()
            self._busy = True
            threading.Thread(
                target=self._verify_in_background,
                args=(snapshot,),
                daemon=True
            ).start()

    @property
    def confidence(self) -> float:
        with self._lock:
            return self._confidence

    @property
    def is_verified(self) -> bool:
        with self._lock:
            return self._is_verified

    @property
    def is_stranger(self) -> bool:
        with self._lock:
            return self._is_stranger

    @property
    def patient_name(self) -> str:
        return self.patient_info.get("name", "Unknown")

    @property
    def patient_pills(self) -> list:
        return self.patient_info.get("pills", [])

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws the face verification status banner on the top of the frame.
        Green = verified patient, Red = unrecognised person.
        """
        h, w = frame.shape[:2]
        conf  = self.confidence
        verif = self.is_verified
        stran = self.is_stranger

        if not self.face_images:
            cv2.putText(frame, "NO PATIENT ENROLLED",
                        (w // 2 - 160, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 80, 255), 2)
            return frame

        # Coloured banner
        if stran:
            color = (0, 0, 220)    # Red — stranger
            label = f"⚠ STRANGER DETECTED! ({conf:.0f}%)"
        elif verif:
            color = (0, 180, 60)   # Green — verified
            label = f"✓ {self.patient_name}  ({conf:.0f}% match)"
        else:
            color = (0, 130, 220)  # Orange — checking
            label = f"Verifying… ({conf:.0f}%)"

        cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
        cv2.putText(frame, label, (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        return frame
