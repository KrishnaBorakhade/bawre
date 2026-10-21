# ==============================================================================
#  PATIENT VERIFICATION  —  patient_verification.py
#
#  Stage 1 ── Patient Identity Verification (ACTIVE)
#             Load reference photos: assets/patient/<name>/face_NNN.jpg or N.png
#             Compare live camera face against stored LBPH embeddings.
#             Block session until the correct patient is confirmed.
#
#  Stage 2 ── Pill-in-Hand Verification  (FUTURE — commented out below)
#
#  Stage 3 ── Medication Intake Monitoring  (lives in app.py)
# ==============================================================================

import cv2
import os
import numpy as np


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

ASSETS_DIR          = os.path.join("assets", "patient")
FACE_CASCADE_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
VERIFY_HOLD_SECONDS = 2.0    # patient must be recognised for this many acc. seconds
LBPH_CONFIDENCE_MAX = 85     # LBPH: lower = better match; < 85 is accepted


# ==============================================================================
#  STAGE 1 — PATIENT IDENTITY VERIFIER  (ACTIVE)
# ==============================================================================

class PatientVerifier:
    """
    Loads all face images from assets/patient/<patient_folder>/
    and trains an LBPH face recogniser on them.

    Usage in run():
        verifier = PatientVerifier(patient_folder_path)
        if verifier.load_references():
            while not verifier.is_verified:
                verifier.update(frame, curr_time)
    """

    def __init__(self, patient_dir: str):
        self.patient_dir  = patient_dir
        self.recogniser   = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.is_verified  = False
        self._accum       = 0.0
        self._last_ts     = None
        self._trained     = False

    def load_references(self) -> bool:
        """
        Scans patient_dir for any .jpg/.png image files and trains the recogniser.
        Returns True if at least one usable face photo was found.
        """
        faces, labels = [], []

        if not os.path.isdir(self.patient_dir):
            print(f"[FaceVerify] ⚠️  Folder not found: {self.patient_dir}")
            return False

        for fname in sorted(os.listdir(self.patient_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(self.patient_dir, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            detected = self.face_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(detected) == 0:
                # No face box detected → use the whole image (already cropped)
                faces.append(cv2.resize(img, (200, 200)))
            else:
                x, y, fw, fh = detected[0]
                faces.append(cv2.resize(img[y:y+fh, x:x+fw], (200, 200)))
            labels.append(0)   # all reference photos = label 0 (the patient)

        if not faces:
            print("[FaceVerify] ❌ No usable face photos found.")
            return False

        self.recogniser.train(faces, np.array(labels))
        self._trained = True
        print(f"[FaceVerify] ✅ Trained on {len(faces)} photo(s) from: {self.patient_dir}")
        return True

    def update(self, frame, curr_time: float) -> bool:
        """
        Call once per frame during the verification phase.
        Draws overlay on frame in-place.
        Returns True once the patient has been recognised for VERIFY_HOLD_SECONDS.
        """
        if self.is_verified:
            return True
        if not self._trained:
            # No training data — pass through immediately
            self.is_verified = True
            return True

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        recognised = False
        for (x, y, fw, fh) in faces:
            face_roi = cv2.resize(gray[y:y+fh, x:x+fw], (200, 200))
            label, confidence = self.recogniser.predict(face_roi)
            ok    = (label == 0 and confidence < LBPH_CONFIDENCE_MAX)
            color = (0, 255, 80) if ok else (0, 60, 255)
            tag   = f"Patient OK ({confidence:.0f})" if ok else f"Unknown ({confidence:.0f})"
            cv2.putText(frame, tag, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            if ok:
                recognised = True

        if recognised:
            if self._last_ts is not None:
                self._accum += curr_time - self._last_ts
            self._last_ts = curr_time
        else:
            self._last_ts = None   # gap resets the run, not the total

        if self._accum >= VERIFY_HOLD_SECONDS:
            self.is_verified = True
            print("[FaceVerify] ✅ Patient identity confirmed.")

        # Progress bar at bottom of frame
        h_f, w_f = frame.shape[:2]
        pct   = min(1.0, self._accum / VERIFY_HOLD_SECONDS)
        bar_w = int(pct * 350)
        cv2.rectangle(frame, (10, h_f - 50), (360, h_f - 36), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, h_f - 50), (10 + bar_w, h_f - 36),
                      (0, 255, 80) if pct >= 1.0 else (0, 200, 255), -1)
        cv2.putText(frame,
                    f"Face verify: {self._accum:.1f}s / {VERIFY_HOLD_SECONDS:.0f}s  —  look at camera",
                    (10, h_f - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "PATIENT VERIFICATION", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        return self.is_verified


# ==============================================================================
#  STAGE 2 — PILL-IN-HAND VERIFIER  (FUTURE — all commented out)
# ==============================================================================

# class PillVerifier:
#     def __init__(self, pill_model, pill_labels):
#         self.model        = pill_model
#         self.labels       = pill_labels
#         self.is_confirmed = False
#         self._accum       = 0.0
#         self._last_ts     = None
#
#     def update(self, frame, hand_landmarks, curr_time, w, h):
#         ... (see git history for full implementation)
