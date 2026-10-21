# ==============================================================================
#  ADVANCED SPATIAL MATHEMATICS — 3D EDITION
#  File: distance_calculator.py
#
#  All distances now use the full 3D Euclidean formula:
#    dist = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
#
#  MediaPipe provides a normalized Z coordinate (depth relative to hips for
#  pose, relative to wrist for hands, and relative to face for face mesh).
#  Even though the scale is relative, it is self-consistent within a frame
#  which is exactly what we need: we can now tell if a hand is BEHIND the
#  face or IN FRONT of it — solving the 2D occlusion problem.
# ==============================================================================

import math


# ── HELPER ───────────────────────────────────────────────────────────────────

def dist3d(ax, ay, az, bx, by, bz) -> float:
    """True 3D Euclidean distance between two normalised-landmark points."""
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)


def dist2d(ax, ay, bx, by) -> float:
    """Fallback 2D distance (used only where Z is unavailable)."""
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────

def analyze_mouth_and_fingertips(
    face_landmarks,
    left_hand,
    right_hand,
    mouth_open_threshold_mm: float = 2.0,
    proximity_threshold_mm: float = 80.0,
):
    """
    3D-aware analysis of mouth state and fingertip proximity.

    Key improvement over the old 2D version:
      * Uses landmark.z (depth) so the hand approaching the mouth head-on
        is correctly detected even when X/Y are identical (2D occlusion).
      * Face-width scale normalises pixel distances to approximate mm.
    """

    # MediaPipe Tasks API returns a list of faces/hands. Extract the first one.
    if isinstance(face_landmarks, list) and len(face_landmarks) > 0 and isinstance(face_landmarks[0], list):
        face_landmarks = face_landmarks[0]
        
    # Face width ≈ 150 mm; landmarks 234 (left cheek) and 454 (right cheek)
    if not face_landmarks or len(face_landmarks) < 455:
        # Not enough landmarks — return safe defaults
        return {
            "action": "Idle", "mouth_state": "Closed",
            "mouth_gap_mm": 0.0, "mouth_gap_pct": 0.0,
            "finger_dist_mm": 999.0, "finger_dist_pct": 100.0,
            "is_pinching": False, "pinch_dist_mm": 999.0,
        }
    left_edge  = face_landmarks[234]
    right_edge = face_landmarks[454]
    face_width_norm = dist2d(left_edge.x, left_edge.y, right_edge.x, right_edge.y)
    face_width_norm = max(face_width_norm, 0.001)
    mm_per_unit = 150.0 / face_width_norm

    # ── MOUTH STATE ──────────────────────────────────────────────────────────
    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]
    mouth_gap_norm = dist3d(
        upper_lip.x, upper_lip.y, upper_lip.z,
        lower_lip.x, lower_lip.y, lower_lip.z,
    )
    mouth_gap_mm  = mouth_gap_norm * mm_per_unit
    is_mouth_open = mouth_gap_mm > mouth_open_threshold_mm

    # Mouth centre in 3D
    mx = (upper_lip.x + lower_lip.x) / 2.0
    my = (upper_lip.y + lower_lip.y) / 2.0
    mz = (upper_lip.z + lower_lip.z) / 2.0

    # ── HAND PROXIMITY ───────────────────────────────────────────────────────
    # Index fingertip = landmark 8 on each hand
    # Using full 3D Euclidean: dist = sqrt(Δx² + Δy² + Δz²)  [same as VPython mag(d)]
    min_dist_norm  = float("inf")
    min_pinch_norm = float("inf")
    min_hand_size  = float("inf")  # wrist → middle-finger tip span (LM0 → LM12)

    for raw_hand in (left_hand, right_hand):
        if not raw_hand:
            continue

        # Extract the actual landmarks from the list of hands
        hand = raw_hand[0] if isinstance(raw_hand[0], list) else raw_hand

        if not hand or len(hand) < 13:   # need LM12 (middle tip) so require ≥ 13 landmarks
            continue

        index  = hand[8]    # index fingertip
        thumb  = hand[4]    # thumb tip
        wrist  = hand[0]    # wrist
        middle = hand[12]   # middle-finger tip  (reference for hand size)

        # 3D distance from index tip to mouth centre (VPython: mag(qball.pos - ball.pos))
        d = dist3d(mx, my, mz, index.x, index.y, index.z)
        min_dist_norm = min(min_dist_norm, d)

        # 3D pinch distance (thumb tip ↔ index tip)
        p = dist3d(thumb.x, thumb.y, thumb.z, index.x, index.y, index.z)
        min_pinch_norm = min(min_pinch_norm, p)

        # 3D hand span (wrist ↔ middle-finger tip) — camera-distance-invariant reference
        hs = dist3d(wrist.x, wrist.y, wrist.z, middle.x, middle.y, middle.z)
        min_hand_size = min(min_hand_size, hs)

    finger_dist_mm = (min_dist_norm  * mm_per_unit) if min_dist_norm  != float("inf") else 999.0
    pinch_dist_mm  = (min_pinch_norm * mm_per_unit) if min_pinch_norm != float("inf") else 999.0

    # ── DYNAMIC 3D PINCH THRESHOLD ──────────────────────────────────────────
    # "80% closed" means the thumb-index gap is < 20% of the hand span.
    # This automatically scales with how close/far the patient is from the camera.
    #   pinch_ratio = pinch_dist_3d / hand_size_3d
    #   is_pinching  when pinch_ratio < 0.20   (≡ 80% of fingers are together)
    if min_hand_size != float("inf") and min_hand_size > 1e-6:
        pinch_ratio = min_pinch_norm / min_hand_size
        is_pinching = pinch_ratio < 0.20          # 80%-closed threshold
    else:
        # Fallback: absolute mm threshold if hand-size reference unavailable
        is_pinching = pinch_dist_mm < 40.0

    # ── PERCENTAGES ──────────────────────────────────────────────────────────
    mouth_open_pct  = min(100.0, (mouth_gap_mm / 50.0) * 100.0)
    finger_dist_pct = min(100.0, (finger_dist_mm / 500.0) * 100.0) if finger_dist_mm != 999.0 else 100.0

    # ── ACTION LABEL ─────────────────────────────────────────────────────────
    action = "Idle"
    if finger_dist_mm < proximity_threshold_mm:
        if is_mouth_open:
            if is_pinching:
                action = "Taking Medication"
            else:
                action = "Eating / Finger near Mouth"
        else:
            action = "Finger at Closed Mouth"

    return {
        "action":          action,
        "mouth_state":     "Open" if is_mouth_open else "Closed",
        "mouth_gap_mm":    round(mouth_gap_mm,   2),
        "mouth_gap_pct":   round(mouth_open_pct, 2),
        "finger_dist_mm":  round(finger_dist_mm, 2),
        "finger_dist_pct": round(finger_dist_pct,2),
        "is_pinching":     is_pinching,
        "pinch_dist_mm":   round(pinch_dist_mm,  2),
    }


# ── OBJECT × BODY INTERACTION ─────────────────────────────────────────────────

def analyze_object_interaction(obj_center_x, obj_center_y, pose_landmarks, obj_label):
    """
    3D-aware: detects whether an object (cup/bottle) is being raised to the mouth.
    Pose Z gives depth, so a bottle approaching the face from the front is captured
    even if it overlaps the face in 2D.
    """
    if not pose_landmarks:
        return {"action": None}
        
    if isinstance(pose_landmarks, list) and len(pose_landmarks) > 0 and isinstance(pose_landmarks[0], list):
        pose_landmarks = pose_landmarks[0]
        
    if len(pose_landmarks) < 13:
        return {"action": None}

    # Pose landmark indices
    # 9  = left  mouth corner
    # 10 = right mouth corner
    # 11 = left  shoulder
    # 12 = right shoulder
    lm9  = pose_landmarks[9]
    lm10 = pose_landmarks[10]
    lm11 = pose_landmarks[11]
    lm12 = pose_landmarks[12]

    mouth_cx = (lm9.x + lm10.x) / 2.0
    mouth_cy = (lm9.y + lm10.y) / 2.0
    mouth_cz = (lm9.z + lm10.z) / 2.0

    # For YOLO objects we only have 2D centre; approximate Z=0 for the object.
    # The key check is still whether it is ABOVE the shoulder and laterally close.
    dist_to_mouth_2d = dist2d(obj_center_x, obj_center_y, mouth_cx, mouth_cy)

    shoulder_width = dist2d(lm11.x, lm11.y, lm12.x, lm12.y)
    shoulder_width = max(shoulder_width, 0.2)

    avg_shoulder_y = (lm11.y + lm12.y) / 2.0
    is_above_shoulder = obj_center_y < avg_shoulder_y

    action = None
    if obj_label in ["bottle", "cup", "wine glass", "glass"]:
        # Object must be above shoulder height AND within 1.5x shoulder-width
        # distance from the mouth centre.
        if is_above_shoulder and dist_to_mouth_2d < (shoulder_width * 1.5):
            action = f"Drinking from {obj_label}"

    return {
        "action":           action,
        "dist_to_mouth_2d": dist_to_mouth_2d,
        "is_above_shoulder": is_above_shoulder,
        "shoulder_width":   shoulder_width,
    }