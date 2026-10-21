# ==============================================================================
#  MODULE 3 — BIOMETRIC + MOTION ANALYTICS
#  File: analytics_module.py
# ==============================================================================
#  Computes per-frame biometric analytics:
#    1. Hand-to-lip Euclidean distance
#    2. Face landmark stability score
#    3. Hand motion velocity
#    4. Object proximity to mouth
#    5. Mouth opening percentage
#  Also generates graph-ready arrays and graph PNG files (Module 4).
# ==============================================================================

import math
import time
import os
import numpy as np

# ── GRAPH GENERATION (MODULE 4) ─────────────────────────────────────────────
# Matplotlib is imported lazily to avoid slowing down startup
_GRAPHS_DIR = "graphs"
os.makedirs(_GRAPHS_DIR, exist_ok=True)


# ── HELPER MATH ─────────────────────────────────────────────────────────────

def _euclidean(x1, y1, x2, y2) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _landmark_to_arr(lm) -> np.ndarray:
    return np.array([lm.x, lm.y])


# ── ANALYTICS ACCUMULATOR ────────────────────────────────────────────────────

class BiometricAnalytics:
    """
    Accumulates per-frame biometric measurements for a single session
    and provides graph generation at the end.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all recorded data — call at the start of each medication event."""
        self.timestamps:          list[float] = []   # wall-clock seconds
        self.hand_lip_distances:  list[float] = []   # mm
        self.face_stability:      list[float] = []   # 0-100 score
        self.hand_velocities:     list[float] = []   # normalised units / sec
        self.object_proximities:  list[float] = []   # mm (999 = no object)
        self.mouth_openings:      list[float] = []   # mm

        self._prev_hand_pos:  np.ndarray | None = None
        self._prev_face_pts:  np.ndarray | None = None
        self._prev_ts:        float = 0.0
        self._start_ts:       float = time.time()

    # ── PER-FRAME UPDATE ─────────────────────────────────────────────────────

    def update(
        self,
        face_landmarks,
        left_hand,
        right_hand,
        mm_per_unit: float,
        object_center_xy: tuple[float, float] | None = None,
    ):
        """
        Call once per frame to record all biometric metrics.

        Args:
            face_landmarks:    MediaPipe face landmark list (raw, not nested).
            left_hand:         MediaPipe left hand landmark list or None.
            right_hand:        MediaPipe right hand landmark list or None.
            mm_per_unit:       Conversion factor from distance_calculator.
            object_center_xy:  Normalised (x, y) of any detected object, or None.
        """
        now = time.time()
        self.timestamps.append(now - self._start_ts)

        # MediaPipe Tasks API returns lists of lists. Extract the first one.
        if isinstance(face_landmarks, list) and len(face_landmarks) > 0 and isinstance(face_landmarks[0], list):
            face_landmarks = face_landmarks[0]

        # Guard: skip frame if MediaPipe returned incomplete face data
        if not face_landmarks or len(face_landmarks) < 15:
            return

        # ── 1. Mouth opening ────────────────────────────────────────────────
        upper = face_landmarks[13]
        lower = face_landmarks[14]
        mouth_mm = _euclidean(upper.x, upper.y, lower.x, lower.y) * mm_per_unit
        self.mouth_openings.append(mouth_mm)

        # Mouth center for distance calculations
        mx = (upper.x + lower.x) / 2
        my = (upper.y + lower.y) / 2

        # ── 2. Hand-to-lip distance ─────────────────────────────────────────
        min_dist_mm = 999.0
        raw_hand = right_hand if right_hand else left_hand
        best_hand = None
        if raw_hand and len(raw_hand) > 0:
            best_hand = raw_hand[0] if isinstance(raw_hand[0], list) else raw_hand
            
        if best_hand and len(best_hand) > 8:
            idx = best_hand[8]
            min_dist_mm = _euclidean(mx, my, idx.x, idx.y) * mm_per_unit
        self.hand_lip_distances.append(round(min_dist_mm, 2))

        # ── 3. Face stability score ─────────────────────────────────────────
        # Track 6 stable face landmarks; lower jitter = higher stability
        KEY_LMK = [1, 152, 234, 454, 33, 263]   # Nose, chin, L/R temple, L/R eye corner
        pts = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in KEY_LMK])
        if self._prev_face_pts is not None:
            jitter = float(np.mean(np.linalg.norm(pts - self._prev_face_pts, axis=1)))
            stability = max(0.0, 100.0 - jitter * 5000)
        else:
            stability = 100.0
        self._prev_face_pts = pts
        self.face_stability.append(round(stability, 2))

        # ── 4. Hand motion velocity ─────────────────────────────────────────
        velocity = 0.0
        if best_hand and len(best_hand) > 0:
            wrist = best_hand[0]
            hand_pos = np.array([wrist.x, wrist.y])
            if self._prev_hand_pos is not None and self._prev_ts > 0:
                dt = now - self._prev_ts
                velocity = float(np.linalg.norm(hand_pos - self._prev_hand_pos) / max(dt, 0.001))
            self._prev_hand_pos = hand_pos
        self.hand_velocities.append(round(velocity, 4))
        self._prev_ts = now

        # ── 5. Object proximity ─────────────────────────────────────────────
        if object_center_xy is not None:
            ox, oy = object_center_xy
            obj_mm = _euclidean(mx, my, ox, oy) * mm_per_unit
        else:
            obj_mm = 999.0
        self.object_proximities.append(round(obj_mm, 2))

    # ── SUMMARY ──────────────────────────────────────────────────────────────

    def compute_confidence(self) -> float:
        """
        Returns a 0–100 medication completion confidence score based on:
          - Hand proximity to mouth (lower = better)
          - Mouth was open at some point
          - Hand-to-mouth distance trajectory (approaching then receding)
        """
        if not self.hand_lip_distances:
            return 0.0

        dists   = np.array(self.hand_lip_distances)
        mouths  = np.array(self.mouth_openings)

        # Was mouth open during the event?
        mouth_opened   = float(np.max(mouths) > 5.0)

        # Did hand get within 60 mm of mouth?
        close_approach = float(np.min(dists) < 60.0) if np.min(dists) < 999 else 0.0

        # Proximity score: how much time was hand close?
        close_frames   = np.sum(dists < 80.0)
        proximity_pct  = float(close_frames / max(len(dists), 1))

        score = (mouth_opened * 40) + (close_approach * 30) + (proximity_pct * 30)
        return round(min(score, 100.0), 2)

    def get_summary(self) -> dict:
        """Returns averaged metrics for logging / Firestore."""
        def safe_avg(arr):
            return round(float(np.mean(arr)), 3) if arr else 0.0

        return {
            "avg_hand_lip_dist_mm":  safe_avg(self.hand_lip_distances),
            "avg_mouth_opening_mm":  safe_avg(self.mouth_openings),
            "avg_face_stability":    safe_avg(self.face_stability),
            "avg_hand_velocity":     safe_avg(self.hand_velocities),
            "avg_object_proximity":  safe_avg(self.object_proximities),
            "completion_confidence": self.compute_confidence(),
            "frames_recorded":       len(self.timestamps),
        }

    # ── MODULE 4 — GRAPH GENERATION ──────────────────────────────────────────

    def generate_graphs(self, prefix: str = "session") -> list[str]:
        """
        Generate and save PNG graphs. Returns list of saved file paths.
        Graphs: Distance vs Time, Motion vs Time, Stability vs Time, Confidence.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # Headless — no GUI window
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Analytics] WARNING: matplotlib not installed. Graphs skipped.")
            return []

        if not self.timestamps:
            return []

        saved_files = []
        t = self.timestamps

        # -- Graph 1: Hand-to-Lip Distance vs Time --
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, self.hand_lip_distances, color="#00C8FF", linewidth=2)
        ax.axhline(y=60, color="#FF4444", linestyle="--", label="Threshold (60 mm)")
        ax.set_title("Hand → Lip Distance vs Time", fontsize=14)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (mm)")
        ax.legend()
        ax.set_facecolor("#0D1117")
        fig.patch.set_facecolor("#0D1117")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        path1 = os.path.join(_GRAPHS_DIR, f"{prefix}_distance.png")
        fig.savefig(path1, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path1)

        # -- Graph 2: Hand Motion Velocity vs Time --
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, self.hand_velocities, color="#FFD700", linewidth=2)
        ax.set_title("Hand Motion Velocity vs Time", fontsize=14)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (units/sec)")
        ax.set_facecolor("#0D1117")
        fig.patch.set_facecolor("#0D1117")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        path2 = os.path.join(_GRAPHS_DIR, f"{prefix}_velocity.png")
        fig.savefig(path2, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path2)

        # -- Graph 3: Face Stability vs Time --
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, self.face_stability, color="#7CFC00", linewidth=2)
        ax.set_ylim(0, 105)
        ax.set_title("Face Landmark Stability vs Time", fontsize=14)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Stability Score (0-100)")
        ax.set_facecolor("#0D1117")
        fig.patch.set_facecolor("#0D1117")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        path3 = os.path.join(_GRAPHS_DIR, f"{prefix}_stability.png")
        fig.savefig(path3, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path3)

        # -- Graph 4: Medication Confidence Gauge --
        score = self.compute_confidence()
        fig, ax = plt.subplots(figsize=(5, 4))
        color = "#00FF99" if score >= 70 else "#FF8800" if score >= 40 else "#FF3333"
        ax.barh(["Confidence"], [score], color=color, height=0.4)
        ax.set_xlim(0, 100)
        ax.set_title("Medication Completion Confidence", fontsize=13)
        ax.text(score + 1, 0, f"{score:.1f}%", va="center", color="white", fontsize=14)
        ax.set_facecolor("#0D1117")
        fig.patch.set_facecolor("#0D1117")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        path4 = os.path.join(_GRAPHS_DIR, f"{prefix}_confidence.png")
        fig.savefig(path4, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path4)

        print(f"[Analytics] {len(saved_files)} graphs saved → {_GRAPHS_DIR}/")
        return saved_files
