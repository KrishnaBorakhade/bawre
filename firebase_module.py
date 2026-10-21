# ==============================================================================
#  MODULE 2 — FIREBASE DATA STORAGE (REALTIME DATABASE)
#  File: firebase_module.py
#
#  Uploads captured frames and session events to Firebase Realtime Database
#  as base64 encoded strings using a background worker queue.
# ==============================================================================

import os
import cv2
import base64
import threading
import queue
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_JSON = os.getenv("SERVICE_ACCOUNT_JSON_PATH", "firebase_service_account.json")
DATABASE_URL         = "https://smrutipankha-4f1fc-default-rtdb.asia-southeast1.firebasedatabase.app"

CAPTURED_FRAMES_DIR = "captured_frames"
os.makedirs(CAPTURED_FRAMES_DIR, exist_ok=True)

_firebase_lock = threading.Lock()
_upload_queue = queue.Queue()

def _init_firebase():
    """Initialise Firebase App safely."""
    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials
        with _firebase_lock:
            if not firebase_admin._apps:
                cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': DATABASE_URL
                })
        return True
    except Exception as e:
        print(f"[Firebase] Init error: {e}")
        return False


def _frame_to_b64(frame) -> str:
    """Encode an OpenCV frame (numpy array) directly to base64 JPEG."""
    # Compress significantly (quality=60) since we are sending 12 FPS
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _img_to_b64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==============================================================================
#  BACKGROUND UPLOAD WORKER
# ==============================================================================

def _upload_worker():
    """Continuously processes items from the queue and uploads them."""
    import firebase_admin
    from firebase_admin import db
    
    firebase_ready = False

    while True:
        task = _upload_queue.get()
        if task is None:  # Shutdown signal
            break
            
        try:
            if not firebase_ready:
                firebase_ready = _init_firebase()
                if not firebase_ready:
                    print(f"⚠️ [Firebase] Upload failed — could not initialize Firebase (Check if {SERVICE_ACCOUNT_JSON} exists).", flush=True)
                    continue
            
            collection = task.get("collection")
            data = task.get("data")
            ref = db.reference(collection)
            ref.push(data)
            print(f"✅ [Firebase] Uploaded to {collection}", flush=True)
        except Exception as e:
            print(f"❌ [Firebase] Upload error: {e}", flush=True)
        finally:
            _upload_queue.task_done()

# Start the worker thread
_worker_thread = threading.Thread(target=_upload_worker, daemon=True)
_worker_thread.start()


# ==============================================================================
#  FRAME CAPTURE
# ==============================================================================

def save_captured_frame(frame, session_prefix: str, frame_idx: int) -> str:
    """
    Save a frame locally to captured_frames/ and add it to the upload queue.
    """
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame_{session_prefix}_{frame_idx:04d}_{ts}.jpg"
    local_path = os.path.join(CAPTURED_FRAMES_DIR, filename)
    cv2.imwrite(local_path, frame)

    # Convert to base64 and queue it up
    b64 = _frame_to_b64(frame)
    _upload_queue.put({
        "collection": "captured_frames",
        "data": {
            "session_id":  session_prefix,
            "frame_index": frame_idx,
            "timestamp":   datetime.now().isoformat(),
            "filename":    filename,
            "image_base64": b64,
        }
    })
    return local_path


# ==============================================================================
#  GRAPH UPLOAD
# ==============================================================================

def upload_graph_to_firebase(local_path: str, remote_name: str = None):
    try:
        b64  = _img_to_b64(local_path)
        name = remote_name or os.path.basename(local_path)
        _upload_queue.put({
            "collection": "graphs",
            "data": {
                "filename":    name,
                "timestamp":   datetime.now().isoformat(),
                "image_base64": b64,
            }
        })
    except Exception:
        pass


# ==============================================================================
#  SESSION EVENT RECORD
# ==============================================================================

def save_session_event(success: bool, pill_name: str = "", confidence: float = 0.0,
                       duration: float = 0.0, session_id: str = ""):
    _upload_queue.put({
        "collection": "medication_events",
        "data": {
            "session_id":      session_id,
            "timestamp":       datetime.now().isoformat(),
            "pill_detected":   pill_name,
            "confidence_pct":  round(confidence * 100, 1),
            "success":         success,
            "duration_seconds": round(duration, 1),
        }
    })


# ==============================================================================
#  VIDEO RECORDER (Stub for compatibility)
# ==============================================================================

class MedicationVideoRecorder:
    def __init__(self):
        self.is_recording    = False
        self.frame_confidences: list = []

    def start_recording(self, frame_shape, action: str, fps: float = 20.0):
        self.is_recording = True
        self.frame_confidences = []

    def write_frame(self, frame, confidence: float = 0.0):
        self.frame_confidences.append(confidence)

    def stop_and_upload(self, success: bool = True):
        self.is_recording = False
