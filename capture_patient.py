# ==============================================================================
#  PATIENT ENROLLMENT — capture_patient.py
#
#  Run this ONCE before starting app.py to register the patient.
#  It opens the camera, captures 10 face shots, and stores them in:
#       assets/patient/<patient_name>/face_NNN.jpg
#
#  Usage (standalone):
#       python capture_patient.py
#
#  Called automatically by run.py — you don't need to run this manually.
# ==============================================================================

import cv2
import os
import time


NUM_SHOTS   = 10       # Number of face images to capture
DELAY_SEC   = 1.5      # Pause between shots (seconds)
ASSETS_DIR  = "assets"


def collect_patient_info() -> dict:
    """Ask the user for patient details interactively in the terminal."""
    print("\n" + "="*60)
    print("  PATIENT ENROLLMENT — Enter Patient Details")
    print("="*60)

    name    = input("\n  Patient name         : ").strip() or "Patient"
    age_str = input("  Age                  : ").strip()
    age     = int(age_str) if age_str.isdigit() else 0
    dosage  = input("  Current dosage       : ").strip() or "As prescribed"

    print("\n  Enter pill names one by one (press ENTER on blank line to finish):")
    pills, colors = [], []
    idx = 1
    while True:
        p = input(f"    Pill {idx} name  : ").strip()
        if not p:
            break
        c = input(f"    Pill {idx} color : ").strip() or "white"
        pills.append(p)
        colors.append(c)
        idx += 1

    if not pills:
        pills  = ["Unknown"]
        colors = ["white"]

    return {
        "name":           name,
        "age":            age,
        "current_dosage": dosage,
        "pills":          pills,
        "pill_colors":    colors,
    }


def save_patient_info(info: dict, patient_dir: str):
    """Write patient metadata to a text file next to the face images."""
    info_path = os.path.join(patient_dir, "patient_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=== PATIENT INFORMATION ===\n")
        for key, val in info.items():
            if isinstance(val, list):
                f.write(f"{key}:\n")
                for item in val:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: {val}\n")
    print(f"\n[Enroll] Patient info saved → {info_path}")


def capture_faces(info: dict) -> str:
    """
    Open the camera, let the patient look at it, capture NUM_SHOTS face crops.
    Returns the patient directory path on success, or '' on failure.
    """
    patient_dir = os.path.join(
        ASSETS_DIR, "patient",
        info["name"].replace(" ", "_")
    )
    os.makedirs(patient_dir, exist_ok=True)
    save_patient_info(info, patient_dir)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print(f"\n=== PATIENT ENROLLMENT FOR: {info['name']} ===")
    print(f"Will capture {NUM_SHOTS} face images.")
    print("Please look directly at the camera and keep your face well-lit.")
    print("Press SPACE to start capturing. Press Q to abort.\n")

    waiting  = True
    count    = 0
    last_cap = 0.0
    success  = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
        )
        display = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 80), 2)

        if waiting:
            cv2.putText(display, "Press SPACE to start enrollment",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        else:
            cv2.putText(display,
                        f"Capturing {count}/{NUM_SHOTS}  — stay still…",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 80), 2)

            now = time.time()
            if len(faces) > 0 and (now - last_cap) >= DELAY_SEC:
                (x, y, w, h) = faces[0]
                face_crop = frame[y:y + h, x:x + w]
                fname     = os.path.join(patient_dir, f"face_{count:03d}.jpg")
                cv2.imwrite(fname, face_crop)
                print(f"  ✅ Saved face {count + 1}/{NUM_SHOTS} → {fname}")
                count    += 1
                last_cap  = now

                if count >= NUM_SHOTS:
                    print(f"\n[Enroll] ✅ Enrollment complete! {NUM_SHOTS} images saved to:")
                    print(f"         {patient_dir}")
                    print(f"\n[Enroll] Patient : {info['name']}")
                    print(f"[Enroll] Age     : {info['age']}")
                    print(f"[Enroll] Pills   : {', '.join(info['pills'])}")
                    success = True
                    break

        cv2.imshow("Patient Enrollment — Face Capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            waiting = False
        if key == ord("q"):
            print("[Enroll] Aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return patient_dir if success else ""


if __name__ == "__main__":
    info = collect_patient_info()
    capture_faces(info)
