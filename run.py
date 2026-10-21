# ==============================================================================
#  ALZHEIMER MEDICATION MONITORING SYSTEM — LAUNCHER
#  run.py
#
#  HOW TO RUN:
#       holistic_env\Scripts\Activate
#       python run.py
#
#  One question is asked at startup:
#
#     S — Simulation mode
#         Full monitoring session runs (camera, face verify, pinch detection).
#         When the session ends FOR ANY REASON, a SUCCESS SMS is sent.
#
#     A — Actual monitoring mode
#         Full monitoring session runs identically.
#         When the session ends without a confirmed dose, a MISSED SMS is sent.
#
#  The SMS is sent AFTER app.py exits, based purely on the mode.
#  No detection result inside app.py can change which SMS is sent.
# ==============================================================================

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────

def banner():
    print("\n" + "=" * 65)
    print("    🧠  ALZHEIMER MEDICATION MONITORING SYSTEM  v2.0")
    print("=" * 65)
    print("    Powered by MediaPipe · YOLOv8 · MobileNetV2 · Twilio")
    print("=" * 65 + "\n")


def ask_mode() -> str:
    """Ask ONE question. Returns 's' or 'a'."""
    while True:
        choice = input("  Mode?  [S = Simulation / A = Actual] : ").strip().lower()
        if choice in ("s", "a"):
            return choice
        print("  ⚠️  Enter S or A.")


def find_patient_dir() -> str | None:
    """Return path to first patient sub-folder that contains at least one image."""
    base = os.path.join(ROOT, "assets", "patient")
    if not os.path.isdir(base):
        return None
    for entry in sorted(os.scandir(base), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        imgs = [f for f in os.listdir(entry.path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if imgs:
            return entry.path
    return None


def send_sms(mode: str):
    """Send the one caregiver SMS after the session. Mode S=success, A=missed."""
    try:
        from twilio_module import send_medication_taken_sms, send_missed_medication_sms
        from twilio_module import get_active_slot
        from datetime import datetime, time as dtime
        if mode == "s":
            send_medication_taken_sms("Medication (Simulation)")
        else:
            slot = get_active_slot()
            if slot is None:
                now  = datetime.now()
                slot = dtime(now.hour, now.minute)
            send_missed_medication_sms(slot)
    except Exception as e:
        pass   # Silent — do not expose SMS details to console



def launch_session(mode: str):
    """
    Common launcher for both modes:
      1. Verify patient photos exist.
      2. Write .sim_mode flag if simulation.
      3. Launch app.py (blocks until session ends).
      4. Send the mode-based SMS.
      5. Clean up flag files.
    """
    # ── Check patient folder ──────────────────────────────────────────────────
    patient_dir = find_patient_dir()
    if patient_dir:
        imgs = [f for f in os.listdir(patient_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"\n  ✅ Patient folder : {patient_dir}")
        print(f"  ✅ Photos found   : {len(imgs)}\n")
    else:
        print("\n  ⚠️  No patient photos found.")
        print("  Place photos in:  assets\\patient\\<PatientName>\\")
        print("  Accepted names :  1.png  2.png … or  face_000.jpg …")
        print("  Then re-run    :  python run.py\n")
        sys.exit(0)

    # ── Write .sim_mode flag (app.py reads this at startup) ───────────────────
    flag_path = os.path.join(ROOT, ".sim_mode")
    if mode == "s":
        with open(flag_path, "w") as f:
            f.write("1")
        print("  ⚡ SIMULATION MODE — caregiver will always receive SUCCESS SMS.\n")
    else:
        print("  🔴 ACTUAL MODE — caregiver will receive MISSED SMS if dose not confirmed.\n")

    # ── Write patient name for app.py face verification ───────────────────────
    with open(os.path.join(ROOT, ".current_patient.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.basename(patient_dir))

    # ── Show instructions ─────────────────────────────────────────────────────
    print("""  ╔═══════════════════════════════════════════════════════╗
  ║  MEDICATION INTAKE INSTRUCTIONS                       ║
  ║                                                       ║
  ║  1. Hold pill between THUMB and INDEX FINGER          ║
  ║     (pinch gesture — fingertips nearly touching)      ║
  ║                                                       ║
  ║  2. Bring hand close to your OPEN MOUTH               ║
  ║                                                       ║
  ║  3. HOLD that position for 5 CONTINUOUS SECONDS       ║
  ║     Watch the cyan bar at the top fill up →           ║
  ║                                                       ║
  ║  ⏱  Session window : 50 seconds total               ║
  ║  Keys: Q = quit  |  G = save graphs now              ║
  ╚═══════════════════════════════════════════════════════╝
""")
    input("  Press ENTER to open the monitoring window …")

    # ── Run app.py (blocks until window closes) ───────────────────────────────
    print("\n  Launching monitoring session …\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "app.py")], cwd=ROOT)

    # ── Send the ONE mode-based SMS ───────────────────────────────────────────
    send_sms(mode)

    # ── Clean up flag files ───────────────────────────────────────────────────
    for f in [flag_path, os.path.join(ROOT, ".current_patient.txt")]:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 65)
    print("  Session complete.  Check  graphs/  for saved analytics.")
    print("=" * 65 + "\n")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    banner()
    mode = ask_mode()
    launch_session(mode)


if __name__ == "__main__":
    main()
