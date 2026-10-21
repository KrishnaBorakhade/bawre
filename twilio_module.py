# ==============================================================================
#  MODULE 1 — TWILIO SMS ALERT SYSTEM
#  File: twilio_module.py
# ==============================================================================
#  Features:
#    - Predefined medication time slots (10 AM, 2 PM, 8 PM)
#    - SMS on medication taken / missed
#    - Daily summary SMS
#    - Load credentials from .env file
# ==============================================================================

import os
from datetime import datetime, time as dtime
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

# ── LOAD CREDENTIALS ────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE       = os.getenv("TWILIO_PHONE", "")
TARGET_PHONE       = os.getenv("TARGET_PHONE", "")

# ── MEDICATION SCHEDULE ──────────────────────────────────────────────────────
# Each slot is (hour, minute) in 24h format
MED_SLOTS = [
    dtime(10, 0),   # 10:00 AM
    dtime(14, 0),   # 2:00 PM
    dtime(20, 0),   # 8:00 PM
]

# Tolerance window: ±20 minutes around each slot
WINDOW_MINUTES = 20


def _get_twilio_client():
    """Returns a Twilio client. Returns None if credentials are missing."""
    if not TWILIO_ACCOUNT_SID or "PASTE" in TWILIO_ACCOUNT_SID:
        print("[Twilio] WARNING: Credentials not set in .env — SMS not sent.")
        return None
    try:
        from twilio.rest import Client
        return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except ImportError:
        print("[Twilio] ERROR: Run  pip install twilio  to enable SMS.")
        return None


def _format_time(t: dtime) -> str:
    """Formats a time object to 12-hour string."""
    # Fixed for Windows compatibility: use %I instead of %-I
    return datetime.combine(datetime.today(), t).strftime("%I:%M %p").lstrip("0")


def get_next_slot(now_time: dtime = None) -> dtime:
    """Returns the next upcoming medication slot."""
    if now_time is None:
        now_time = datetime.now().time()
    for slot in MED_SLOTS:
        if slot > now_time:
            return slot
    return MED_SLOTS[0]  # Wrap around to first slot of next day


def is_within_window(slot: dtime, now_time: dtime = None) -> bool:
    """Checks if the current time is within ±WINDOW_MINUTES of a given slot."""
    if now_time is None:
        now_time = datetime.now().time()
    now_dt  = datetime.combine(datetime.today(), now_time)
    slot_dt = datetime.combine(datetime.today(), slot)
    delta   = abs((now_dt - slot_dt).total_seconds() / 60)
    return delta <= WINDOW_MINUTES


def get_active_slot() -> dtime | None:
    """Returns the currently active medication slot (if within its window)."""
    now_time = datetime.now().time()
    for slot in MED_SLOTS:
        if is_within_window(slot, now_time):
            return slot
    return None


def send_sms(body: str) -> bool:
    """Sends an SMS to the caregiver's phone. Returns True on success."""
    client = _get_twilio_client()
    if not client:
        return False
    try:
        msg = client.messages.create(
            body=body,
            from_=TWILIO_PHONE,
            to=TARGET_PHONE,
        )
        print(f"[Twilio] SMS sent: {msg.sid}")
        return True
    except Exception as e:
        print(f"[Twilio] ERROR sending SMS: {e}")
        return False


def send_medication_taken_sms(pill_name: str = "Medication") -> bool:
    """Sends SMS confirming medication was taken."""
    now_str  = datetime.now().strftime("%I:%M %p")
    next_slot = get_next_slot()
    body = (
        f"✅ {pill_name} taken at {now_str}.\n"
        f"Next medication: {_format_time(next_slot)}."
    )
    return send_sms(body)


def send_missed_medication_sms(slot: dtime) -> bool:
    """Sends SMS alerting that medication was missed."""
    body = (
        f"⚠️ MISSED MEDICATION ALERT!\n"
        f"Medication was not taken at {_format_time(slot)}.\n"
        f"Please assist the patient immediately."
    )
    return send_sms(body)


def send_daily_summary_sms(taken_count: int, total_slots: int, pill_log: list) -> bool:
    """Sends a daily summary SMS."""
    log_str = "\n".join(f"  • {entry}" for entry in pill_log) or "  (none)"
    body = (
        f"📊 Daily Medication Summary\n"
        f"Taken: {taken_count}/{total_slots} doses\n"
        f"Log:\n{log_str}"
    )
    return send_sms(body)


def send_no_activity_alert(minutes: int) -> bool:
    """Emergency alert when no activity is detected for too long."""
    body = (
        f"🚨 EMERGENCY ALERT!\n"
        f"No activity detected for {minutes} minutes.\n"
        f"Please check on the patient immediately."
    )
    return send_sms(body)
