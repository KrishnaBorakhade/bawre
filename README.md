# bawre

Alzheimer Patient Monitoring & Assistance System.

## Overview
An AI-powered patient monitoring system featuring:
- MediaPipe Holistic pose and landmark tracking
- Face detection and patient identification
- Pill adherence and interaction tracking
- Twilio automated SMS emergency alerting
- Firebase Firestore & Storage integration for real-time cloud sync
- Interactive React dashboard for caretakers

## Setup & Installation

### Python Backend
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # or install required packages
   ```
3. Copy `.env.example` to `.env` and fill in your Twilio and Firebase configuration:
   ```bash
   cp .env.example .env
   ```
4. Place your `firebase_service_account.json` in the root directory.

### React Dashboard
```bash
cd react-dashboard
npm install
npm run dev
```
