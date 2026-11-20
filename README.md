# Real-Time Road & Speed Limit Alert System

A modular Python desktop project (`road_awareness_system`) separating the user interface (`gui.py`) from core vision, detection, road/speed lookup, and alert logic (`backend.py`), designed for real-time road awareness, speed limit alerts, and driver drowsiness/distraction detection.

## Architecture & Files

- `gui.py` — Tkinter desktop interface for controls, live display, and system status.
- `backend.py` — Core processing pipeline using OpenCV (`cv2`), YOLOv8 (`ultralytics`), `pyttsx3` voice alerts, and OpenStreetMap APIs for road/speed limit context.
- `.gitignore` — Excludes virtual environments, Python cache files, local IDE/macOS files, and model weights (`*.pt`).

## Setup on macOS

1. Create and activate a virtual environment inside the project folder:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate