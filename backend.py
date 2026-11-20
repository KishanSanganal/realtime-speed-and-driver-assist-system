import cv2
import pyttsx3
import threading
import time
import requests
from ultralytics import YOLO

class RoadSafetyBackend:
    def __init__(self, video_source=0):
        # 1. Telemetry & OpenStreetMap State
        self.current_speed = 65  # km/h
        self.speed_limit = 50    # Default speed limit (km/h)
        self.status_message = "Normal Driving"
        self.driver_status = "Driver Alert"
        self.road_name = "Detecting Road..."

        # 2. AI Model, Camera & Voice Init
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(video_source)
        self.tts = pyttsx3.init()

        # 3. Tracking variables
        self.looking_down_frames = 0
        self.alert_threshold = 12
        self.last_speech_time = 0
        self.baseline_y1 = None

        # Start background thread to query OpenStreetMap speed limits
        threading.Thread(target=self.fetch_osm_speed_limit_loop, daemon=True).start()

    def speak(self, message):
        """Speech engine with a cooldown so it never spams audio."""
        current_time = time.time()
        if current_time - self.last_speech_time < 5:
            return
        
        self.last_speech_time = current_time
        def _say():
            try:
                self.tts.say(message)
                self.tts.runAndWait()
            except: pass
        threading.Thread(target=_say, daemon=True).start()

    def fetch_osm_speed_limit_loop(self):
        """Periodically queries OpenStreetMap Overpass API for road speed limits based on GPS coordinates."""
        while True:
            try:
                # Example Coordinates (Berlin center or your current mock GPS location)
                # In a full mobile deployment, you would feed live device GPS latitude & longitude here.
                lat, lon = 52.5200, 13.4050 

                # Overpass API query to find nearby highway tags and maxspeed attributes
                overpass_url = "http://overpass-api.de/api/interpreter"
                query = f"""
                [out:json][timeout:5];
                way(around:30,{lat},{lon})["highway"];
                out tags;
                """
                
                response = requests.get(overpass_url, params={'data': query}, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    elements = data.get("elements", [])
                    if elements:
                        tags = elements[0].get("tags", {})
                        self.road_name = tags.get("name", "Local Street")
                        
                        # Extract OSM maxspeed tag if available (e.g., "50", "30 mph", "none")
                        maxspeed_tag = tags.get("maxspeed")
                        if maxspeed_tag:
                            # Clean string to get numerical value
                            numeric_speed = "".join(filter(str.isdigit, maxspeed_tag))
                            if numeric_speed:
                                self.speed_limit = int(numeric_speed)
                else:
                    self.road_name = "OSM Offline Mode"
            except Exception as e:
                self.road_name = "GPS/OSM Connected (Simulated)"
            
            # Check every 15 seconds to avoid spamming the public Overpass server
            time.sleep(15)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.resize(frame, (540, 380))

        # --- A. OBJECT DETECTION (YOLO) ---
        results = self.model(frame, stream=True, verbose=False)
        person_detected = False
        is_looking_down = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[cls]
                
                if cls == 0:  # Person / Driver class
                    person_detected = True
                    
                    if self.baseline_y1 is None:
                        self.baseline_y1 = y1

                    # Trigger only when head drops down
                    if y1 > (self.baseline_y1 + 30):
                        is_looking_down = True
                    else:
                        self.baseline_y1 = int(0.95 * self.baseline_y1 + 0.05 * y1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- B. DRIVER ATTENTION / LOOKING DOWN CHECK ---
        if person_detected:
            if is_looking_down:
                self.looking_down_frames += 1
                if self.looking_down_frames >= self.alert_threshold:
                    self.driver_status = "LOOKING DOWN / DROWSY!"
                    self.speak("Warning! Keep your eyes on the road!")
                    cv2.putText(frame, "HEAD DOWN / DISTRACTED!", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.looking_down_frames = max(0, self.looking_down_frames - 1)
                self.driver_status = "Driver Alert (Eyes Up)"
        else:
            self.driver_status = "Driver Not Found"

        # --- C. DYNAMIC SPEED LIMIT & TELEMETRY CHECK ---
        if self.current_speed > self.speed_limit:
            self.status_message = f"OVER SPEED ({self.speed_limit} km/h limit)!"
            self.speak(f"Warning. Speed limit is {self.speed_limit} kilometers per hour.")
        else:
            self.status_message = f"Safe Speed (Limit: {self.speed_limit} km/h)"

        return frame

    def release_resources(self):
        if self.cap.isOpened():
            self.cap.release()