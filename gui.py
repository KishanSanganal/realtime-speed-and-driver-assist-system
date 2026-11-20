import cv2
import tkinter as tk
from PIL import Image, ImageTk
from backend import RoadSafetyBackend

class RoadSafetyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Road & Driver Awareness System")
        self.root.geometry("950x620")
        self.root.configure(bg="#1e1e1e")

        self.backend = RoadSafetyBackend(video_source=0)
        self.setup_ui()

        self.update_video_loop()
        self.update_telemetry_loop()

    def setup_ui(self):
        tk.Label(
            self.root, 
            text="DRIVER AWARENESS SYSTEM", 
            font=("Arial", 14, "bold"), 
            bg="#1e1e1e", 
            fg="#00E676"
        ).pack(pady=10)

        container = tk.Frame(self.root, bg="#1e1e1e")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video Panel (Left)
        self.video_label = tk.Label(container, bg="black", width=540, height=380)
        self.video_label.pack(side=tk.LEFT, padx=10)

        # Telemetry Panel (Right)
        panel = tk.Frame(container, bg="#2d2d2d", width=330)
        panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(panel, text="SYSTEM MONITORING HUD", font=("Arial", 12, "bold"), bg="#2d2d2d", fg="white").pack(pady=15)

        self.lbl_driver = tk.Label(panel, text="Driver: Initializing...", font=("Arial", 11, "bold"), bg="#2d2d2d", fg="#00E676")
        self.lbl_driver.pack(anchor="w", padx=20, pady=8)

        self.lbl_speed = tk.Label(panel, text="Speed: 65 km/h", font=("Arial", 11), bg="#2d2d2d", fg="#00E676")
        self.lbl_speed.pack(anchor="w", padx=20, pady=8)

        self.lbl_limit = tk.Label(panel, text="Speed Limit: 50 km/h", font=("Arial", 11), bg="#2d2d2d", fg="#FF3D00")
        self.lbl_limit.pack(anchor="w", padx=20, pady=8)

        self.lbl_status = tk.Label(panel, text="Status: Normal", font=("Arial", 11, "bold"), bg="#2d2d2d", fg="#FFEA00")
        self.lbl_status.pack(anchor="w", padx=20, pady=20)

        info_text = "Architecture:\n• gui.py (Tkinter Frontend)\n• backend.py (OpenCV, MediaPipe & YOLOv8)\n• pyttsx3 (Audio Engine)"
        tk.Label(panel, text=info_text, font=("Arial", 9), bg="#2d2d2d", fg="#aaaaaa", justify="left").pack(anchor="w", padx=20, pady=10)

    def update_video_loop(self):
        frame = self.backend.process_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video_loop)

    def update_telemetry_loop(self):
        driver_state = self.backend.driver_status
        if "DROWSY" in driver_state or "CLOSED" in driver_state:
            self.lbl_driver.config(text=f"Driver: {driver_state}", fg="#FF3D00")
        else:
            self.lbl_driver.config(text=f"Driver: {driver_state}", fg="#00E676")

        status = self.backend.status_message
        if "OVER" in status:
            self.lbl_status.config(text=f"Status: {status}", fg="#FF3D00")
        else:
            self.lbl_status.config(text=f"Status: {status}", fg="#00E676")

        self.root.after(1000, self.update_telemetry_loop)

    def on_closing(self):
        self.backend.release_resources()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RoadSafetyGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()