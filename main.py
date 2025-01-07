import cv2
from mediapipe.python.solutions import pose as mp_pose
import threading
from winsound import Beep
from json import load, dump
import tkinter as tk
import ctypes
from time import time as current_time

# TO DO :monitor cpu usage, reduce frame processing rate if necessary...
# from psutil import cpu_percent



class PostureMonitor:
    def __init__(self):
        self.default_threshold = 0.35
        self.default_slouching_duration = 5
        self.slouching_start = None
        self.threshold, self.slouching_duration = self.load_settings()

        # Initialize Mediapipe Pose
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        
        # Variables for sound control
        self.stop_event = threading.Event()
        self.stop_event.set()  # Start with sound stopped
        self.sound_thread = None
        
        # Load Haar Cascade classifiers
        self.frontal_face_classifier = cv2.CascadeClassifier("frontal_face_detection_openCV_Github_file.xml")
        self.profile_face_classifier = cv2.CascadeClassifier("profile_side_face_detection_openCV_Github_file.xml")

    # Load the threshold from the settings file
    def load_settings(self):
        try:
            with open("posture_settings.json", "r") as f:
                settings = load(f)
                return settings.get('threshold'), settings.get('slouching_duration')
        except FileNotFoundError:
                return self.default_threshold, self.default_slouching_duration
    
    def save_settings(self, threshold, slouching_duration):
        with open('posture_settings.json', "w") as f:
            dump({"threshold": self.threshold, "slouching_duration": self.slouching_duration}, f)

    def play_sound_loop(self):
        while not self.stop_event.is_set():
            Beep(1000, 1000)  # Single continuous 1000Hz beep, 1 second duration

    def get_ear_positions(self, results, frame, frontal_faces, profile_faces):
            # Initialize ear positions
            left_ear, right_ear = None, None

            # Use Mediapipe ear landmarks if available
            if results.pose_landmarks:
                left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].z
                right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].z

            # Fallback to Haar Cascade face detection if Mediapipe fails to detect ears
            if left_ear is None or right_ear is None:
                if len(frontal_faces) > 0:
                    # Use the first detected frontal face
                    left_ear, right_ear = self.estimate_ear_positions(frontal_faces[0], frame.shape[1], frame.shape[0])
                elif len(profile_faces) > 0:
                    # Use the first detected profile face
                    left_ear, right_ear = self.estimate_ear_positions(profile_faces[0], frame.shape[1], frame.shape[0])
            return left_ear, right_ear

    # Function to estimate ear positions from face detection
    def estimate_ear_positions(self, face_rect, frame_width, frame_height):
        x, y, w, h = face_rect
        # Estimate ear positions relative to the face rectangle
        left_ear = (x + int(w * 0.2), y + int(h * 0.5))  # 20% from the left edge, 50% from the top
        right_ear = (x + int(w * 0.8), y + int(h * 0.5))  # 80% from the left edge, 50% from the top
        return left_ear, right_ear


    def start_monitoring(self):
        # Capture webcam feed
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce frame width for faster processing
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduce frame height for faster processing
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower frame rate to 15 FPS

        frame_skip = 1  # Skip frames to reduce CPU usage
        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip processing for some frames to reduce cpu usage
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            # Convert the frame to RGB for Mediapipe and grayscale for Haar Cascade
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Process the frame with Mediapipe Pose
            results = self.pose.process(rgb_frame)

            # Detect faces using Haar Cascade classifiers
            frontal_faces = self.frontal_face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            profile_faces = self.profile_face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            left_ear, right_ear = self.get_ear_positions(results, frame, frontal_faces, profile_faces)

            # Analyze posture if ear positions are available
            if left_ear is not None and right_ear is not None and results.pose_landmarks:
                # Get keypoints for shoulders (including Z-coordinate)
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z

                # Calculate depth difference between shoulders and ears
                left_diff = abs(left_ear - left_shoulder)
                right_diff = abs(right_ear - right_shoulder)

                # Classify posture
                if left_diff > self.threshold or right_diff > self.threshold:
                    self.handle_bad_posture(frame, left_diff, right_diff)
                else:
                    self.handle_good_posture(frame, left_diff, right_diff)

            # Show the frame
            cv2.imshow('Posture Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.clean_up(cap)

    def display_feedback(self, frame, posture, color):
            # Display feedback
            cv2.putText(frame, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


    def handle_bad_posture(self, frame, left_diff, right_diff):
            posture = f" bad : L {left_diff:.2f} and R {right_diff:.2f}"
            color = (0, 0, 255)  # Red
            self.display_feedback(frame, posture, color)

            if self.slouching_start is None:
                self.slouching_start = current_time() # slouching start time

            if (current_time() - self.slouching_start) > self.slouching_duration:
                if self.sound_thread is None or not self.sound_thread.is_alive():
                    self.stop_event.clear()
                    self.sound_thread = threading.Thread(target=self.play_sound_loop)
                    self.sound_thread.daemon = True  # Make thread daemon so it exits when main program exits
                    self.sound_thread.start()

    def handle_good_posture(self, frame, left_diff, right_diff):
        posture = f" good : L {left_diff:.2f} and R {right_diff:.2f}"
        color = (0, 255, 0)  # Green
        self.display_feedback(frame, posture, color)
        self.slouching_start = None
        self.stop_event.set()

    def clean_up(self, cap):
        self.stop_event.set()
        if self.sound_thread is not None and self.sound_thread.is_alive():
            self.sound_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()

class PostureApp:
    def __init__(self):
        self.monitor = PostureMonitor()  # Create an instance of PostureMonitor

    def run_gui(self):
        posture_monitor_interface = tk.Tk()
        posture_monitor_interface.title("Posture Monitor Settings")
        posture_monitor_interface.geometry("350x220")

        # this is the part that displays the app icon on both the taskbar and the tkinter window 
        myappid = 'tkinter.python.test'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        posture_monitor_interface.iconbitmap("posture_monitor_icon.ico")

        # the tkinter interface isn't automatically the main active window, so force focus on it
        posture_monitor_interface.focus_force()

        # Font styles
        heading_font = ("Helvetica", 14, "bold")
        label_font = ("Helvetica", 12)
        button_font = ("Helvetica", 12, "bold")

        # Header
        heading_label = tk.Label(master=posture_monitor_interface, text="Posture Monitor", fg='#388E3C', font=heading_font)
        heading_label.pack(pady=10)

        # Threshold
        threshold_var = tk.DoubleVar(value=self.monitor.threshold)
        slouching_duration_var = tk.IntVar(value=self.monitor.slouching_duration)

        def save_and_launch():
            self.monitor.save_settings(float(threshold_var.get()), int(slouching_duration_var.get()))
            posture_monitor_interface.destroy()
            self.monitor.start_monitoring()

        # Widgets
        tk.Label(master= posture_monitor_interface, font= label_font,
                        text = "Adjust sensitivity threshold \n (Lower = more sensitive, Higher = less sensitive)").pack()
        tk.Spinbox(posture_monitor_interface, textvariable=threshold_var, from_=0.25, to=0.5, increment=0.01, font=label_font, state="readonly").pack(pady=5)

        tk.Label(master= posture_monitor_interface, font= label_font,
                        text = "Duration in seconds after which you will be notified").pack()
        tk.Spinbox(posture_monitor_interface, textvariable=slouching_duration_var, from_=1, to=30, increment=1, font=label_font, state="readonly").pack(pady=5)

        tk.Button(posture_monitor_interface, text="Save and Start Monitoring",
                                        bg='#4CAF50', fg='white', activebackground='#388E3C', activeforeground='white', relief="raised",
                                        command=save_and_launch, font=button_font).pack(pady=10)

        # associate some action/shortcut/keyboard clicks with a function
        posture_monitor_interface.bind("<Control-w>", lambda _ : posture_monitor_interface.destroy())
        posture_monitor_interface.mainloop()


if __name__ == "__main__":
    app = PostureApp()
    app.run_gui()