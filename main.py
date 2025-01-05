import cv2
from mediapipe.python.solutions import pose as mp_pose
import threading
import winsound
from json import load, dump
import tkinter as tk
import ctypes
from time import time as current_time

default_threshold = 0.35
default_slouching_duration = 5 
slouching_start = None

# Load the threshold from the settings file
try:
    with open("posture_settings.json", "r") as f:
        settings = load(f)
        threshold = settings.get('threshold')
        slouching_duration = settings.get('slouching_duration')
except FileNotFoundError:
    threshold = default_threshold
    slouching_duration = default_slouching_duration

def play_sound_loop(stop_event):
    while not stop_event.is_set():
        winsound.Beep(1000, 1000)  # Single continuous 1000Hz beep, 1 second duration

def start_monitoring():
    global threshold
    global slouching_start
    global slouching_duration

    # Initialize Mediapipe Pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    # Load Haar Cascade classifiers
    frontal_face_classifier = cv2.CascadeClassifier("frontal_face_detection_openCV_Github_file.xml")
    profile_face_classifier = cv2.CascadeClassifier("profile_side_face_detection_openCV_Github_file.xml")

    # Function to estimate ear positions from face detection
    def estimate_ear_positions(face_rect):
        x, y, w, h = face_rect
        # Estimate ear positions relative to the face rectangle
        left_ear = (x + int(w * 0.2), y + int(h * 0.5))  # 20% from the left edge, 50% from the top
        right_ear = (x + int(w * 0.8), y + int(h * 0.5))  # 80% from the left edge, 50% from the top
        return left_ear, right_ear

    # Variables for sound control
    stop_event = threading.Event()
    stop_event.set()  # Start with sound stopped
    sound_thread = None

    # Capture webcam feed
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce frame width for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce frame height for faster processing
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower frame rate to 15 FPS

    frame_skip = 1  # Skip frames to reduce CPU usage
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip processing for some frames
        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        # Convert the frame to RGB for Mediapipe and grayscale for Haar Cascade
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process the frame with Mediapipe Pose
        results = pose.process(rgb_frame)

        # Detect faces using Haar Cascade classifiers
        frontal_faces = frontal_face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        profile_faces = profile_face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
                left_ear, right_ear = estimate_ear_positions(frontal_faces[0], frame.shape[1], frame.shape[0])
            elif len(profile_faces) > 0:
                # Use the first detected profile face
                left_ear, right_ear = estimate_ear_positions(profile_faces[0], frame.shape[1], frame.shape[0])

        # Analyze posture if ear positions are available
        if left_ear is not None and right_ear is not None and results.pose_landmarks:
            # Get keypoints for shoulders (including Z-coordinate)
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z

            # Calculate depth difference between shoulders and ears
            left_diff = abs(left_ear - left_shoulder)
            right_diff = abs(right_ear - right_shoulder)

            # Classify posture
            if left_diff > threshold or right_diff > threshold:
                state = "bad"
                posture = f" {state} : L {left_diff:.2f} and R {right_diff:.2f}"
                color = (0, 0, 255)  # Red

                if slouching_start is None:
                    slouching_start = current_time() # slouching start time

                if slouching_start is not None and (current_time() - slouching_start) > slouching_duration:
                    if sound_thread is None or not sound_thread.is_alive():
                        stop_event.clear()
                        sound_thread = threading.Thread(target=play_sound_loop, args=(stop_event,))
                        sound_thread.daemon = True  # Make thread daemon so it exits when main program exits
                        sound_thread.start()
            else:
                state = "good"
                posture = f"{state} : L {left_diff:.2f} and R {right_diff:.2f}"
                color = (0, 255, 0)  # Green
                slouching_start = None
                stop_event.set()


            # Display posture feedback
            cv2.putText(frame, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Posture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    stop_event.set()

    if sound_thread is not None and sound_thread.is_alive():
        sound_thread.join(timeout=2)

    cap.release()
    cv2.destroyAllWindows()




posture_monitor_interface = tk.Tk()
posture_monitor_interface.title("POSTURE MONITOR SETTINGS")
posture_monitor_interface.geometry("350x220")

# this is the part that displays the app icon on both the taskbar and the tkinter window 
myappid = 'tkinter.python.test'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
posture_monitor_interface.iconbitmap("posture_monitor_icon.ico")


# the tkinter interface isn't automatically the main active window, so force focus on it
posture_monitor_interface.focus_force()

# Set font styles
heading_font = ("Helvetica", 14, "bold")
label_font = ("Helvetica", 12)
button_font = ("Helvetica", 12, "bold")

# Add a header with custom styling
heading_label = tk.Label(master=posture_monitor_interface, text="Posture Monitor", fg='#388E3C', font= heading_font)
heading_label.pack(pady=10)

threshold_text = tk.Label(master= posture_monitor_interface, font= label_font,
                        text = "Adjust sensitivity threshold \n (Lower = more sensitive, Higher = less sensitive)")
threshold_text.pack()

# Use a StringVar to hold the current threshold value
threshold_var = tk.StringVar(value=f"{threshold}")  # Set the initial value to the loaded threshold
slouching_duration_var = tk.StringVar(value=f"{slouching_duration}")  # Set the initial value to the loaded slouching duration
print(threshold_var)

def update_settings():
    threshold = float(threshold_var.get())
    slouching_duration = float(slouching_duration_var.get())
    with open("posture_settings.json", "r") as f:
        settings = load(f)
        settings["threshold"] = threshold
        settings["slouching_duration"] = slouching_duration
    with open("posture_settings.json", "w") as f:
        dump(settings, f)
    print(f"Threshold is {threshold}, duration is {slouching_duration}")


# Function to close Tkinter interface and start main process
def close_and_launch():
        update_settings()  # Save the updated threshold
        posture_monitor_interface.destroy()  # Close Tkinter window
        start_monitoring()  # Start the main posture analysis process

# the spinbox will update the value of threshold_var variable, not the threshold variable
spinbox_threshold = tk.Spinbox(
    master=posture_monitor_interface,
    textvariable=threshold_var,
    from_= 0.25,  # Minimum value
    to=0.5,  # Maximum value
    increment=0.01,  # Step size
    width=8,
    font=label_font,
    format="%.2f",  # Format with 2 decimals
    state="readonly",  # Make the Spinbox read-only, so that the user can't manually input values
)
spinbox_threshold.pack(pady=5)

slouching_duration_text = tk.Label(master= posture_monitor_interface, font= label_font,
                        text = "Duration in seconds after which you will be notified")
slouching_duration_text.pack()


spinbox_slouching_duration = tk.Spinbox(
    master=posture_monitor_interface,
    textvariable= slouching_duration_var,
    from_= 1,  # Minimum value
    to=30,  # Maximum value
    increment= 1,  # Step size
    width=8,
    font=label_font,
    format="%.2f",  # Format with 2 decimals
    state="readonly",  # Make the Spinbox read-only, so that the user can't manually input values
)
spinbox_slouching_duration.pack(pady=5)

# NOTE : in order to assign a button click to 2 commands, create a function that calls both functions
validate_threshold_button = tk.Button(master=posture_monitor_interface, font=button_font,
                                    bg='#4CAF50', fg='white', activebackground='#388E3C', activeforeground='white',
                                    relief="raised",
                                    text="Start Posture Monitoring", command= close_and_launch)
validate_threshold_button.pack()

# associate some action/shortcut/keyboard clicks with a function
posture_monitor_interface.bind("<Control-w>", lambda _ : posture_monitor_interface.destroy())
posture_monitor_interface.mainloop()
