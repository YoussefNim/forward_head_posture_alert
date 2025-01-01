import cv2
import mediapipe as mp
import threading
import winsound


def play_sound_loop(file_path, stop_event):
    while not stop_event.is_set():
        winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_LOOP)
        stop_event.wait(2)
        print("Sound playing...")
    print("Sound stopped")




# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Load Haar Cascade classifiers
frontal_face_classifier = cv2.CascadeClassifier("frontal_face_detection_openCV_Github_file.xml")
profile_face_classifier = cv2.CascadeClassifier("profile_side_face_detection_openCV_Github_file.xml")

# Function to calculate depth difference
def calculate_depth_difference(shoulder, ear):
    return abs(ear[2] - shoulder[2])  # Depth difference (Z-coordinate)

# Function to estimate ear positions from face detection
def estimate_ear_positions(face_rect, frame_width, frame_height):
    x, y, w, h = face_rect
    # Estimate ear positions relative to the face rectangle
    left_ear = (x + int(w * 0.2), y + int(h * 0.5))  # 20% from the left edge, 50% from the top
    right_ear = (x + int(w * 0.8), y + int(h * 0.5))  # 80% from the left edge, 50% from the top
    return left_ear, right_ear

# Variables for sound control
stop_event = threading.Event()
sound_thread = threading.Thread(target=play_sound_loop, args=("beep one second.wav", stop_event))
sound_thread.start()
# sound_thread = None


# Capture webcam feed
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce frame width for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce frame height for faster processing
cap.set(cv2.CAP_PROP_FPS, 15)  # Lower frame rate to 15 FPS

print("Press 'Q' to stop the program.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe and grayscale for Haar Cascade
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade classifiers
    frontal_faces = frontal_face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profile_faces = profile_face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process the frame with Mediapipe Pose
    results = pose.process(rgb_frame)

    # Initialize ear positions
    left_ear = None
    right_ear = None

    # Use Mediapipe ear landmarks if available
    if results.pose_landmarks:
        left_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].z]
        right_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].z]

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
        left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
        right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]

        # Calculate depth difference between shoulders and ears
        left_diff = calculate_depth_difference(left_shoulder, left_ear)
        right_diff = calculate_depth_difference(right_shoulder, right_ear)

        # Classify posture
        threshold = 0.35  # Adjust this threshold based on testing
        if left_diff > threshold or right_diff > threshold:
            posture = f" bad : L {left_diff:.2f} and R {right_diff:.2f}"
            color = (0, 0, 255)  # Red
            #if sound_thread is None:  # Start sound if not already playing
            stop_event.clear()
                # sound_thread = threading.Thread(target=play_sound_loop, args=("beep one second.wav", stop_event))
                # sound_thread.start()
            print("Sound started: Bad posture detected.")

        else:
            posture = f"good : L {left_diff:.2f} and R {right_diff:.2f}"
            color = (0, 255, 0)  # Green
            #if sound_thread is not None:  # Stop sound if playing
            stop_event.set()
                # sound_thread.join()
                # sound_thread = None
            print("Sound stopped: Posture corrected.")

        # Display posture feedback
        cv2.putText(frame, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Draw circles on keypoints for visualization
        cv2.circle(frame, (int(left_shoulder[0] * frame.shape[1]), int(left_shoulder[1] * frame.shape[0])), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(right_shoulder[0] * frame.shape[1]), int(right_shoulder[1] * frame.shape[0])), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(left_ear[0] * frame.shape[1]), int(left_ear[1] * frame.shape[0])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(right_ear[0] * frame.shape[1]), int(right_ear[1] * frame.shape[0])), 5, (0, 0, 255), -1)

        
    # Show the frame
    cv2.imshow('Posture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if 'Q' is pressed
    # if keyboard.is_pressed('q'):
    #     print("Program stopped by user.")
    #     break

# Clean up
if sound_thread is not None:
    stop_event.set()
    sound_thread.join(timeout=5)
    sound_thread = None

cap.release()
cv2.destroyAllWindows()

