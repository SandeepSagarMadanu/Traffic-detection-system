import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# Constants
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
MAR_THRESHOLD = 0.75  # Mouth Aspect Ratio threshold
CONSECUTIVE_FRAMES_EYE = 20  # Number of consecutive frames for eye closure
CONSECUTIVE_FRAMES_MOUTH = 15  # Number of consecutive frames for yawning

# Initialize counters
eye_closed_frames = 0
yawning_frames = 0

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    # Vertical distances
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth_landmarks):
    # Vertical distances
    A = dist.euclidean(mouth_landmarks[0], mouth_landmarks[2])  # Upper lip top to lower lip top
    B = dist.euclidean(mouth_landmarks[1], mouth_landmarks[3])  # Upper lip bottom to lower lip bottom
    # Horizontal distance
    D = dist.euclidean(mouth_landmarks[4], mouth_landmarks[5])  # Left corner to right corner
    # MAR formula
    mar = (A + B) / (2.0 * D)
    return mar

# Function to draw landmarks on the frame
def draw_landmarks(frame, landmarks):
    for landmark in landmarks:
        x = int(landmark[0] * frame.shape[1])
        y = int(landmark[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

# Real-time drowsiness detection
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye and mouth landmarks
            eye_landmarks_left = [
                (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y),
                (face_landmarks.landmark[160].x, face_landmarks.landmark[160].y),
                (face_landmarks.landmark[158].x, face_landmarks.landmark[158].y),
                (face_landmarks.landmark[133].x, face_landmarks.landmark[133].y),
                (face_landmarks.landmark[153].x, face_landmarks.landmark[153].y),
                (face_landmarks.landmark[144].x, face_landmarks.landmark[144].y),
            ]
            eye_landmarks_right = [
                (face_landmarks.landmark[362].x, face_landmarks.landmark[362].y),
                (face_landmarks.landmark[385].x, face_landmarks.landmark[385].y),
                (face_landmarks.landmark[387].x, face_landmarks.landmark[387].y),
                (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y),
                (face_landmarks.landmark[373].x, face_landmarks.landmark[373].y),
                (face_landmarks.landmark[380].x, face_landmarks.landmark[380].y),
            ]
            mouth_landmarks = [
                (face_landmarks.landmark[13].x, face_landmarks.landmark[13].y),  # Upper lip top
                (face_landmarks.landmark[14].x, face_landmarks.landmark[14].y),  # Upper lip bottom
                (face_landmarks.landmark[17].x, face_landmarks.landmark[17].y),  # Lower lip top
                (face_landmarks.landmark[18].x, face_landmarks.landmark[18].y),  # Lower lip bottom
                (face_landmarks.landmark[78].x, face_landmarks.landmark[78].y),  # Left corner
                (face_landmarks.landmark[308].x, face_landmarks.landmark[308].y), # Right corner
            ]

            # Calculate EAR and MAR
            ear_left = eye_aspect_ratio(eye_landmarks_left)
            ear_right = eye_aspect_ratio(eye_landmarks_right)
            ear_avg = (ear_left + ear_right) / 2.0
            mar = mouth_aspect_ratio(mouth_landmarks)

            # Detect drowsiness
            if ear_avg < EAR_THRESHOLD:
                eye_closed_frames += 1
            else:
                eye_closed_frames = 0

            if mar > MAR_THRESHOLD:
                yawning_frames += 1
            else:
                yawning_frames = 0

            # Trigger alert if drowsiness is detected
            if eye_closed_frames > CONSECUTIVE_FRAMES_EYE or yawning_frames > CONSECUTIVE_FRAMES_MOUTH:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Add sound alert (optional)
                # os.system("afplay alert.wav")  # For macOS
                # os.system("aplay alert.wav")  # For Linux

            # Display EAR and MAR on the frame
            cv2.putText(frame, f"EAR: {ear_avg:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw landmarks on the frame
            draw_landmarks(frame, eye_landmarks_left + eye_landmarks_right + mouth_landmarks)

    # Display the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
