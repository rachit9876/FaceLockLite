import cv2
import mediapipe as mp
import time
import os

# Constants
DETECTION_CONFIDENCE = 0.9
MULTIPLE_FACE_LOCK_DELAY = 0.0  # seconds

# Setup MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=DETECTION_CONFIDENCE)

# Webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam to maximum resolution (example: 1920x1080)
# You can try other resolutions like 1280x720, 3840x2160, etc., depending on your camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify the resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution: {width}x{height}")

# Tracking multiple face detection time
multiple_faces_start_time = None
locked = False

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    results = face_detection.process(rgb_frame)
    num_faces = len(results.detections) if results.detections else 0

    # Draw bounding boxes
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Lock condition: 2+ faces for 0.5s
    if num_faces >= 2:
        if multiple_faces_start_time is None:
            multiple_faces_start_time = time.time()
        elif (time.time() - multiple_faces_start_time) >= MULTIPLE_FACE_LOCK_DELAY and not locked:
            os.system("rundll32.exe user32.dll,LockWorkStation")
            locked = True
    else:
        multiple_faces_start_time = None
        locked = False  # reset lock flag if condition not met

    # Display the frame
    cv2.imshow("Simple Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()