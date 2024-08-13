import threading
import cv2
from deepface import DeepFace
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Unable to open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_matches = False
matched_name = "No match"

# Define reference images along with their names
reference_images = {
    "Tanvi": cv2.imread("C:\\Users\\Tanvi\\photos\\1714112927629.jpg"),
    "Shagun": cv2.imread("C:\\Users\\Tanvi\\Downloads\\1713771469094.jpg"),
    "Anshika": cv2.imread("C:\\Users\\Tanvi\\photos\\1714112927647.jpg"),
}

def check_face(frame):
    global face_matches, matched_name
    face_matches = False  # Reset face_matches for each check
    try:
        for name, reference_img in reference_images.items():
            result = DeepFace.verify(frame, reference_img, enforce_detection=False)
            if result['verified']:
                face_matches = True
                matched_name = name
                logging.info(f"Face match found: {name}")
                return
    except Exception as e:
        logging.error(f"Error in face verification: {e}")
        face_matches = False

# Capture one frame from the camera
ret, frame = cap.read()
if ret:
    # Check the face once
    check_face(frame)

    # Display the result on the frame
    if face_matches:
        cv2.putText(frame, f"MATCH: {matched_name}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with the result
    cv2.imshow('Face Verification', frame)

    # Wait for a key press to close the window
    cv2.waitKey(0)

# Release resources
cap.release()
cv2.destroyAllWindows()
