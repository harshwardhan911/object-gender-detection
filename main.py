import cv2
import torch
from deepface import DeepFace
from mtcnn import MTCNN
import pyttsx3
import requests
import numpy as np

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak a message
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Function to get live geolocation using IPstack API
def get_geolocation():
    access_key = '868bd3abe95ae3d7b2d2259fb656b08a'  # Replace with your IPstack access key
    url = f'http://api.ipstack.com/check?access_key={access_key}'
    response = requests.get(url)
    data = response.json()
    latitude = data.get('latitude', 37.7749)  # Default to San Francisco if latitude is not found
    longitude = data.get('longitude', -122.4194)  # Default to San Francisco if longitude is not found
    return latitude, longitude

# Load YOLOv5 model for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Replace this with your phone's IP camera URL
cap = cv2.VideoCapture(0)

# Get screen resolution
screen_res = (1920, 1080)  # Replace with your screen resolution if necessary

frame_skip = 2  # Analyze every 2nd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Downscale the frame for faster processing
    small_frame = cv2.resize(frame, (640, 360))

    # Perform object detection on the smaller frame
    results = yolo_model(small_frame)

    # Extract bounding boxes and labels
    bbox_xyxy = results.xyxy[0].cpu().numpy()
    labels = results.names

    # Initialize counters for gender
    total_people = 0
    women_count = 0
    men_count = 0

    for bbox in bbox_xyxy:
        x1, y1, x2, y2, conf, cls = bbox
        cls = int(cls)  # Convert class index to int
        label = labels[cls]  # Get the label for the class

        # Skip non-person classes (assuming 0 is the class for person)
        if cls != 0:
            continue

        total_people += 1
        # Rescale bounding box coordinates back to the original frame size
        x1 = int(x1 * frame.shape[1] / small_frame.shape[1])
        x2 = int(x2 * frame.shape[1] / small_frame.shape[1])
        y1 = int(y1 * frame.shape[0] / small_frame.shape[0])
        y2 = int(y2 * frame.shape[0] / small_frame.shape[0])

        # Draw bounding box for detected person
        color = (0, 255, 255)  # Yellow for general objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Draw label and confidence score below the bounding box
        text = f"{label} {conf:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1
        text_y = y2 + text_size[1] + 10  # Position text below the bounding box

        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Extract face regions using MTCNN
        faces = mtcnn.detect_faces(frame)
        for face in faces:
            x1, y1, x2, y2 = face['box']
            if x2 > 0 and y2 > 0:  # Ensure face region is valid
                # Extract face region
                face_region = frame[y1:y1 + y2, x1:x1 + x2]
                if face_region.size != 0:
                    # Convert face to RGB format
                    face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    # Use DeepFace for gender classification
                    try:
                        analysis = DeepFace.analyze(face_rgb, actions=['gender'], enforce_detection=False)
                        gender = analysis[0]['gender']
                        # Ensure gender is a string
                        gender_str = str(gender).lower()

                        # Set color and count based on gender
                        if gender_str == 'woman':
                            color = (0, 255, 0)  # Green for women
                            women_count += 1
                        elif gender_str == 'man':
                            color = (0, 0, 255)  # Red for men
                            men_count += 1
                        else:
                            color = (2, 2, 255)  # White for unknown or other

                        # Draw bounding box and gender on the frame (thinner rectangle)
                        cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), color, 1)
                        cv2.putText(frame, gender_str.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except Exception as e:
                        print(f"Error in gender classification: {e}")

    # Determine the majority gender
    majority_gender = None
    if total_people > 0:
        women_percentage = (women_count / total_people) * 100
        men_percentage = (men_count / total_people) * 100

        if women_percentage > 50:
            majority_gender = 'Women'
        elif men_percentage > 50:
            majority_gender = 'Men'

    # Display the majority gender at the top of the frame
    if majority_gender:
        cv2.putText(frame, f"Majority: {majority_gender}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        speak(f"There is a majority of {majority_gender}")

    # Get live geolocation data
    latitude, longitude = get_geolocation()
    # Display coordinates on the frame
    coord_text = f"Lat: {latitude:.4f}, Lon: {longitude:.4f}"
    cv2.putText(frame, coord_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Resize frame to fit screen resolution
    frame_resized = cv2.resize(frame, screen_res)

    # Display the frame in full screen
    cv2.namedWindow('Real-time Object and Gender Detection with Geolocation', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Real-time Object and Gender Detection with Geolocation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Real-time Object and Gender Detection with Geolocation', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
