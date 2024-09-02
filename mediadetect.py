import cv2
import torch
from deepface import DeepFace
import os
import csv
from datetime import datetime, timedelta

# Load YOLOv5 model for detecting bodies
body_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
body_model.eval()

# CSV file for logging detections
log_file = 'detection_log.csv'
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Gender", "Age", "Emotion"])

def log_detection(timestamp, gender, age, emotion):
    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, gender, age, emotion])

# Load sound notification
sound_path = "notification_sound.wav"

def play_notification_sound():
    os.system(f"aplay {sound_path}")

# Use a video file instead of a camera feed
video_file_path = 'v1.mp4'
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_file_path}.")
    exit()

# Track the last time a detection was logged
last_logged_time = datetime.now()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to grab frame")
        break

    # Detect bodies
    results = body_model(frame)
    bbox_xyxy = results.xyxy[0].cpu().numpy()

    for bbox in bbox_xyxy:
        x1, y1, x2, y2, conf, cls = bbox
        if conf > 0.5 and cls == 0:  # Body detected
            body_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if body_crop.size != 0:
                try:
                    # For demonstration purposes, using DeepFace for facial analysis
                    # You might need a custom model for body-based gender detection
                    body_rgb = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)
                    analysis = DeepFace.analyze(body_rgb, actions=['gender', 'age', 'emotion'], enforce_detection=False)
                    
                    gender = analysis[0]['gender']
                    age = analysis[0]['age']
                    emotion = analysis[0]['dominant_emotion']
                    gender_str = str(gender).lower()
                    color = (0, 255, 0) if gender_str == 'woman' else (0, 0, 255)

                    # Draw a rectangle around the detected body and add gender, age, emotion text
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{gender_str.capitalize()}, {age}, {emotion.capitalize()}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Log the detection with timestamp, only if the interval has passed
                    current_time = datetime.now()
                    if current_time - last_logged_time > timedelta(seconds=1):
                        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        log_detection(timestamp, gender, age, emotion)
                        last_logged_time = current_time

                    # Real-time notification for "woman" detection
                    if gender_str == "woman":
                        play_notification_sound()
                        print("Notification: Woman detected")

                except Exception as e:
                    print(f"Error in gender classification: {e}")

    cv2.imshow('Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
