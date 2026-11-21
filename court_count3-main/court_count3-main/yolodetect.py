import torch
import cv2
from picamera2 import Picamera2, Preview
import time

# Initialize and capture image from PiCamera
picam = Picamera2()
picam.start_preview(Preview.NULL)
time.sleep(2)  # Give camera time to adjust (optional but recommended)
picam.capture_file("raspi_count.jpg")
picam.stop_preview()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the captured image
img = cv2.imread('raspi_count.jpg')

# Convert image from BGR (OpenCV format) to RGB (YOLO model format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
results = model(img_rgb)

# Get class ID for 'person'
person_class_id = [k for k, v in model.names.items() if v == 'person'][0]

# Filter results to only include 'person' detections
person_detections = results.xyxy[0][results.xyxy[0][:, 5] == person_class_id]

# Print number of people detected
print(f"Number of people detected: {len(person_detections)}")

# Draw bounding boxes and confidence scores
for *box, conf, cls in person_detections:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'Person {conf:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Resize for display (adjust as needed)
display_img = cv2.resize(img, (1280, 720))

# Display the image with detections
cv2.imshow('Person Detection', display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
