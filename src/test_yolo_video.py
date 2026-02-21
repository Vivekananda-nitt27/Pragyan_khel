import cv2
from ultralytics import YOLO

# Load model (person detection first — face later)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data/test.mp4")  # put any video here

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()