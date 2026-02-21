import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8n-face.pt")

cap = cv2.VideoCapture("data/test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Face Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()