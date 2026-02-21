import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/cricket2.mp4"
MODEL_PATH = "runs/detect/train/weights/best.pt"

CONF_THRESHOLD = 0.30
SHRINK = 0.17

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    if results.boxes is not None:
        for box in results.boxes:

            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue   # skip weak detection

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            # shrink
            w = x2 - x1
            h = y2 - y1

            x1n = int(x1 + w * SHRINK)
            y1n = int(y1 + h * SHRINK)
            x2n = int(x2 - w * SHRINK)
            y2n = int(y2 - h * SHRINK)

            cv2.rectangle(frame,(x1n,y1n),(x2n,y2n),(0,255,0),2)
            cv2.putText(frame,f"{cls} {conf:.2f}",
                        (x1n,y1n-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("Object Detect", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()