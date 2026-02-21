import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np

# =========================
# PATHS
# =========================
VIDEO_PATH = "data/cricket2.mp4"
OBJECT_MODEL = "runs/detect/train/weights/best.pt"
FACE_MODEL = "models/yolov8n-face.pt"   # your face model

# =========================
# LOAD MODELS
# =========================
obj_model = YOLO(OBJECT_MODEL)
face_model = YOLO(FACE_MODEL)

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Video not opened")
    exit()
else:
    print("✅ Video opened")

while True:
    print("Reading frame")
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # OBJECT DETECTION
    # =========================
    obj_res = obj_model(frame)[0]

    if obj_res.boxes is not None:
        for box in obj_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"Obj {cls} {conf:.2f}",
                        (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # =========================
    # FACE DETECTION + EMBEDDING
    # =========================
    face_res = face_model(frame)[0]

    if face_res.boxes is not None:
        for box in face_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            try:
                emb = DeepFace.represent(
                    face,
                    model_name="ArcFace",
                    enforce_detection=False
                )[0]["embedding"]

                emb = np.array(emb)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(frame,"Face",(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

            except:
                pass

    cv2.imshow("Video Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()