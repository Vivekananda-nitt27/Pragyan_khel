import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# =====================================================
# CONFIG
# =====================================================
VIDEO_PATH = "data/cricket.mp4"
MODEL_PATH = "models/yolov8n-face.pt"
DB_PATH = "face_db.json"

SIMILARITY_THRESHOLD = 0.6


# =====================================================
# Load YOLO
# =====================================================
model = YOLO(MODEL_PATH)


# =====================================================
# Load identity DB (JSON)
# =====================================================
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r") as f:
        face_db = json.load(f)
    face_db = {int(k): np.array(v) for k, v in face_db.items()}
else:
    face_db = {}

next_id = max(face_db.keys(), default=0) + 1


# =====================================================
# Cosine similarity
# =====================================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =====================================================
# Video
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    results = model(frame)[0]

    if results.boxes is not None:
        for box in results.boxes.xyxy:

            x1, y1, x2, y2 = map(int, box)

            # =====================================================
            # ⭐ FACE SIZE FILTER (FINAL — tuned for your video)
            # =====================================================
            w = x2 - x1
            h = y2 - y1

            # remove tiny detections
            if w < 60 or h < 60:
                continue

            # adaptive rule
            if w < frame_w * 0.045:
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # =====================================================
            # Embedding
            # =====================================================
            try:
                emb = DeepFace.represent(
                    face,
                    model_name="ArcFace",
                    enforce_detection=False
                )[0]["embedding"]

                emb = np.array(emb)

            except:
                continue

            # =====================================================
            # Identity matching
            # =====================================================
            assigned_id = None

            for fid, db_emb in face_db.items():
                sim = cosine_similarity(emb, db_emb)
                if sim > SIMILARITY_THRESHOLD:
                    assigned_id = fid
                    break

            # =====================================================
            # New identity → store
            # =====================================================
            if assigned_id is None:
                assigned_id = next_id
                face_db[assigned_id] = emb
                next_id += 1

                print(f"New face stored → ID {assigned_id}")

            # =====================================================
            # Draw
            # =====================================================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {assigned_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Identity Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()


# =====================================================
# Save DB → JSON
# =====================================================
save_db = {k: v.tolist() for k, v in face_db.items()}

with open(DB_PATH, "w") as f:
    json.dump(save_db, f)

print("Face DB saved ✅")