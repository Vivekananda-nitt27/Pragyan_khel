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

CONF_THRESHOLD = 0.33         # ⭐ CONF FILTER
SIMILARITY_THRESHOLD = 0.65    # Identity match threshold


# =====================================================
# LOAD YOLO
# =====================================================
model = YOLO(MODEL_PATH)


# =====================================================
# LOAD JSON DB (identity memory)
# =====================================================
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r") as f:
        face_db = json.load(f)
    face_db = {int(k): np.array(v) for k, v in face_db.items()}
else:
    face_db = {}

next_id = max(face_db.keys(), default=0) + 1


# =====================================================
# COSINE SIMILARITY
# =====================================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =====================================================
# VIDEO LOOP
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    results = model(frame)[0]

    if results.boxes is not None:

        boxes = results.boxes.xyxy
        confs = results.boxes.conf

        for i in range(len(boxes)):

            x1, y1, x2, y2 = map(int, boxes[i])
            conf = float(confs[i])

            # =====================================================
            # ⭐ CONFIDENCE FILTER (NEW UPGRADE)
            # =====================================================
            if conf < CONF_THRESHOLD:
                continue

            # =====================================================
            # ⭐ FRAME / SIZE FILTER (REDUCED — CRICKET TUNED)
            # Pixel rule → keeps far players
            # Ratio rule → removes tiny audience faces
            # =====================================================
            w = x2 - x1
            h = y2 - y1

            if w < 32 or h < 32:           # pixel constraint
                continue

            if w < frame_w * 0.028:        # frame ratio constraint
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # =====================================================
            # ⭐ EMBEDDING (ArcFace)
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
            # ⭐ IDENTITY MATCHING
            # =====================================================
            assigned_id = None

            for fid, db_emb in face_db.items():
                sim = cosine_similarity(emb, db_emb)

                if sim > SIMILARITY_THRESHOLD:
                    assigned_id = fid
                    break

            # =====================================================
            # ⭐ NEW IDENTITY → STORE
            # =====================================================
            if assigned_id is None:
                assigned_id = next_id
                face_db[assigned_id] = emb
                next_id += 1

                print(f"New face stored → ID {assigned_id}")

            # =====================================================
            # DRAW RESULT
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
# ⭐ SAVE DB → JSON (PERSISTENT MEMORY)
# =====================================================
save_db = {k: v.tolist() for k, v in face_db.items()}

with open(DB_PATH, "w") as f:
    json.dump(save_db, f)

print("Face DB saved ✅")