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

CONF_THRESHOLD = 0.33
MATCH_THRESHOLD = 0.65
UPDATE_THRESHOLD = 0.70
MAX_EMB_PER_ID = 10


# =====================================================
# LOAD YOLO FACE
# =====================================================
model = YOLO(MODEL_PATH)


# =====================================================
# LOAD DB (MULTI EMB)
# =====================================================
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r") as f:
        raw = json.load(f)

    face_db = {
        int(k): [np.array(e) for e in v]
        for k, v in raw.items()
    }
else:
    face_db = {}

next_id = max(face_db.keys(), default=0) + 1


# =====================================================
# COSINE
# =====================================================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =====================================================
# MATCH FUNCTION (multi emb)
# =====================================================
def match_identity(emb, db):
    best_id = None
    best_sim = 0

    for fid, emb_list in db.items():
        sims = [cosine(emb, e) for e in emb_list]
        sim = max(sims)

        if sim > best_sim:
            best_sim = sim
            best_id = fid

    if best_sim >= MATCH_THRESHOLD:
        return best_id, best_sim

    return None, best_sim


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

            if conf < CONF_THRESHOLD:
                continue

            w = x2 - x1
            h = y2 - y1

            if w < 32 or h < 32:
                continue

            if w < frame_w * 0.028:
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # ================= EMBEDDING =================
            try:
                emb = DeepFace.represent(
                    face,
                    model_name="ArcFace",
                    enforce_detection=False
                )[0]["embedding"]

                emb = np.array(emb)

            except:
                continue

            # ================= MATCH =================
            assigned_id, best_sim = match_identity(emb, face_db)

            # ================= NEW ID =================
            if assigned_id is None:
                assigned_id = next_id
                face_db[assigned_id] = [emb]
                next_id += 1

                print(f"New ID → {assigned_id}")

            else:
                # ================= UPDATE BUFFER =================
                if best_sim >= UPDATE_THRESHOLD:

                    emb_list = face_db[assigned_id]

                    # avoid duplicate embeddings
                    sims = [cosine(emb, e) for e in emb_list]
                    if max(sims) < 0.95:

                        emb_list.append(emb)

                        if len(emb_list) > MAX_EMB_PER_ID:
                            emb_list.pop(0)

            # ================= DRAW =================
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID {assigned_id}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,0),2)

    cv2.imshow("Identity Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


# =====================================================
# SAVE DB
# =====================================================
save_db = {
    k: [e.tolist() for e in v]
    for k, v in face_db.items()
}

with open(DB_PATH, "w") as f:
    json.dump(save_db, f)

print("Face DB saved ✅")