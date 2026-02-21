import cv2
from deepface import DeepFace

cap = cv2.VideoCapture("data/test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        emb = DeepFace.represent(
            frame,
            model_name="ArcFace",
            enforce_detection=False
        )

        print("Embedding length:", len(emb[0]["embedding"]))

    except Exception as e:
        print("skip frame")

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()