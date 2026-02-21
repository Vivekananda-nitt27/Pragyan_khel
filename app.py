from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from tracker import ObjectTracker

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

tracker = ObjectTracker("runs/detect/train/weights/best.pt")

video_path = None


@app.route("/", methods=["GET","POST"])
def index():
    global video_path

    if request.method == "POST":
        file = request.files["video"]
        if file:
            video_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(video_path)

    return render_template("index.html")


def gen_frames():
    global video_path

    if video_path is None:
        return

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.process(frame)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/click", methods=["POST"])
def click():
    data = request.json
    tracker.select(data["x"], data["y"])
    return {"status":"ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)