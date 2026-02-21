from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print(len(model.names))
print(model.names)