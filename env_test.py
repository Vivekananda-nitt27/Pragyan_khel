import torch
import cv2
import ultralytics
import insightface
import onnxruntime as ort

print("----- ENV TEST START -----")

# Torch GPU
print("\n[1] TORCH GPU")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# OpenCV
print("\n[2] OPENCV")
print("OpenCV version:", cv2.__version__)

# Ultralytics
print("\n[3] ULTRALYTICS")
print("Ultralytics imported OK")

# InsightFace
print("\n[4] INSIGHTFACE")
print("InsightFace version:", insightface.__version__)

# ONNX Runtime
print("\n[5] ONNX RUNTIME")
print("Providers:", ort.get_available_providers())

print("\n----- ENV TEST DONE -----")