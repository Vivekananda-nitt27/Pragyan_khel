
## Problem Statement : 
AI-Based Smart auto focus & Dynamic subject Tracking System 

## Team Name : INNOVISION

## Drive Link (Demo Video)- 
https://drive.google.com/file/d/15dTUaYdgX-JwdAB-c_SqKpo44_WXESnP/view?usp=sharing

## Members
Vivekananda Sahoo , 
Aman Jayswal 

# 🧠 Multi-Sport Object Tracking Model — Dataset & Training Guide

## 📌 Project Goal

Train a **multi-sport object detection model (YOLO)** capable of detecting players and sports equipment across sports like:

* Cricket
* Football
* Basketball
* Tennis
* Badminton
* Hockey
* Volleyball
* Baseball

This model will later be used for **tracking, interaction logic, and auto-focus camera systems**.

---



## 📂 Dataset Structure (YOLO Format)

```
dataset/
 ├── images/
 │    ├── train/
 │    └── val/
 │
 └── labels/
      ├── train/
      └── val/
```

Each image must have a corresponding `.txt` label file.

Example:

```
image_001.jpg
image_001.txt
```

---

## 🏷️ Label Format (YOLO)

Each line in label file:

```
class_id x_center y_center width height
```

Values must be **normalized (0-1)**.

Example:

```
3 0.52 0.61 0.22 0.40
```

---

## 📸 Step 1 — Collect Images

### Recommended Sources

* YouTube sports highlights (extract frames)
* Google Images
* Roboflow public datasets
* Match broadcast videos (best)

### Minimum Images Per Class

* Prototype: 20–30
* Decent: 80–150
* Good: 300+

⚠️ Ball requires more images.

---

## ✏️ Step 2 — Annotate Images

### Recommended Annotation Tools

* Roboflow (recommended)
* CVAT
* LabelImg
* makesense.ai

Draw bounding boxes for each object and export in **YOLO format**.

---

## 🔁 Step 3 — Data Augmentation

Apply augmentation **after collecting real images**.

Recommended augmentations:

* Horizontal flip
* Scale
* Rotation
* Brightness / contrast
* Motion blur ⭐ important for sports
* Random crop
* Zoom simulation (broadcast style)

Augmentation improves robustness but does not replace real data.

---

## 📄 Step 4 — Create data.yaml

Example:

```yaml
path: dataset

train: images/train
val: images/val

names:
  0: person
  1: ball
  2: racket
  3: bat
  4: stick
  5: helmet
  6: gloves
  7: shoes
  8: pads
  9: guard
  10: goal_post
  11: basketball_hoop
  12: stumps
  13: bails
  14: net
  15: shuttlecock
```

---

## 🚀 Step 5 — Train YOLO Model

Example command:

```
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=100 batch=16
```

---

## ⚙️ Recommended Starter Training Settings

* Model: yolov8n (lightweight)
* Image size: 640
* Epochs: 100
* Batch: 16 (adjust to GPU)
* Mixed precision: enabled (default)

---

## 🧠 Important Dataset Guidelines

✔ Diversity is more important than quantity
✔ Include wide camera + zoom camera
✔ Include motion blur samples
✔ Include different lighting
✔ Include occlusion scenarios

Do NOT use only clean images.

---

## 🔄 Retraining & Model Expansion (IMPORTANT ⭐)

The model is designed to support **continuous retraining** when new objects appear.

### When Retraining Is Needed

Retraining can be performed if:

* A new sports object needs to be tracked
* Detection accuracy is low for a class
* A new sport is added
* Domain changes (different broadcast style / camera)
* More data becomes available

### Retraining Strategy

1. Collect new images for the new object
2. Annotate using the same YOLO format
3. Add the new class to `data.yaml`
4. Merge new data with existing dataset
5. Resume training (fine-tuning)

Example:

```
yolo detect train model=last.pt data=data.yaml epochs=50
```

This allows the model to **learn new objects without training from scratch**.

---



## 📈 Recommended Training Strategy

### Phase 1 (Core Objects)

* person
* ball
* bat
* racket
* helmet
* stumps

### Phase 2 (Full Classes)

Add remaining classes.

---

## 🔮 Future Expansion (V2)

* Ball type split (football / cricket / tennis)
* Sport specific classes
* Referee detection
* Player interaction logic
* Tracking integration (ByteTrack / BoT-SORT)
* Re-identification embeddings

---

## 🎯 Final Outcome

After training, this model enables:

* Multi-sport object detection
* Player-equipment linking
* Event detection
* Tracking pipelines
* Auto camera focus systems

---

## ✅ Current Status Checklist

* [ ] Class list defined
* [ ] Images collected
* [ ] Annotation completed
* [ ] Dataset structure ready
* [ ] data.yaml created
* [ ] Training started

---

**This README describes the full pipeline for training a multi-sport YOLO object detection model and supports future retraining when new objects need to be tracked.**
