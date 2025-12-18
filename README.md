![](banner.png)
# Multi-Object Tracking with YOLOv8 and ByteTrack

An end-to-end computer vision project for **object detection, multi-object tracking, and motion visualization** using YOLOv8 and ByteTrack.

This repository demonstrates how to build a clean, local inference pipeline that goes beyond object detection by adding **persistent identities**, **motion understanding**, and **trajectory visualization**.

---

## Project Overview

This project implements a complete **multi-object tracking (MOT) pipeline**:

1. Fine-tune a YOLOv8 detector on a video-based dataset  
2. Perform online multi-object tracking using ByteTrack  
3. Visualize object trajectories and motion dynamics  
4. Export structured tracking outputs for downstream analysis  

The focus is on **engineering clarity, reproducibility, and interpretability**, rather than UI or cloud deployment.

---

## Features

### Object Detection
- Fine-tuned **YOLOv8** (Ultralytics)
- Anchor-free, single-stage detector
- Trained on **MOT17 (person class)**

### Multi-Object Tracking
- **ByteTrack** for online tracking
- Persistent object IDs across frames
- Robust to occlusions and missed detections

### Motion Visualization
- Per-object **trajectories** with fading history
- **Velocity vectors** showing direction and relative speed
- Stable ID visualization across frames

### Outputs
- Annotated video with boxes, IDs, trajectories, and velocity arrows
- Frame-level CSV containing tracking data

---

## Design Philosophy

This project is designed to reflect **real-world computer vision systems**, not just model training:

- Clear separation of data, models, scripts, and outputs
- Local inference (no cloud or UI dependencies)
- Modular code that can be extended easily
- Visualizations focused on interpretability

---

## Repository Structure
```
.
├── convert_mot17_to_yolo.py
├── data
│   ├── dataset.yaml
│   ├── images
│   └── labels
├── models
├── outputs
│   ├── tracked_trajectory_velocity.mp4
│   └── tracks.csv
├── README.md
├── requirements.txt
├── runs
│   └── detect
├── track_mot17.py
├── train_yolov8.py
├── yolo11n.pt
└── yolov8n.pt

7 directories, 10 files
```
> Training artifacts under `runs/` are generated automatically by Ultralytics and are not part of the final model deliverables.

## Dataset

- **MOT17** (Multiple Object Tracking Benchmark)
- Person class only
- Sequence-level train/validation split

> The dataset is **not included** in this repository.  
> Please download MOT17 separately and update paths in the scripts.

## Data Preparation

Convert MOT17 annotations to YOLO format:

``` bash
python scripts/prepare_mot17_for_yolov8.py
```

This step:
- Converts MOT annotations to YOLO format
- Ensures bounding boxes are valid and normalized
- Produces a YOLOv8-compatible dataset layout

## Training YOLOv8

Fine-tune a pretrained YOLOv8 model:

``` bash
python scripts/train_yolov8.py
```

## Tracking and Visualization

Run detection + tracking with motion visualization:

``` bash
python scripts/track_mot17.py
```

This generates:
- Annotated video with:
    - Bounding boxes
    - Persistent IDs
    - Fading trajectories
    - Velocity vectors
- Frame-level CSV output with object positions

## Technical Highlights

- YOLOv8 anchor-free detection
- ByteTrack motion-based association
- Trajectory history visualization
- Velocity estimation from motion
- Clean, modular Python implementation
