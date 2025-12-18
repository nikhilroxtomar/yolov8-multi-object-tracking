from ultralytics import YOLO
import cv2
import os
import csv
from collections import defaultdict
import numpy as np

## Configration
MODEL_PATH = "runs/detect/mot17_yolov8n2/weights/best.pt"
IMAGE_DIR = "/media/nikhil/New Volume/ML_DATASET/MOT17/test/MOT17-03-FRCNN/img1"

OUTPUT_DIR = "outputs"
VIDEO_OUT = os.path.join(OUTPUT_DIR, "tracked_trajectory_velocity.mp4")
CSV_OUT = os.path.join(OUTPUT_DIR, "tracks.csv")

IMG_EXT = ".jpg"
CONF_THRES = 0.25

MAX_TRAJ_LENGTH = 30
VELOCITY_SCALE = 2.5   # controls arrow length
MAX_ARROW_LEN = 40     # pixel cap

os.makedirs(OUTPUT_DIR, exist_ok=True)

## Load Model
model = YOLO(MODEL_PATH)

## Prepare Video Writer
images = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(IMG_EXT))
assert images, "No images found!"

first_img = cv2.imread(os.path.join(IMAGE_DIR, images[0]))
h, w, _ = first_img.shape

video_writer = cv2.VideoWriter(
    VIDEO_OUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    25,
    (w, h),
)

## CSV Setup
csv_file = open(CSV_OUT, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "object_id", "x", "y", "w", "h"])

## Trajectory Storage
tracks_history = defaultdict(list)

## Tracking Loop
frame_idx = 0

for img_name in images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    frame = cv2.imread(img_path)

    results = model.track(
        source=frame,
        conf=CONF_THRES,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for (x, y, bw, bh), obj_id in zip(boxes, ids):
            x1 = int(x - bw / 2)
            y1 = int(y - bh / 2)
            cx, cy = int(x), int(y)

            # Update trajectory
            traj = tracks_history[obj_id]
            traj.append((cx, cy))
            if len(traj) > MAX_TRAJ_LENGTH:
                traj.pop(0)

            ## Draw Bounding Box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x1 + int(bw), y1 + int(bh)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID {obj_id}",
                (x1, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            ## Draw Fading Trajectory
            for i in range(1, len(traj)):
                alpha = i / len(traj)  # older → lighter
                color = (
                    int(255 * alpha),
                    0,
                    int(255 * (1 - alpha))
                )

                cv2.line(
                    frame,
                    traj[i - 1],
                    traj[i],
                    color,
                    2
                )

            ## Draw Velocity Vector
            if len(traj) >= 2:
                (x_prev, y_prev), (x_curr, y_curr) = traj[-2], traj[-1]
                vx = x_curr - x_prev
                vy = y_curr - y_prev

                length = min(
                    MAX_ARROW_LEN,
                    int(np.hypot(vx, vy) * VELOCITY_SCALE)
                )

                if length > 1:
                    norm = np.hypot(vx, vy)
                    dx = int((vx / norm) * length)
                    dy = int((vy / norm) * length)

                    cv2.arrowedLine(
                        frame,
                        (cx, cy),
                        (cx + dx, cy + dy),
                        (0, 0, 255),  # red arrow
                        2,
                        tipLength=0.4
                    )

            ## Write CSV
            csv_writer.writerow([
                frame_idx,
                obj_id,
                round(x1, 2),
                round(y1, 2),
                round(bw, 2),
                round(bh, 2),
            ])

    video_writer.write(frame)
    frame_idx += 1

# -----------------------------
# Cleanup
# -----------------------------
video_writer.release()
csv_file.close()

print("Tracking + fading trajectory + velocity vectors completed")
print(f"Video saved to: {VIDEO_OUT}")
print(f"CSV saved to: {CSV_OUT}")
