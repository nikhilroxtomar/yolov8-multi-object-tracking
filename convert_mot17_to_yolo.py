import os
import shutil
import configparser
from tqdm import tqdm
from glob import glob

## Configuration
MOT_ROOT = "../MOT17/train/"
OUT_ROOT = "data"

SEQUENCES = sorted(glob(os.path.join(MOT_ROOT, "MOT17-*FRCNN")))

TRAIN_SEQS = SEQUENCES[:5]
VAL_SEQS = SEQUENCES[5:]

## Create Output Directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

## Utils
def read_seqinfo(seq_path):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(seq_path, "seqinfo.ini"))
    w = int(cfg["Sequence"]["imWidth"])
    h = int(cfg["Sequence"]["imHeight"])
    return w, h

def load_gt(gt_file):
    gt = {}
    with open(gt_file) as f:
        for line in f:
            frame, _, x, y, w, h, _, cls, _ = line.strip().split(",")
            if int(cls) != 1:  # person only
                continue
            gt.setdefault(frame.zfill(6), []).append(
                (float(x), float(y), float(w), float(h))
            )
    return gt

def sanitize_bbox(x, y, w, h, img_w, img_h):
    """
    Clip bounding box to image boundaries.
    Returns None if box becomes invalid.
    """
    ## Clip top-left
    x = max(0.0, x)
    y = max(0.0, y)

    ## Clip width & height
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    ## Reject invalid boxes
    if w <= 1 or h <= 1:
        return None

    return x, y, w, h

## Main Processing
print("Preparing MOT17 for YOLOv8 (with bbox sanitization)...")

for seq_path in TRAIN_SEQS + VAL_SEQS:
    split = "train" if seq_path in TRAIN_SEQS else "val"
    seq = os.path.basename(seq_path)

    img_dir = os.path.join(seq_path, "img1")
    gt_path = os.path.join(seq_path, "gt", "gt.txt")

    img_w, img_h = read_seqinfo(seq_path)
    gt_data = load_gt(gt_path)

    images = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))

    for img_file in tqdm(images, desc=f"{seq} [{split}]"):
        frame_id = os.path.splitext(img_file)[0]

        src_img = os.path.join(img_dir, img_file)
        dst_img = os.path.join(OUT_ROOT, "images", split, f"{seq}_{img_file}")
        dst_lbl = os.path.join(OUT_ROOT, "labels", split, f"{seq}_{frame_id}.txt")

        ## Copy image
        shutil.copy2(src_img, dst_img)

        valid_boxes = []

        for x, y, w, h in gt_data.get(frame_id, []):
            box = sanitize_bbox(x, y, w, h, img_w, img_h)
            if box is None:
                continue

            x, y, w, h = box

            ## Normalize
            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            wn = w / img_w
            hn = h / img_h

            ## Final strict validation
            if not (0.0 < xc < 1.0 and 0.0 < yc < 1.0 and
                    0.0 < wn < 1.0 and 0.0 < hn < 1.0):
                continue

            valid_boxes.append((xc, yc, wn, hn))

        ## Write label file (even if empty)
        with open(dst_lbl, "w") as f:
            for xc, yc, wn, hn in valid_boxes:
                f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

print("MOT17 dataset prepared successfully with sanitized bounding boxes")
