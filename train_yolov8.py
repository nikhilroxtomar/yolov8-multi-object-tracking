from ultralytics import YOLO
import torch

def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    DATA_YAML = "data/dataset.yaml"
    MODEL_NAME = "yolov8n.pt"   # start small
    EPOCHS = 30
    IMG_SIZE = 640
    BATCH_SIZE = 16
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    RUN_NAME = "mot17_yolov8n"

    # -----------------------------
    # Load Pretrained Model
    # -----------------------------
    model = YOLO(MODEL_NAME)

    # -----------------------------
    # Train
    # -----------------------------
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=8,
        name=RUN_NAME,
        pretrained=True,
        verbose=True,
    )

    print("\nTraining completed successfully")
    print("Best model saved at:")
    print(f"runs/detect/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
