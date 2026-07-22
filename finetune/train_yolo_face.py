from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = Path(__file__).resolve().parent / "yolo_face" / "data.yaml"

EPOCHS = 12
IMG_SIZE = 320  # smaller than YOLOv8's usual 640 -- trades some accuracy on
# very small/far faces for CPU-feasible training time (~7.6s/image at this
# size, measured on this machine before committing to a full run).
BATCH = 8


def main():
    model = YOLO(str(ROOT / "yolov8n.pt"))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",
        workers=0,
        project=str(Path(__file__).resolve().parent / "runs"),
        name="face_finetune",
    )


if __name__ == "__main__":
    main()
