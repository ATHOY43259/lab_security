from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = Path(__file__).resolve().parent / "yolo_face" / "data.yaml"
PREV_WEIGHTS = Path(__file__).resolve().parent / "runs" / "face_finetune" / "weights" / "last.pt"

EPOCHS = 18  # additional epochs on top of the first 12 -- metrics were
# still climbing every epoch at the end of that run (precision 0.32->0.48,
# recall 0.09->0.19, mAP50 0.07->0.18), so this continues from where it
# left off rather than restarting.
IMG_SIZE = 320
BATCH = 8


def main():
    model = YOLO(str(PREV_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",
        workers=0,
        project=str(Path(__file__).resolve().parent / "runs"),
        name="face_finetune_continued",
    )


if __name__ == "__main__":
    main()
