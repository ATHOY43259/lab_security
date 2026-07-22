from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = Path(__file__).resolve().parent / "yolo_face" / "data.yaml"
PREV_WEIGHTS = Path(__file__).resolve().parent / "runs" / "face_finetune_continued" / "weights" / "last.pt"

EPOCHS = 30  # bigger batch this time -- box_loss/cls_loss were still
# falling after 30 total epochs (0.52 precision, 0.214 recall, 0.209
# mAP50), just slowly; worth a longer uninterrupted push now that the
# sleep-during-training issue is fixed rather than more small increments.
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
        name="face_finetune_v3",
    )


if __name__ == "__main__":
    main()
