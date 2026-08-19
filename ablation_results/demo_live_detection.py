"""Quick live demo: fine-tuned face detector on a real dark image, draws
boxes, saves + prints results. Runs in ~2 seconds."""
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "finetune" / "runs" / "face_finetune_v3" / "weights" / "best.pt"
IMAGE = ROOT / "finetune" / "yolo_face" / "images" / "val"
OUT = ROOT / "demo_output"
OUT.mkdir(exist_ok=True)


def main():
    model = YOLO(str(MODEL))
    img_path = sorted(IMAGE.glob("*.png"))[0]
    print(f"Running fine-tuned face detector on: {img_path.name}")

    result = model.predict(str(img_path), conf=0.25, classes=[0], verbose=False)[0]
    print(f"Faces detected: {len(result.boxes)}")

    annotated = result.plot()
    out_path = OUT / "live_detection_demo.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated result to: {out_path}")


if __name__ == "__main__":
    main()
