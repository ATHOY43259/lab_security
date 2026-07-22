import random
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from darkface_dataset import load_darkface_boxes  # noqa: E402

SRC_IMAGE_DIR = ROOT / "datasets" / "ieee_darkface" / "image"
SRC_LABEL_DIR = ROOT / "datasets" / "ieee_darkface" / "label"
OUT_DIR = Path(__file__).resolve().parent / "yolo_face"
VAL_FRACTION = 0.1
SEED = 0


def convert_and_link(img_path: Path, split: str):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    boxes = load_darkface_boxes(SRC_LABEL_DIR / f"{img_path.stem}.txt")

    lines = []
    for (x1, y1, x2, y2) in boxes:
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    img_out = OUT_DIR / "images" / split / img_path.name
    label_out = OUT_DIR / "labels" / split / f"{img_path.stem}.txt"
    if not img_out.exists():
        try:
            img_out.hardlink_to(img_path)
        except OSError:
            img_out.write_bytes(img_path.read_bytes())
    label_out.write_text("\n".join(lines))
    return len(lines)


def main():
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    files = sorted(SRC_IMAGE_DIR.glob("*.png"))
    rng = random.Random(SEED)
    rng.shuffle(files)
    n_val = int(len(files) * VAL_FRACTION)
    val_files, train_files = files[:n_val], files[n_val:]

    total_boxes = {"train": 0, "val": 0}
    for f in train_files:
        total_boxes["train"] += convert_and_link(f, "train")
    for f in val_files:
        total_boxes["val"] += convert_and_link(f, "val")

    print(f"train images={len(train_files)} boxes={total_boxes['train']}")
    print(f"val images={len(val_files)} boxes={total_boxes['val']}")

    data_yaml = {
        "path": str(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "face"},
    }
    (OUT_DIR / "data.yaml").write_text(yaml.dump(data_yaml))
    print(f"wrote {OUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
