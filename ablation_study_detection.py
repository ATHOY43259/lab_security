"""
Detection-only ablation study (Phase 1).

Compares 4 real, working configurations for the *face detection* stage of
the pipeline, measured on 600 real held-out DARK FACE validation images
(never used for a gradient update in any training run in this project).

Recognition metrics (FAR/FRR, the full 5-config table from the original
proposal) are NOT included here -- they need multiple photos per identity,
and none of the identity-labeled sources (Kaggle face_recognition, IEEE
FaceEngine, or your own captured lab photos) have arrived yet. This script
only claims what it can actually measure right now.

Evaluation protocol (documented because object detectors output different
box classes across configs, so a naive IoU-vs-ground-truth-face-box
comparison would be unfair to the stock person-detectors):
  - A ground-truth face counts as DETECTED if its box center falls inside
    any predicted box (class 'face' for the fine-tuned model, class
    'person' for the stock COCO models) with confidence >= CONF_THRESH.
  - Precision is computed over predicted boxes: a predicted box counts as
    a TP if it contains at least one ground-truth face center.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data"))
from darkface_dataset import load_darkface_boxes  # noqa: E402
from face_utils import face_attention_mask, illumination_map  # noqa: E402

sys.path.insert(0, str(ROOT / "gan"))
from model import FALEGenerator  # noqa: E402

VAL_IMAGE_DIR = ROOT / "finetune" / "yolo_face" / "images" / "val"
VAL_LABEL_DIR = ROOT / "datasets" / "ieee_darkface" / "label"
RESULTS_DIR = ROOT / "ablation_results"
RESULTS_DIR.mkdir(exist_ok=True)

CONF_THRESH = 0.25
GAN_IMG_SIZE = 128

CONFIGS = {
    "Config A: YOLOv5 baseline (stock, person-class)": {
        "weights": "yolov5nu.pt", "target_class": "person", "gan": False,
    },
    "Config B: YOLOv8 baseline (stock, person-class)": {
        "weights": "yolov8n.pt", "target_class": "person", "gan": False,
    },
    "Config C: YOLOv8 fine-tuned face detector": {
        "weights": "finetune/runs/face_finetune_v3/weights/best.pt", "target_class": "face", "gan": False,
    },
    "Config D (proposed): fine-tuned face detector + TD-FALE-GAN": {
        "weights": "finetune/runs/face_finetune_v3/weights/best.pt", "target_class": "face", "gan": True,
    },
}


def load_gan():
    G = FALEGenerator()
    G.load_state_dict(torch.load(ROOT / "gan" / "checkpoints" / "fale_generator.pt", map_location="cpu"))
    G.eval()
    return G


CANDIDATE_CONF = 0.05  # low bar on purpose: at this stage we only need rough
# face locations to build the attention mask, not confident final detections


def enhance_with_gan(G, img_bgr, detector, target_class_idx):
    """Runs the generator at the image's *native* resolution. FALEGenerator
    is fully convolutional (no fixed-size layers), so it doesn't need the
    128px training resolution at inference -- and it must not get it here,
    since DARK FACE faces are tiny and a downsample/upsample round-trip
    blurs away exactly the detail a detector depends on (verified: this
    silently destroyed detection recall before this fix).
    """
    h0, w0 = img_bgr.shape[:2]

    # Stage 1: get rough candidate face locations from the *raw* dark image
    # -- the GAN was trained with a real face mask on every sample, so an
    # all-zero mask at inference is out-of-distribution input.
    candidates = detector.predict(img_bgr, conf=CANDIDATE_CONF, classes=[target_class_idx], verbose=False)[0]
    boxes = candidates.boxes.xyxy.cpu().numpy() if len(candidates.boxes) else np.zeros((0, 4))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = face_attention_mask((h0, w0), boxes.tolist(), 1.0, 1.0)

    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    illum_t = illumination_map(img_t)
    with torch.no_grad():
        enhanced, _ = G(img_t, mask_t, illum_t)
    enh = (enhanced[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(enh, cv2.COLOR_RGB2BGR)


def luminance_bucket(img_bgr):
    mean_lum = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean()
    if mean_lum >= 60:
        return "bright"
    elif mean_lum >= 30:
        return "normal"
    elif mean_lum >= 15:
        return "dim"
    return "dark"


def evaluate_config(name, cfg, val_files, gan_model):
    model = YOLO(str(ROOT / cfg["weights"]))
    class_idx = [k for k, v in model.names.items() if v == cfg["target_class"]][0]

    tp, fp, fn = 0, 0, 0
    bucket_stats = {"bright": [0, 0], "normal": [0, 0], "dim": [0, 0], "dark": [0, 0]}  # [detected, total]
    latencies = []

    for img_path in val_files:
        img_bgr = cv2.imread(str(img_path))
        bucket = luminance_bucket(img_bgr)
        gt_boxes = load_darkface_boxes(VAL_LABEL_DIR / f"{img_path.stem}.txt")

        t0 = time.time()
        infer_img = enhance_with_gan(gan_model, img_bgr, model, class_idx) if cfg["gan"] else img_bgr
        result = model.predict(infer_img, conf=CONF_THRESH, classes=[class_idx], verbose=False)[0]
        latencies.append(time.time() - t0)  # full pipeline time -- GAN preprocessing (when enabled)
        # plus detection, not detection alone, since that's what a real deployment pays per frame

        pred_boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))

        matched_pred = set()
        for (x1, y1, x2, y2) in gt_boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hit = False
            for i, (px1, py1, px2, py2) in enumerate(pred_boxes):
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    hit = True
                    matched_pred.add(i)
            bucket_stats[bucket][1] += 1
            if hit:
                tp += 1
                bucket_stats[bucket][0] += 1
            else:
                fn += 1
        fp += len(pred_boxes) - len(matched_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency_ms = float(np.mean(latencies) * 1000)
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    lighting_recall = {b: (d / t if t > 0 else None) for b, (d, t) in bucket_stats.items()}

    return {
        "config": name, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "avg_latency_ms": avg_latency_ms, "fps": fps,
        **{f"recall_{b}": v for b, v in lighting_recall.items()},
    }


def main():
    val_files = sorted(VAL_IMAGE_DIR.glob("*.png"))
    print(f"Evaluating on {len(val_files)} held-out DARK FACE validation images")

    gan_model = load_gan()

    rows = []
    for name, cfg in CONFIGS.items():
        print(f"\n--- {name} ---")
        row = evaluate_config(name, cfg, val_files, gan_model)
        print({k: v for k, v in row.items() if k != "config"})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ablation_results_detection.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'ablation_results_detection.csv'}")
    print(df[["config", "precision", "recall", "f1", "fps", "avg_latency_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
