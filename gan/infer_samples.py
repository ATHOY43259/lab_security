import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from darkface_dataset import load_darkface_boxes  # noqa: E402
from face_utils import face_attention_mask, illumination_map  # noqa: E402
from model import FALEGenerator
IMAGE_DIR = ROOT / "datasets" / "ieee_darkface" / "image"
LABEL_DIR = ROOT / "datasets" / "ieee_darkface" / "label"
CKPT = ROOT / "gan" / "checkpoints" / "fale_generator.pt"
OUT_DIR = ROOT / "gan" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
SAMPLE_IDS = [1, 100, 1006, 2500, 4001]  # spread across the darkest/hardest test cases


def load_input(img_id: int):
    img_path = IMAGE_DIR / f"{img_id}.png"
    label_path = LABEL_DIR / f"{img_id}.txt"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    boxes = load_darkface_boxes(label_path)
    mask = face_attention_mask((IMG_SIZE, IMG_SIZE), boxes, IMG_SIZE / w0, IMG_SIZE / h0)

    img_t = torch.from_numpy(img_r.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    return img_t, mask_t, img_r, len(boxes)


def main():
    G = FALEGenerator()
    G.load_state_dict(torch.load(CKPT, map_location="cpu"))
    G.eval()

    for img_id in SAMPLE_IDS:
        img_t, mask_t, orig_rgb, n_faces = load_input(img_id)
        with torch.no_grad():
            illum_t = illumination_map(img_t)
            enhanced, _ = G(img_t, mask_t, illum_t)
        enh_rgb = (enhanced[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        side_by_side = np.concatenate([orig_rgb, enh_rgb], axis=1)
        side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        out_path = OUT_DIR / f"compare_{img_id}.png"
        cv2.imwrite(str(out_path), side_by_side_bgr)

        orig_mean = orig_rgb.mean()
        enh_mean = enh_rgb.mean()
        print(f"id={img_id} faces={n_faces} mean_lum orig={orig_mean:.1f} enhanced={enh_mean:.1f} -> {out_path.name}")


if __name__ == "__main__":
    main()
