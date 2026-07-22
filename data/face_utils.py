import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def crop_face_regions(img_batch, mask_batch, out_size=96):
    """Crop+resize the face-mask bounding region per sample. Shared by the
    face-patch discriminator and the detection-guided loss so both score
    exactly the same region of the enhanced image.
    """
    b = img_batch.shape[0]
    crops = []
    for i in range(b):
        m = mask_batch[i, 0]
        ys, xs = torch.where(m > 0.5)
        if len(ys) == 0:
            crops.append(F.interpolate(img_batch[i:i + 1], size=(out_size, out_size),
                                        mode="bilinear", align_corners=False))
            continue
        y1, y2 = ys.min().item(), ys.max().item() + 1
        x1, x2 = xs.min().item(), xs.max().item() + 1
        crop = img_batch[i:i + 1, :, y1:y2, x1:x2]
        crops.append(F.interpolate(crop, size=(out_size, out_size), mode="bilinear", align_corners=False))
    return torch.cat(crops, dim=0)


def illumination_map(img_batch):
    """Per-sample mean luminance, broadcast back to a full spatial map, so
    the generator can be conditioned on how dark the *whole frame* is (not
    just told which pixels are faces) and learn a lighting-adaptive
    correction strength instead of one fixed curve for every brightness
    level.
    """
    lum = img_batch.mean(dim=[1, 2, 3], keepdim=True)
    return lum.expand(-1, 1, img_batch.shape[2], img_batch.shape[3])


def face_attention_mask(shape_hw, boxes, scale_x, scale_y):
    """Build a soft attention mask from face boxes, shared by every dataset
    loader (DARK FACE, Kaggle, FaceEngine, lab_captured) so the GAN and any
    future face-aware model see the same mask convention regardless of
    source dataset.
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        x1, x2 = sorted((int(x1 * scale_x), int(x2 * scale_x)))
        y1, y2 = sorted((int(y1 * scale_y), int(y2 * scale_y)))
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    if mask.max() > 0:
        # Feather the hard box edges so the attention gate blends smoothly
        # into the background instead of producing visible seams.
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask / (mask.max() + 1e-6)
    return mask


def split_domains_by_luminance(image_dir: Path, sample_limit: int, bright_frac: float, dark_frac: float,
                                pattern: str = "*.png", seed: int = 0):
    """Carve a directory of images into a 'dark' and a pseudo-'bright' domain
    by mean luminance, for unpaired adversarial training when no ground-truth
    normal-light counterpart exists. Reusable across any single-domain
    low-light dataset, not just DARK FACE.
    """
    files = sorted(image_dir.glob(pattern))
    rng = random.Random(seed)
    rng.shuffle(files)
    files = files[:sample_limit]

    scored = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        scored.append((float(img.mean()), f))
    scored.sort(key=lambda t: t[0])

    n = len(scored)
    n_dark = max(1, int(n * dark_frac))
    n_bright = max(1, int(n * bright_frac))
    dark_files = [f for _, f in scored[:n_dark]]
    bright_files = [f for _, f in scored[-n_bright:]]
    return dark_files, bright_files
