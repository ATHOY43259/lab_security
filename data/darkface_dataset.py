from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from face_utils import face_attention_mask


def load_darkface_boxes(label_path: Path):
    """DARK FACE label format: first line = face count, then one 'x1 y1 x2 y2' per line."""
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return []
    n = int(lines[0])
    boxes = []
    for line in lines[1:1 + n]:
        parts = line.split()
        if len(parts) != 4:
            continue
        x1, y1, x2, y2 = map(int, parts)
        boxes.append((x1, y1, x2, y2))
    return boxes


class DarkFaceDomainDataset(Dataset):
    """Yields (dark_image, face_attention_mask, bright_image) triplets for
    unpaired GAN training. Bright/dark are index-independent domains, so we
    just cycle the shorter list.
    """

    def __init__(self, dark_files, bright_files, label_dir: Path, img_size: int = 256):
        self.dark_files = dark_files
        self.bright_files = bright_files
        self.label_dir = label_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.dark_files)

    def _load_rgb(self, path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img, h0, w0

    def __getitem__(self, idx):
        dark_path = self.dark_files[idx]
        bright_path = self.bright_files[idx % len(self.bright_files)]

        dark_img, h0, w0 = self._load_rgb(dark_path)
        bright_img, _, _ = self._load_rgb(bright_path)

        boxes = load_darkface_boxes(self.label_dir / f"{dark_path.stem}.txt")
        scale_x = self.img_size / w0
        scale_y = self.img_size / h0
        mask = face_attention_mask((self.img_size, self.img_size), boxes, scale_x, scale_y)

        dark_t = torch.from_numpy(dark_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        bright_t = torch.from_numpy(bright_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return dark_t, mask_t, bright_t
