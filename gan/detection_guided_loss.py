import sys
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from face_utils import crop_face_regions  # noqa: E402

PERSON_CLASS_IDX = 0  # COCO class 0 == 'person'; verified against yolo.names[0]


class DetectionGuidedLoss(nn.Module):
    """Task-driven loss: instead of only optimizing generic perceptual
    quality (exposure/color/smoothness), push the generator to raise a
    frozen, pretrained YOLOv8n's person-class confidence on the enhanced
    face region. Gradients flow through YOLOv8n's raw (pre-NMS) conv stack
    like any other frozen feature extractor -- verified empirically
    (forward pass on a face crop keeps grad_fn through to the input).

    This is what makes the enhancer's objective match what the downstream
    security pipeline actually needs (a detectable/recognizable face), not
    just what looks brighter to a human.
    """

    def __init__(self, weights_path: str, device="cpu"):
        super().__init__()
        yolo = YOLO(weights_path)
        assert yolo.names[PERSON_CLASS_IDX] == "person"
        self.detector = yolo.model.to(device)
        self.detector.eval()
        for p in self.detector.parameters():
            p.requires_grad_(False)

    def forward(self, enhanced, mask):
        face_crop = crop_face_regions(enhanced, mask, out_size=128)
        _, decoded = self.detector(face_crop)
        person_scores = decoded["scores"][:, PERSON_CLASS_IDX, :]  # [B, n_anchors] raw logits
        best_per_sample = person_scores.max(dim=1).values
        return (1.0 - torch.sigmoid(best_per_sample)).mean()
