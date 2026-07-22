import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from face_utils import crop_face_regions  # noqa: E402

N_ITERS = 4  # curve-adjustment iterations. Zero-DCE's usual 8 compounds
# per-channel curve noise 8x in near-zero (true black) pixels, which DARK
# FACE has plenty of (real night sky, not just dim indoor light) -- that
# showed up as magenta/green chroma speckle in the first training run.


class FALEGenerator(nn.Module):
    """Face-Attentive, Illumination-Conditioned Low-light Enhancement
    generator (the "IC" + "FA" halves of TD-FALE-GAN).

    Standard zero-reference curve estimation (Zero-DCE) predicts one fixed
    set of per-pixel curve maps regardless of how dark the frame actually
    is, and treats a dim background wall the same as a dim face. Two
    conditioning signals are concatenated onto the input here to fix both:
    a face-attention mask (from ground-truth boxes at train time, from the
    detector's own boxes at inference) so correction concentrates on faces,
    and a per-frame illumination map so one network learns a *lighting-
    adaptive* correction strength instead of a single fixed curve family
    that has to compromise between bright and pitch-dark inputs.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.e1 = nn.Conv2d(5, c, 3, 1, 1)
        self.e2 = nn.Conv2d(c, c, 3, 1, 1)
        self.e3 = nn.Conv2d(c, c, 3, 1, 1)
        self.e4 = nn.Conv2d(c, c, 3, 1, 1)
        self.d3 = nn.Conv2d(c * 2, c, 3, 1, 1)
        self.d2 = nn.Conv2d(c * 2, c, 3, 1, 1)
        self.d1 = nn.Conv2d(c * 2, N_ITERS * 3, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

        self.attn_gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, face_mask, illum_map):
        inp = torch.cat([x, face_mask, illum_map], dim=1)
        x1 = self.act(self.e1(inp))
        x2 = self.act(self.e2(x1))
        x3 = self.act(self.e3(x2))
        x4 = self.act(self.e4(x3))
        y3 = self.act(self.d3(torch.cat([x3, x4], dim=1)))
        y2 = self.act(self.d2(torch.cat([x2, y3], dim=1)))
        curves = torch.tanh(self.d1(torch.cat([x1, y2], dim=1)))

        gate = 1.0 + self.attn_gain * face_mask  # >=1x boost inside faces
        curves = curves * gate

        enhanced = x
        curve_maps = torch.split(curves, 3, dim=1)
        for a in curve_maps:
            enhanced = enhanced + a * (torch.pow(enhanced, 2) - enhanced)
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        return enhanced, curves


def _patch_discriminator_body(in_ch: int, c: int = 32):
    return nn.Sequential(
        nn.Conv2d(in_ch, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c, c * 2, 4, 2, 1), nn.InstanceNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.InstanceNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c * 4, 1, 4, 1, 1),
    )


class DualDiscriminator(nn.Module):
    """Global PatchGAN + a second PatchGAN restricted to face-patch crops.

    A plain global discriminator can be satisfied by a realistic-looking
    background while the (small, security-critical) face region stays
    degraded. Scoring cropped face patches with their own discriminator
    gives the generator a gradient that is specifically about face realism,
    which is the quantity the downstream ArcFace matcher actually cares
    about.
    """

    def __init__(self):
        super().__init__()
        self.global_d = _patch_discriminator_body(3)
        self.face_d = _patch_discriminator_body(3)

    def forward(self, img, mask):
        global_out = self.global_d(img)
        face_crop = crop_face_regions(img, mask)
        face_out = self.face_d(face_crop)
        return global_out, face_out
