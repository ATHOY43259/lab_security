import torch
import torch.nn as nn
import torch.nn.functional as F


def color_constancy_loss(img):
    mean_rgb = img.mean(dim=[2, 3])
    r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
    return ((r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2).mean()


def exposure_loss(img, patch_size=16, target=0.6):
    pooled = F.avg_pool2d(img.mean(dim=1, keepdim=True), patch_size)
    return ((pooled - target) ** 2).mean()


def illumination_smoothness_loss(curves):
    dh = curves[:, :, 1:, :] - curves[:, :, :-1, :]
    dw = curves[:, :, :, 1:] - curves[:, :, :, :-1]
    return (dh ** 2).mean() + (dw ** 2).mean()


def spatial_consistency_loss(enhanced, original, patch_size=4):
    kernels = {
        "left": torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32),
        "right": torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32),
        "up": torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32),
        "down": torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32),
    }
    device = enhanced.device
    e_pool = F.avg_pool2d(enhanced.mean(dim=1, keepdim=True), patch_size)
    o_pool = F.avg_pool2d(original.mean(dim=1, keepdim=True), patch_size)
    loss = 0.0
    for k in kernels.values():
        k = k.view(1, 1, 3, 3).to(device)
        d_e = F.conv2d(e_pool, k, padding=1)
        d_o = F.conv2d(o_pool, k, padding=1)
        loss = loss + ((d_e - d_o) ** 2).mean()
    return loss


def chroma_smoothness_loss(enhanced):
    """Direct TV loss on the enhanced RGB output (not just the curve maps).
    The per-channel curve amplification in near-black pixels produces
    high-frequency chroma speckle that illumination_smoothness_loss (which
    only regularizes the curve maps) does not fully suppress -- this
    penalizes it where it actually shows up, in the output pixels.
    """
    dh = enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :]
    dw = enhanced[:, :, :, 1:] - enhanced[:, :, :, :-1]
    return dh.abs().mean() + dw.abs().mean()


def lsgan_loss(pred, is_real: bool):
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


class FALEGANLoss(nn.Module):
    def __init__(self, w_adv=1.0, w_spa=1.0, w_exp=8.0, w_col=5.0, w_tv=200.0, w_chroma=15.0, w_det=2.0):
        super().__init__()
        self.w_adv, self.w_spa, self.w_exp, self.w_col, self.w_tv, self.w_chroma, self.w_det = (
            w_adv, w_spa, w_exp, w_col, w_tv, w_chroma, w_det
        )

    def generator_loss(self, enhanced, original, curves, d_global_fake, d_face_fake, det_loss):
        l_adv = lsgan_loss(d_global_fake, True) + lsgan_loss(d_face_fake, True)
        l_spa = spatial_consistency_loss(enhanced, original)
        l_exp = exposure_loss(enhanced)
        l_col = color_constancy_loss(enhanced)
        l_tv = illumination_smoothness_loss(curves)
        l_chroma = chroma_smoothness_loss(enhanced)
        total = (self.w_adv * l_adv + self.w_spa * l_spa + self.w_exp * l_exp
                 + self.w_col * l_col + self.w_tv * l_tv + self.w_chroma * l_chroma
                 + self.w_det * det_loss)
        parts = {"adv": l_adv.item(), "spa": l_spa.item(), "exp": l_exp.item(),
                 "col": l_col.item(), "tv": l_tv.item(), "chroma": l_chroma.item(),
                 "det": det_loss.item()}
        return total, parts

    @staticmethod
    def discriminator_loss(d_real, d_fake):
        return 0.5 * (lsgan_loss(d_real, True) + lsgan_loss(d_fake, False))
