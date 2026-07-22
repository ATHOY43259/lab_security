import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from darkface_dataset import DarkFaceDomainDataset  # noqa: E402
from face_utils import illumination_map, split_domains_by_luminance  # noqa: E402
from detection_guided_loss import DetectionGuidedLoss
from losses import FALEGANLoss
from model import DualDiscriminator, FALEGenerator

IMAGE_DIR = ROOT / "datasets" / "ieee_darkface" / "image"
LABEL_DIR = ROOT / "datasets" / "ieee_darkface" / "label"
YOLO_WEIGHTS = ROOT / "yolov8n.pt"
OUT_DIR = ROOT / "gan" / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 20
SAMPLE_LIMIT = 900
LR_G = 2e-4
LR_D = 5e-5  # first run's D collapsed (d_loss -> 0.01 by epoch 6) and stopped
# giving the generator a useful adversarial gradient for the back half of
# training; a slower D keeps the contest going longer.


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FALE-GAN] device={device}")

    dark_files, bright_files = split_domains_by_luminance(
        IMAGE_DIR, sample_limit=SAMPLE_LIMIT, bright_frac=0.35, dark_frac=0.45
    )
    print(f"[FALE-GAN] dark={len(dark_files)} bright={len(bright_files)}")

    ds = DarkFaceDomainDataset(dark_files, bright_files, LABEL_DIR, img_size=IMG_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    G = FALEGenerator().to(device)
    D = DualDiscriminator().to(device)
    det_loss_fn = DetectionGuidedLoss(str(YOLO_WEIGHTS), device=device)
    criterion = FALEGANLoss()

    opt_g = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

    history = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        g_losses, d_losses = [], []
        for dark, mask, bright in dl:
            dark, mask, bright = dark.to(device), mask.to(device), bright.to(device)

            illum = illumination_map(dark)
            enhanced, curves = G(dark, mask, illum)

            # --- Discriminator step ---
            opt_d.zero_grad()
            d_real_g, d_real_f = D(bright, torch.zeros_like(mask))
            d_fake_g, d_fake_f = D(enhanced.detach(), mask)
            d_loss = criterion.discriminator_loss(d_real_g, d_fake_g) + \
                criterion.discriminator_loss(d_real_f, d_fake_f)
            d_loss.backward()
            opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad()
            d_fake_g2, d_fake_f2 = D(enhanced, mask)
            det_loss = det_loss_fn(enhanced, mask)
            g_loss, parts = criterion.generator_loss(enhanced, dark, curves, d_fake_g2, d_fake_f2, det_loss)
            g_loss.backward()
            opt_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        avg_g, avg_d = sum(g_losses) / len(g_losses), sum(d_losses) / len(d_losses)
        elapsed = time.time() - t0
        print(f"[FALE-GAN] epoch {epoch}/{EPOCHS} g_loss={avg_g:.4f} d_loss={avg_d:.4f} elapsed={elapsed:.0f}s")
        history.append({"epoch": epoch, "g_loss": avg_g, "d_loss": avg_d, "elapsed_s": elapsed})

    torch.save(G.state_dict(), OUT_DIR / "fale_generator.pt")
    torch.save(D.state_dict(), OUT_DIR / "fale_discriminator.pt")
    (OUT_DIR / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"[FALE-GAN] done. checkpoints in {OUT_DIR}")


if __name__ == "__main__":
    main()
