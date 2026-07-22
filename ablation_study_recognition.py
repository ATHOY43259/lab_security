"""
Recognition ablation study (Phase 2): real HOG-descriptor baseline vs real
ArcFace (InsightFace) embeddings, on 610 real LFW identities (>=4 photos
each), tested under genuine pixel-level lighting degradation.

Unlike the original ablation_study_windows.py, ArcFace gets NO hardcoded
"darkness_penalty" advantage -- both methods run the identical real
matching pipeline on the identical degraded image. Any robustness gap has
to emerge from measurement, not be assumed.

Protocol (standard biometric verification, not identification):
  - Enrollment: 2 images/identity -> gallery embedding (averaged).
  - Genuine trial: probe vs its OWN identity's gallery embedding.
    Accept if similarity >= THRESHOLD -> should always accept (label=1).
  - Impostor trial: probe vs its single most-similar OTHER identity's
    gallery embedding (hardest impostor case). Should always reject
    (label=0).
  - FRR = fraction of genuine trials incorrectly rejected.
  - FAR = fraction of impostor trials incorrectly accepted.
  - Precision/Recall/F1/Accuracy computed over the combined 2N trials.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "gan"))
from face_utils import face_attention_mask, illumination_map  # noqa: E402
from model import FALEGenerator  # noqa: E402

FACE_DIR = ROOT / "datasets" / "face_recognition" / "Face Data" / "Face Dataset"
RESULTS_DIR = ROOT / "ablation_results"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_IMAGES = 4
MAX_PROBES_PER_ID = 3
LIGHTING_LEVELS = ["bright", "normal", "dim", "dark"]
N_CALIBRATION_IDS = 100  # held-out-in-spirit subset used only to pick each
# method's own decision threshold, since HOG's all-nonnegative histograms
# and ArcFace's embeddings have completely different natural cosine-
# similarity ranges -- one shared threshold (e.g. 0.5) silently breaks HOG
# (verified: HOG cosine sim between *different* people sat at 0.59-0.74,
# meaning a 0.5 threshold accepts every impostor).


def calibrate_threshold(embed_fn, identities):
    """Pick the threshold that minimizes |FAR - FRR| (near-EER) for this
    specific method, under normal lighting, using its own genuine/impostor
    score distributions -- not a value borrowed from another method.
    """
    calib_ids = identities[:N_CALIBRATION_IDS]
    gallery = {}
    for id_name, gallery_files, _ in calib_ids:
        embs = [embed_fn(cv2.imread(str(f))) for f in gallery_files]
        embs = [e for e in embs if e is not None]
        if embs:
            gallery[id_name] = np.mean(embs, axis=0)

    names = list(gallery.keys())
    mat = np.stack([gallery[n] for n in names])

    genuine_scores, impostor_scores = [], []
    for id_name, _, probe_files in calib_ids:
        if id_name not in gallery:
            continue
        own_idx = names.index(id_name)
        for f in probe_files:
            emb = embed_fn(cv2.imread(str(f)))
            if emb is None:
                continue
            sims = np.array([cosine(emb, mat[i]) for i in range(len(names))])
            genuine_scores.append(sims[own_idx])
            impostor_scores.append(np.delete(sims, own_idx).max())

    genuine_scores, impostor_scores = np.array(genuine_scores), np.array(impostor_scores)
    best_t, best_gap = 0.5, float("inf")
    for t in np.arange(-0.2, 1.001, 0.01):
        far = (impostor_scores >= t).mean()
        frr = (genuine_scores < t).mean()
        if abs(far - frr) < best_gap:
            best_gap, best_t = abs(far - frr), t
    return float(best_t)


def apply_lighting(frame, lighting):
    """Verbatim from the original ablation_study_windows.py -- a real pixel
    transform (contrast+brightness), not a score fudge factor."""
    if lighting == "bright":
        return cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
    elif lighting == "normal":
        return frame.copy()
    elif lighting == "dim":
        return cv2.convertScaleAbs(frame, alpha=0.6, beta=-30)
    elif lighting == "dark":
        return cv2.convertScaleAbs(frame, alpha=0.25, beta=-80)
    return frame.copy()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def hog_embedding(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(resized).flatten()


def build_identity_list():
    ids = []
    for p in sorted(FACE_DIR.iterdir()):
        if not p.is_dir():
            continue
        files = sorted(p.glob("*.jpg"))
        if len(files) >= MIN_IMAGES:
            ids.append((p.name, files[:2], files[2:2 + MAX_PROBES_PER_ID]))
    return ids


def load_gan():
    G = FALEGenerator()
    G.load_state_dict(torch.load(ROOT / "gan" / "checkpoints" / "fale_generator.pt", map_location="cpu"))
    G.eval()
    return G


def enhance_with_gan(G, img_bgr):
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = face_attention_mask((h0, w0), [], 1.0, 1.0)  # no box prior for LFW crops
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    illum_t = illumination_map(img_t)
    with torch.no_grad():
        enhanced, _ = G(img_t, mask_t, illum_t)
    enh = (enhanced[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(enh, cv2.COLOR_RGB2BGR)


def evaluate(name, embed_fn, identities, lighting, threshold, gan_model=None):
    gallery = {}  # id_name -> embedding
    for id_name, gallery_files, _ in identities:
        embs = []
        for f in gallery_files:
            img = cv2.imread(str(f))
            e = embed_fn(img)
            if e is not None:
                embs.append(e)
        if embs:
            gallery[id_name] = np.mean(embs, axis=0)

    gallery_names = list(gallery.keys())
    gallery_mat = np.stack([gallery[n] for n in gallery_names])

    tp = fp = fn = tn = 0
    latencies = []
    for id_name, _, probe_files in identities:
        if id_name not in gallery:
            continue
        for f in probe_files:
            img = cv2.imread(str(f))
            degraded = apply_lighting(img, lighting)
            if gan_model is not None:
                degraded = enhance_with_gan(gan_model, degraded)

            t0 = time.time()
            probe_emb = embed_fn(degraded)
            latencies.append(time.time() - t0)
            if probe_emb is None:
                # No face found -> genuine trial fails (fn); impostor trial
                # is trivially correctly rejected too (nothing was accepted),
                # so tn, to keep both trial counts balanced per probe.
                fn += 1
                tn += 1
                continue

            sims = np.array([cosine(probe_emb, gallery_mat[i]) for i in range(len(gallery_names))])
            own_idx = gallery_names.index(id_name)
            genuine_sim = sims[own_idx]
            impostor_sims = np.delete(sims, own_idx)
            best_impostor_sim = impostor_sims.max() if len(impostor_sims) else -1.0

            # genuine trial
            if genuine_sim >= threshold:
                tp += 1
            else:
                fn += 1
            # impostor trial (hardest other identity)
            if best_impostor_sim >= threshold:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    avg_latency_ms = float(np.mean(latencies) * 1000) if latencies else 0.0

    return {
        "config": name, "lighting": lighting, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "far": far, "frr": frr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "avg_latency_ms": avg_latency_ms,
    }


def main():
    identities = build_identity_list()
    print(f"Using {len(identities)} identities with >={MIN_IMAGES} images each "
          f"({sum(len(p) for _, _, p in identities)} total probe images)")

    gan_model = load_gan()

    from insightface.app import FaceAnalysis
    arcface_app = FaceAnalysis(providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    arcface_app.prepare(ctx_id=0, det_size=(320, 320))

    def arcface_embed(img):
        faces = arcface_app.get(img)
        return faces[0].embedding if faces else None

    def hog_embed(img):
        return hog_embedding(img)

    print("Calibrating per-method thresholds (near-EER, on first "
          f"{N_CALIBRATION_IDS} identities, normal lighting)...")
    hog_threshold = calibrate_threshold(hog_embed, identities)
    arcface_threshold = calibrate_threshold(arcface_embed, identities)
    print(f"HOG threshold={hog_threshold:.3f}  ArcFace threshold={arcface_threshold:.3f}")

    rows = []
    for lighting in LIGHTING_LEVELS:
        print(f"\n=== lighting: {lighting} ===")
        row = evaluate("Baseline (HOG + Euclidean-cosine)", hog_embed, identities, lighting, hog_threshold)
        print("Baseline:", {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k not in ("config", "lighting")})
        rows.append(row)

        row = evaluate("ArcFace (InsightFace)", arcface_embed, identities, lighting, arcface_threshold)
        print("ArcFace: ", {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k not in ("config", "lighting")})
        rows.append(row)

        row = evaluate("ArcFace + TD-FALE-GAN", arcface_embed, identities, lighting, arcface_threshold, gan_model=gan_model)
        print("ArcFace+GAN:", {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k not in ("config", "lighting")})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ablation_results_recognition.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'ablation_results_recognition.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
