"""Quick live demo: enroll 3 real LFW identities, then show ArcFace
correctly matching a held-out probe photo of one of them. Runs in ~5 seconds."""
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parent
FACE_DIR = ROOT / "datasets" / "face_recognition" / "Face Data" / "Face Dataset"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    app = FaceAnalysis(providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    identities = [p for p in sorted(FACE_DIR.iterdir()) if p.is_dir() and len(list(p.glob("*.jpg"))) >= 4][:3]
    print(f"Enrolling {len(identities)} identities: {[p.name for p in identities]}")

    gallery = {}
    for p in identities:
        files = sorted(p.glob("*.jpg"))
        img = cv2.imread(str(files[0]))
        faces = app.get(img)
        gallery[p.name] = faces[0].embedding

    probe_identity = identities[1]
    probe_file = sorted(probe_identity.glob("*.jpg"))[2]
    print(f"\nProbe (query) photo: identity '{probe_identity.name}', file {probe_file.name}")
    probe_img = cv2.imread(str(probe_file))
    probe_emb = app.get(probe_img)[0].embedding

    print("\nSimilarity to each enrolled identity:")
    best_name, best_sim = None, -1
    for name, emb in gallery.items():
        sim = cosine(probe_emb, emb)
        marker = " <-- best match" if sim > best_sim else ""
        if sim > best_sim:
            best_sim, best_name = sim, name
        print(f"  {name}: {sim:.3f}{marker}")

    correct = "CORRECT" if best_name == probe_identity.name else "WRONG"
    print(f"\nTrue identity: {probe_identity.name} | Predicted: {best_name} | {correct}")


if __name__ == "__main__":
    main()
