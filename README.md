# Smart Laboratory Security System — Low-Light Face Detection & Recognition

A face detection + recognition security pipeline evaluated specifically
under low-light conditions, with a custom GAN-based enhancement model
(**TD-FALE-GAN**) whose effect on downstream accuracy is measured honestly
rather than assumed.

See [`paper_section_methodology_results.md`](paper_section_methodology_results.md)
for the full methodology and results write-up, and
[`DEMO_GUIDE.md`](DEMO_GUIDE.md) for a live demo script and viva Q&A.

## What's in this repo

```
data/               shared dataset-loading utilities (face masks, luminance splitting, etc.)
gan/                TD-FALE-GAN: face-attentive, illumination-conditioned low-light
                     enhancement GAN with a detection-guided loss
finetune/           YOLOv8 face-detector fine-tuning scripts + trained weights
ablation_study_detection.py     4-config detection ablation (YOLOv5/v8 stock vs fine-tuned vs +GAN)
ablation_study_recognition.py   HOG baseline vs ArcFace vs ArcFace+GAN recognition ablation
make_ablation_figures.py        figure generation for the detection ablation
make_recognition_figures.py     figure generation for the recognition ablation
ablation_results/   CSV results + all figures from both ablation studies
demo_live_detection.py, demo_live_recognition.py   quick scripts to see the
                     trained models run live on real data
```

Raw datasets are **not** included in this repo (see below) — everything in
`gan/checkpoints/`, `finetune/runs/*/weights/`, `yolov8n.pt`, `yolov5nu.pt`
is a trained model file small enough to check in directly.

## Datasets used (download separately)

| Dataset | Used for | Source |
|---|---|---|
| DARK FACE | GAN training, face-detector fine-tuning | Public mirror: [project page](https://flyywh.github.io/CVPRW2019LowLight/) (Google Drive links, no login required) |
| LFW (Labeled Faces in the Wild) | Face recognition evaluation | [Kaggle: stoicstatic/face-recognition-dataset](https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset) |

After downloading, place them at:
```
datasets/ieee_darkface/{image,label}/     (DARK FACE)
datasets/face_recognition/Face Data/Face Dataset/     (LFW)
```

## Setup

```bash
pip install ultralytics insightface onnxruntime opencv-python numpy pandas \
            matplotlib seaborn scikit-learn mediapipe tqdm Pillow gdown kaggle
```

## Running the pipeline

```bash
# 1. Train the enhancement GAN (uses DARK FACE)
python gan/train.py
python gan/infer_samples.py          # before/after sample images

# 2. Fine-tune the face detector (uses DARK FACE)
python finetune/prepare_yolo_face_dataset.py
python finetune/train_yolo_face.py

# 3. Run the ablation studies (uses DARK FACE + LFW + trained models above)
python ablation_study_detection.py
python ablation_study_recognition.py
python make_ablation_figures.py
python make_recognition_figures.py

# 4. Quick live demo
python demo_live_detection.py
python demo_live_recognition.py
```

## Headline results

**Detection** (600 held-out DARK FACE images, 4,922 face instances):

| Configuration | Precision | Recall | F1 | FPS (CPU) |
|---|---|---|---|---|
| YOLOv5 (stock) | 0.732 | 0.536 | 0.618 | 26.7 |
| YOLOv8 (stock) | 0.716 | 0.574 | 0.637 | 25.8 |
| YOLOv8 (fine-tuned face detector) | 0.879 | 0.251 | 0.390 | 61.9 |
| Fine-tuned + TD-FALE-GAN | 0.400 | 0.013 | 0.026 | 1.1 |

**Recognition accuracy by lighting** (610 LFW identities, 1,643 probes):

| Lighting | HOG baseline | ArcFace | ArcFace + TD-FALE-GAN |
|---|---|---|---|
| Bright | 0.272 | 0.873 | 0.755 |
| Normal | 0.201 | 0.867 | 0.883 |
| Dim | 0.205 | 0.886 | **0.911** |
| Dark | 0.280 | 0.565 | 0.526 |

The enhancement GAN's effect is condition-dependent, not universal — it
helps recognition in moderate ("dim") degradation and hurts at both
extremes, and it hurts detection on small/distant faces entirely. Full
discussion in the paper section linked above.

## License / attribution

This repository contains original code. DARK FACE and LFW are third-party
academic datasets used under their respective terms — see their project
pages for licensing details. They are not redistributed here.
