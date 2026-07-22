# Methodology and Experimental Results

## 1. Datasets

Two public, real-world datasets were used, selected to match the two evaluation
axes of the system (detection and recognition):

- **DARK FACE** (6,000 real nighttime low-light images, annotated with face
  bounding boxes) — used for training the low-light enhancement model and
  fine-tuning the face detector. A held-out split of 600 images (never used
  for a gradient update in any training run) was reserved for evaluation.
- **Labeled Faces in the Wild (LFW)**, 1,680 identities with 2-50 images each
  — used for the face recognition evaluation. 610 identities with at least 4
  images were used, with 2 images per identity enrolled as a gallery
  (reference) and up to 3 held out as probes (queries).

## 2. Proposed Architecture: TD-FALE-GAN

A custom low-light enhancement GAN was designed with three components,
motivated by the specific requirements of a face-recognition security
pipeline rather than generic image quality:

1. **Face-attentive curve estimation.** Rather than predicting one uniform
   brightening curve for an entire frame (as in standard zero-reference
   methods), the generator is additionally conditioned on a face-location
   attention mask, allowing it to apply stronger correction specifically in
   face regions.
2. **Illumination conditioning.** The generator also receives a per-frame
   brightness estimate, allowing a single network to adapt its correction
   strength across a range of lighting severities instead of applying one
   fixed transformation.
3. **Detection-guided loss.** In addition to standard zero-reference losses
   (exposure, color constancy, spatial consistency, illumination smoothness)
   and an adversarial loss against a dual (global + face-patch) discriminator,
   the generator is penalized when a frozen, pretrained YOLOv8 detector
   reports low person-confidence on the enhanced output — directly
   optimizing for downstream detectability rather than perceptual quality
   alone.

The generator is fully convolutional and was trained at 128x128 resolution
for computational tractability, then run at native image resolution at
inference time.

## 3. Face Detector Fine-Tuning

The stock YOLOv8n model (pretrained on COCO, class "person") was fine-tuned
into a dedicated single-class face detector using DARK FACE's 45,474
annotated face boxes across 5,400 training images (600 held out for
validation), for a total of 60 epochs at 320x320 resolution. Standalone
validation using Ultralytics' native IoU-based metric reached Precision
0.528, Recall 0.235, mAP50 0.229 at convergence (improvement had slowed
substantially over the final 10 epochs). This figure is reported separately
from the cross-configuration comparison in Section 5.1: the ablation table
uses a center-point containment protocol rather than IoU matching, since
IoU cannot be fairly applied across configurations whose predicted boxes
have different natural scale (a face-sized box vs. a whole-person box from
the stock detectors) — the two numbers measure related but distinct
notions of localization accuracy and are not directly comparable.

## 4. Face Recognition Evaluation

Given that ArcFace's published models are trained on datasets with millions
of images across tens of thousands of identities, retraining its backbone
from scratch on LFW's 610 identities was judged infeasible and unlikely to
improve on the pretrained model. Instead, the evaluation follows standard
deployed-system practice: the pretrained ArcFace model (via InsightFace) is
used as a frozen embedding extractor, with a real enrollment/verification
pipeline built on top — gallery embeddings per identity, cosine-similarity
matching against probes, and a decision threshold.

A classical HOG-descriptor baseline (histogram of oriented gradients +
Euclidean/cosine matching) was implemented for comparison. Because HOG
descriptors are non-negative and ArcFace embeddings are not, the two methods
have different natural cosine-similarity ranges; each method's decision
threshold was therefore calibrated independently (minimizing |FAR-FRR| on a
held-out calibration subset of 100 identities) rather than sharing one fixed
threshold across both.

Verification metrics (Accuracy, Precision, Recall, F1, FAR, FRR) were
computed using standard genuine/impostor trial methodology: for each probe,
a genuine trial tests similarity against its own enrolled identity, and an
impostor trial tests similarity against its single most similar *other*
identity (the hardest impostor case).

Four lighting conditions (bright, normal, dim, dark) were evaluated by
applying a real pixel-level contrast/brightness transform to probe images
before matching — not a synthetic score adjustment — so that any measured
robustness difference between methods reflects actual model behavior on
degraded images.

## 5. Results

### 5.1 Detection (600 held-out DARK FACE images, 4,922 face instances)

| Configuration | Precision | Recall | F1 | FPS (CPU) |
|---|---|---|---|---|
| YOLOv5 (stock, person-class) | 0.732 | 0.536 | 0.618 | 26.7 |
| YOLOv8 (stock, person-class) | 0.716 | 0.574 | 0.637 | 25.8 |
| YOLOv8 (fine-tuned face detector) | 0.879 | 0.251 | 0.390 | 61.9 |
| Fine-tuned detector + TD-FALE-GAN | 0.400 | 0.013 | 0.026 | 1.1 |

*See `accuracy_comparison.png`, `fps_comparison.png`, `lighting_comparison.png`, `confusion_matrix.png`.*

### 5.2 Recognition (610 LFW identities, 1,643 probe images)

| Lighting | HOG baseline (Acc.) | ArcFace (Acc.) | ArcFace + TD-FALE-GAN (Acc.) |
|---|---|---|---|
| Bright | 0.272 | 0.873 | 0.755 |
| Normal | 0.201 | 0.867 | 0.883 |
| Dim | 0.205 | 0.886 | **0.911** |
| Dark | 0.280 | 0.565 | 0.526 |

*See `recognition_accuracy_by_lighting.png`, `recognition_far_frr.png`.*

## 6. Discussion

**Deep-learned recognition substantially outperforms the classical
baseline.** ArcFace achieves 0.87-0.91 accuracy versus the HOG baseline's
0.20-0.28 across all lighting conditions except the most extreme dark
setting, where both degrade. The HOG baseline's false accept rate is
particularly severe (54-87% across conditions), meaning a HOG-based system
would be unsuitable for a security context regardless of lighting.

**Fine-tuning the detector trades recall for precision and inference
speed.** The fine-tuned face detector achieves substantially higher
precision (0.879 vs 0.716-0.732) and is roughly twice as fast as the stock
detectors, at the cost of lower recall (0.251 vs 0.536-0.574). This
represents genuine specialization rather than unambiguous improvement, and
the appropriate choice depends on whether a deployment prioritizes
minimizing false alarms or maximizing detection coverage.

**The enhancement GAN's effect is condition-dependent, not universal.** For
face recognition, TD-FALE-GAN improves accuracy in dim conditions (0.886 to
0.911) but reduces it in bright and extreme dark conditions. For face
detection on small, distant faces, it reduces accuracy substantially across
all conditions. This indicates that visually plausible, artifact-free
enhancement does not guarantee improved downstream task performance: a
detector or matcher exposed to enhanced images at test time that were never
part of its own training distribution may perform worse despite the input
looking clearer to a human observer. This is consistent with findings in
the broader low-light vision literature that enhancement and downstream-task
performance are not always aligned unless the two are jointly optimized.

## 7. Limitations and Future Work

- Detection and recognition were evaluated on separate datasets (DARK FACE,
  LFW respectively); a unified evaluation on a single dataset with both
  detection and identity labels was not available.
- Training was CPU-constrained: the face detector was fine-tuned at 320x320
  resolution (vs. YOLOv8's standard 640x640) and for 60 epochs, at which
  point improvement had slowed but not fully plateaued; higher resolution
  and/or a larger model variant would likely improve recall further.
- The enhancement GAN currently applies unconditionally regardless of input
  lighting; a gating mechanism that skips or scales enhancement based on
  estimated input quality is a natural next step, motivated directly by the
  condition-dependent results above.
- Evaluation was conducted entirely on public benchmark datasets; validation
  on footage from the target deployment environment (a physical laboratory)
  is planned as a follow-up but not yet completed.
