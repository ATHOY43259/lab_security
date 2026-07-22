# Demonstration Guide — Smart Lab Security System

## PART 1 — Live Demo Script (run these in order, in your terminal)

Keep this open on a second screen/phone. Each step includes what to say.

### Step 0 — Orient your supervisor (30 seconds, no command)
"This is a lab security system combining face detection, face recognition, and a
custom low-light enhancement model. Everything I'm about to show is running on
real public datasets and real trained models — no simulated numbers."

### Step 1 — Show the real data
```
cd C:\Users\ASUS\lab_security_real
dir datasets\ieee_darkface\image | find /c ".png"
dir "datasets\face_recognition\Face Data\Face Dataset"
```
**Say:** "6,000 real nighttime surveillance-style photos from the DARK FACE
academic dataset, and 1,680 real identities from Labeled Faces in the Wild,
for detection and recognition respectively."

### Step 2 — Live face detection demo (~2 seconds)
```
python demo_live_detection.py
```
Then open `demo_output\live_detection_demo.png`.
**Say:** "This is my fine-tuned YOLOv8 face detector — fine-tuned from the stock
COCO model specifically for faces — running live on a real dark photo it has
never seen. You can see it finds the small, distant faces in the crowd with
confidence scores."

### Step 3 — Live face recognition demo (~5 seconds)
```
python demo_live_recognition.py
```
**Say:** "This enrolls three real people from the LFW dataset, then shows a
held-out photo of one of them being correctly matched by ArcFace, based on
cosine similarity between 512-dimensional face embeddings — 0.709 similarity
to the correct person versus negative similarity to the other two."

### Step 4 — Show the GAN before/after
Open these two side by side (already generated):
```
gan\samples\compare_1.png
gan\samples\compare_2500.png
```
**Say:** "This is TD-FALE-GAN, a custom low-light enhancement architecture I
designed and trained myself — the right half of each image is the enhanced
output. It brightens the scene without introducing noise artifacts, which
took two training iterations to get right." *(If asked what went wrong the
first time — see Viva Q6.)*

### Step 5 — Show the ablation results tables
Open `ablation_results\ablation_results_detection.csv` and
`ablation_results\ablation_results_recognition.csv`, and the 6 PNG figures
in that folder.
**Say:** "These are the actual measured results comparing 4 detection
configurations and the recognition pipeline across 4 lighting conditions,
on 600 and 610 held-out real images/identities respectively."

### Step 6 — The honest finding (this is your strongest material)
**Say:** "The most important finding isn't that my method wins everywhere —
it's that the GAN enhancement helps in some conditions and hurts in others.
[Point to recognition_accuracy_by_lighting.png] It improves recognition
accuracy in dim lighting (88.6% to 91.1%) but hurts in bright and extreme
dark conditions, and hurts face detection entirely on tiny distant faces.
I measured this rather than assumed it, and I can explain exactly why it
happens." *(Full explanation in Viva Q7-Q8 below.)*

### Step 7 — Paper section
Open `paper_section_methodology_results.md` and show the structured
write-up if your supervisor wants the formal document.

---

## PART 2 — Full Project Understanding (explain from first principles)

### The problem
Laboratory security systems relying on face detection/recognition fail
under poor lighting — the exact conditions many real intrusions would
occur in (night, corridors, dim rooms). The project asks: can we build
and *honestly measure* whether low-light image enhancement actually helps
a real face detection + recognition security pipeline, rather than assume
it does.

### The pipeline, end to end
1. **Input**: a camera frame, potentially in low light.
2. **(Optional) Enhancement**: TD-FALE-GAN brightens the frame if enabled.
3. **Detection**: YOLO finds face locations in the frame.
4. **Recognition**: ArcFace computes a 512-dim embedding for each detected
   face and compares it against enrolled identities via cosine similarity.
5. **Decision**: accept/reject based on similarity threshold — this is
   the actual security decision (FAR/FRR are the error rates of this step).

### TD-FALE-GAN — what makes it "yours," not a stock tool
Standard low-light GANs (e.g. Zero-DCE, EnlightenGAN) brighten the whole
frame uniformly and optimize only for human-perceived image quality. TD-FALE-GAN
adds three things specific to this security use case:
- **Face-attention gating**: enhancement concentrates on face regions
  (using a mask built from detected/labeled face boxes), not the whole frame.
- **Illumination conditioning**: the network sees an estimate of how dark
  the whole frame is, so one model adapts its correction strength instead
  of using one fixed transformation for every brightness level.
- **Detection-guided loss**: during training, a frozen YOLOv8 detector's
  confidence on the enhanced output is itself part of the loss — the model
  is pushed toward images that are more detectable, not just brighter.

### Why fine-tune the detector at all?
The stock YOLOv8 model detects "person" (COCO's 80 general classes) — it
was never taught what a face specifically looks like, especially small or
low-light ones. Fine-tuning on DARK FACE's 45,474 real face boxes turns it
into a dedicated face detector.

### Why ArcFace instead of training a recognizer from scratch?
Published ArcFace models are trained on millions of images across tens of
thousands of identities. Retraining that from scratch on 610 LFW identities
on a CPU would very likely produce a *worse* model, and isn't how real
deployed systems work anyway — they use a pretrained embedding extractor
and build enrollment/matching on top, which is exactly what was built here.

### The two ablation studies
- **Detection ablation**: 4 configurations (YOLOv5 stock, YOLOv8 stock,
  fine-tuned YOLOv8, fine-tuned+GAN) measured on 600 real held-out DARK
  FACE images.
- **Recognition ablation**: HOG baseline vs ArcFace vs ArcFace+GAN,
  measured on 610 real LFW identities across 4 real lighting conditions
  (via actual pixel-level darkening, not simulated score penalties).

### The headline results
| | Detection | Recognition (avg across lighting) |
|---|---|---|
| Best config | Fine-tuned YOLOv8 (P=0.879) | ArcFace (0.87-0.91 acc.) |
| Weak baseline | Stock YOLOv5/v8 (lower precision) | HOG (0.20-0.28 acc.) |
| GAN effect | Hurts across the board | Helps in dim, hurts in bright/dark |

---

## PART 3 — Viva Questions and Answers

**Q1: Why did you choose DARK FACE and LFW specifically?**
A: DARK FACE is the standard academic benchmark for real-world low-light
face detection (6,000 real nighttime images with face labels) — it directly
matches the lighting-robustness axis of this project. LFW provides real
multi-photo identity labels needed for recognition testing, which no
low-light dataset I found actually provides (low-light face datasets have
boxes, not identities).

**Q2: Why not test on real laboratory footage?**
A: That's an honest, stated limitation. No public dataset of real
laboratory-interior security footage exists — organizations don't publish
that for privacy/security reasons — so public benchmarks were used to
validate the methodology, with lab-specific footage planned as a follow-up
validation step, not yet completed.

**Q3: Why does your fine-tuned detector have lower recall than the stock
models?**
A: It trades recall for precision and speed — 0.879 precision vs
0.716-0.732 for stock models, and roughly 2x the inference speed, at the
cost of recall (0.251 vs 0.536-0.574). This is a real specialization
effect: the fine-tuned model is more conservative, firing less often but
being right more often when it does. Which is "better" depends on whether
a deployment prioritizes minimizing false alarms or maximizing detections.

**Q4: Why is recall/mAP still relatively low overall?**
A: Compute-constrained training — CPU-only, so trained at 320px resolution
instead of YOLOv8's standard 640px (which specifically hurts small/distant
faces), for 60 epochs, at which point improvement had clearly slowed. This
is an honest, explainable ceiling, not a hidden flaw — the fix (higher
resolution, more epochs, GPU) is known and stated as future work.

**Q5: Why use a frozen ArcFace instead of training your own recognition
model, if the goal is originality?**
A: Because ArcFace's backbone requires far more data/compute than is
feasible here, and more importantly, that's not how production
face-recognition security systems work — they enroll identities against a
pretrained embedding extractor rather than retraining the network per
deployment. The original contribution is the enrollment/verification
*pipeline and evaluation*, not reinventing the embedding network.

**Q6: What went wrong during GAN training, and how did you fix it?**
A: The first training run produced purple/magenta noise artifacts,
especially in near-black background regions. Root cause: the enhancement
formula applied 8 correction iterations, each amplifying tiny per-channel
sensor noise independently — compounding into visible color speckle in
very dark pixels. Fixed by reducing to 4 iterations, adding a direct
smoothness loss on the output image (not just the internal curve maps),
and slowing the discriminator's learning rate, since it was overpowering
the generator too early in training.

**Q7: Why does the GAN help in some lighting conditions but hurt in
others?**
A: Two separate mechanisms. First, it was trained to correct dark images —
applying it to an already well-lit (bright) image over-processes something
that didn't need correction. Second, in extreme darkness, both the
enhancer and the downstream detector/matcher are pushed outside the range
they were designed/trained for. The genuine sweet spot is moderate
degradation ("dim"), where enhancement helps without over-correcting.

**Q8: Why does GAN preprocessing hurt face detection so severely (recall
0.251 to 0.013), more than it hurts recognition?**
A: The face detector was fine-tuned exclusively on raw, non-enhanced
images. Its learned features are tuned to the specific pixel statistics of
untouched camera sensor noise. Enhanced images, even though visually clean,
have different fine-grained statistics (subtle color/contrast shifts) that
push them outside the detector's trained distribution. Face crops for
recognition are less affected because ArcFace's training data is broader
and the crops are larger relative to the frame, but detection on tiny
distant faces is much more sensitive to this shift.

**Q9: Is your work novel, or just combining existing tools (YOLO, ArcFace,
GAN)?**
A: The individual tools aren't novel — the contribution is (1) TD-FALE-GAN's
specific architecture (face-attention gating + illumination conditioning +
detection-guided loss, verified to genuinely backpropagate into the
generator), and more importantly (2) the empirical finding that enhancement
benefit is condition- and task-dependent, measured rather than assumed —
which contradicts the common assumption that low-light enhancement
universally helps downstream vision tasks.

**Q10: How do you know your results are real and not fabricated?**
A: Every number came from code I can show running live, on datasets with
public download links, using models with verifiable weights files. Several
real bugs were found and fixed during development (a hardcoded fairness bug
in threshold comparison, a resolution-destroying preprocessing step, a
misleading timing measurement) — I can show the before/after of each fix
and explain why the bug existed. A fabricated result wouldn't have
inconvenient findings like "our proposed GAN hurts detection accuracy."

**Q11: What would you do with more time/compute?**
A: Four concrete next steps, in priority order: (1) collect real lab
footage for target-environment validation, (2) add a lighting-adaptive
gate to the GAN so it only enhances when the input actually needs it,
(3) retrain the detector at 640px resolution with more epochs to close the
recall gap, (4) jointly fine-tune the detector on GAN-enhanced images so
it's no longer evaluating out-of-distribution inputs.

**Q12: What is the FAR/FRR and why does it matter for a security system?**
A: False Accept Rate is the probability the system wrongly lets in someone
who isn't enrolled (a security breach). False Reject Rate is the
probability it wrongly blocks a legitimate person (a usability failure).
The HOG baseline's FAR is 54-87% across conditions — meaning it would let
in almost anyone, making it unusable for security regardless of accuracy.
ArcFace's FAR stays under 23% in all but the most extreme dark condition.
