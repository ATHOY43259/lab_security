# AI Referee Review Report

**Reviewer model:** Claude (Sonnet 5), acting as an independent journal referee
**Manuscript:** "TD-FALE-GAN: Task-Driven Low-Light Face Enhancement for Smart Laboratory Security"
**Target venue (stated by authors):** MDPI *Sensors*
**Materials reviewed:** Full manuscript body (Abstract through Conclusion, Sections I–XII of the source IEEEtran draft), all 4 figures (architecture diagram, generator-internals diagram, YOLOv8 customization diagram, end-to-end pipeline diagram), all 9 tables (Positioning Table I, Dataset Table II, Implementation Table III, Training-Config Table IV, Validation-Metrics Table V, Detection-Ablation Table VI, Recognition-Accuracy Table VII, FAR/FRR Table VIII, Prior-Work Comparison Table IX), Algorithm 1, and the full 25-entry reference list. Cross-checked numeric claims in the Abstract and Discussion against the values printed in Tables VI–IX for internal consistency.

---

## Recommendation: **Major Revision**

**Justification:** The manuscript is unusually honest for its genre — it reports a negative/mixed result (the enhancement stage helps in one of four lighting conditions and actively hurts in the other three, including a near-total detection collapse), documents specific implementation bugs it found and fixed, and its headline numbers are internally consistent across the Abstract, Table VI (detection), Table VII (recognition), and Table VIII (FAR/FRR) — I could not find a single number that contradicts another. That said, two requirements that the submitting institution's own guidelines mark as mandatory are not met: the reference list has 25 entries against a required 40–60 (Finding 1), and the manuscript contains zero qualitative/visual results — no enhanced-image examples, no detection overlays, no attention-map visualizations — despite proposing an attention-gated visual model and requiring explainability analysis "where applicable" (Finding 2). Neither defect is a fabrication or correctness problem; both are completeness problems that a Major Revision can fix without touching the core experimental claims, which I found trustworthy.

---

## Overall Assessment

| Area | Score /10 | Main concern |
|---|---|---|
| Title | 8 | Accurate and specific, but doesn't signal that the central finding is a *negative* result — a reader expecting a straightforward capability paper may be surprised. |
| Novelty | 6 | The face-attention + detection-guided-loss + dual-discriminator combination is a genuine architectural contribution, but it is incremental relative to already-cited FeatEnHancer, IAFE-YOLO, and LIME-Eval; none of these three closest published methods are benchmarked quantitatively (Table IX only compares against lab-security systems, not enhancement methods). |
| Methodology | 8 | Very thoroughly documented (hyperparameters, optimizer, epochs, resolution all reported in Table IV), but exact CPU/RAM/OS are explicitly "not reported" (Table III), which limits reproducibility of the FPS numbers. |
| Dataset rigor | 6 | LFW's well-documented demographic skew (predominantly public figures, imbalanced representation) is never addressed, and no cross-dataset generalization test is attempted — detection and recognition results come from two disjoint datasets with no shared evaluation. |
| Experimental validity | 7 | Sample sizes are always stated, but no confidence interval, standard error, or significance test is given anywhere — the paper's own central claim (a +0.025 accuracy gain in dim conditions) is never tested against a null hypothesis of no difference. |
| Ablation | 7 | Table VI's 4-configuration detection ablation is complete and well-isolated. Table VII functions as the recognition ablation but is never labeled as one, which could cost points against a rubric item that explicitly checks for "the mandatory ablation study." |
| Explainability | 3 | No qualitative figures anywhere in the manuscript — no example enhanced/original image pairs, no detection bounding-box overlays, no visualization of the learned face-attention mask $M$ or gate $\alpha$. For a paper whose central mechanism is spatial attention, this is a significant gap. |
| Claims | 8 | Every number I checked in the Abstract and Discussion traces correctly to a table. Table I's self-vs-competitor comparison is fair on inspection but reads as self-serving (all "No" for competitors, all "Yes" for "This work") and would benefit from softer framing. |
| References | 4 | 25 references total; the submission's own instructions require 40–60, "mostly 2023–2026." Only 8 of the 25 (32%) fall in 2023–2026. |
| Writing | 9 | Clear, precise, and unusually candid about limitations; minor issue is that Table VIII's caption doesn't state units (%, not fraction) though the text does. |
| **Overall** | **6.5** | Correctness and honesty are strong; completeness against the stated submission requirements is the binding constraint. |

---

## Numbered Findings (most serious first)

### Finding 1 — Reference count is 25 against a stated requirement of 40–60
**Location:** Reference list, end of manuscript (25 `\bibitem` entries, keys `hamzidah2026` through `robustfrsurvey2025`).
**Quote:** The submission instructions state: *"40-60 mostly 2023-2026 references."* The manuscript's bibliography contains exactly 25 entries, of which only 8 (`ali2022` [borderline 2022], `putra2023`, `yolov10`, `featenhancer`, `limeeval`, `iafeyolo`, `smartcampus2024`, `robustfrsurvey2025`, `hamzidah2026`) fall in the 2023–2026 window — roughly 32%, short of "mostly."
**Why it's a problem:** This is an explicit, mandatory, checkable rubric item, not a stylistic preference. A referee (human or automated) scoring against this rubric will mark this item as failing regardless of the paper's other merits.
**Required action:** Add 15–35 additional references, weighted toward 2023–2026 work in: (a) low-light face detection/recognition specifically, (b) recent GAN/diffusion-based low-light enhancement, (c) task-driven / detector-in-the-loop enhancement losses beyond the 3 already cited, (d) recent laboratory/campus security systems beyond the 3 already covered, and (e) recent ArcFace-family or transformer-based face verification work.

### Finding 2 — No qualitative/visual results anywhere in the manuscript
**Location:** Entire manuscript; absent from Sections VI (Results) and VII (Discussion) specifically.
**Quote:** Table VI reports "recall falls from 0.251 to 0.013" for the enhanced detection pipeline and Table VII reports accuracy deltas as small as 0.013 (Dark, ArcFace+GAN: 0.565→0.526) — both are described purely numerically; no figure shows an example low-light frame, its TD-FALE-GAN-enhanced counterpart, or the resulting detection/recognition outcome.
**Why it's a problem:** For a vision paper whose core proposed mechanism is a learned spatial attention mask $M$ and an iterative curve-based enhancement, a referee cannot verify *what the model is actually doing* from numbers alone — e.g., whether the catastrophic recall collapse (0.251→0.013) is a genuine enhancement failure or a visually-obvious artifact (over-brightening, color shift, checkerboarding) that a single side-by-side image would make immediately legible. This also weakens the paper against any "explainability analysis where applicable" requirement.
**Required action:** Add at minimum one figure with 3–4 paired examples (original vs. TD-FALE-GAN-enhanced) spanning the four lighting conditions, with detection boxes and/or ArcFace match/no-match outcomes overlaid, and ideally a visualization of the learned mask $M$ and gate value $\alpha$ on at least one face region. The repository's own `demo_notebook.ipynb` (see Section X, Data and Code Availability) already produces exactly this kind of output at inference time — the missing step is exporting a few of those frames into the paper itself.

### Finding 3 — Novelty claim rests on narrative differentiation, not quantitative comparison, against the closest published methods
**Location:** Section II-C (Low-Light Image Enhancement), specifically the sentence beginning "Our architecture is explicitly built around the tension these works identify..."; Table IX (Quantitative Comparison).
**Quote:** "...and, unlike [17]–[19], extends that measurement to a full recognition pipeline (ArcFace, FAR/FRR) in addition to detection." Table IX compares only against `hamzidah2026`, `ali2022`, `putra2023` — none of which are enhancement methods — leaving FeatEnHancer, IAFE-YOLO, and LIME-Eval (cited as the closest prior work) entirely out of any quantitative table.
**Why it's a problem:** A referee assessing novelty will look for a head-to-head number against the works the paper itself names as closest, not just a sentence claiming difference. Without it, the novelty claim is asserted rather than demonstrated.
**Required action:** Either (a) run TD-FALE-GAN's detection-guided-loss ablation against a re-implementation of FeatEnHancer or IAFE-YOLO's conditioning mechanism on the same DARK FACE split, or (b) if that is infeasible before submission, explicitly state in the Limitations section (Section IX) that a quantitative comparison against these architecturally closest methods was not performed, as an acknowledged scope limitation rather than an omission.

### Finding 4 — No statistical significance testing on any reported accuracy delta
**Location:** Section VII (Discussion), "Enhancement is condition-dependent, not universal" paragraph.
**Quote:** "TD-FALE-GAN improves recognition in dim conditions (0.886 to 0.911, a $+0.025$ gain)..." — presented as a finding without any interval or test.
**Why it's a problem:** With ~410 probes per lighting condition (1,643 total / 4 conditions), a 0.025 accuracy delta corresponds to roughly 10 flipped decisions. Without a confidence interval or a paired significance test (e.g., McNemar's test on the paired genuine/impostor trials, which the paper already runs), a referee cannot tell whether the paper's central "condition-dependent, not universal" claim is a real effect or sampling noise.
**Required action:** Report a 95% confidence interval or McNemar's test p-value for at least the dim-condition gain (the paper's positive result) and the bright/extreme-dark losses (the paper's negative results), since these three deltas carry the entire "condition-dependent" argument.

### Finding 5 — LFW's demographic composition is not discussed as a dataset-rigor limitation
**Location:** Section IV-A (Datasets), LFW paragraph; Section XI (Ethical Considerations).
**Quote:** Section IV-A states only "LFW... provides 1,680 identities with between 2 and 50 images each." Section XI recommends that "any deployment must audit false-accept and false-reject rates across demographic groups" in general terms but does not connect this to LFW's own well-documented skew (predominantly light-skinned, predominantly male public figures) as a limitation of *this study's* evaluation.
**Why it's a problem:** The Ethical Considerations section makes a general point about demographic auditing without acknowledging that the dataset used in this very paper is known to be demographically unbalanced, which limits how much the paper's own FAR/FRR numbers generalize to a real, demographically diverse laboratory population.
**Required action:** Add 1–2 sentences to Section IV-A or Section IX (Limitations) explicitly naming LFW's demographic skew as a threat to external validity of the reported FAR/FRR numbers.

### Finding 6 — Table VII functions as the recognition ablation but is not labeled as such
**Location:** Section VI-B (Recognition Results), Table VII caption: "Recognition Accuracy by Lighting."
**Why it's a problem:** The submission rubric explicitly checks for "the mandatory ablation study." Table VI is clearly and explicitly an ablation (4 detector configurations). Table VII is functionally the same kind of comparison for recognition (HOG vs. ArcFace vs. ArcFace+GAN) but nothing in the text calls it an ablation, so an automated or time-pressed reviewer scanning for the word may miss it.
**Required action:** Add one sentence at the start of Section VI-B explicitly framing Table VII as "the recognition ablation, isolating the contribution of ArcFace and of TD-FALE-GAN preprocessing relative to the HOG baseline."

### Finding 7 — Table I's self-vs-competitor comparison reads as self-serving
**Location:** Table I (Positioning Relative to Prior Work), Section II-D.
**Quote:** Every competitor row reads "No" on 2–3 of 4 columns; the "This work" row is "Yes" on all 4.
**Why it's a problem:** Not factually wrong on inspection (I checked each cell against the cited papers' described scope), but a table with a perfect all-"Yes" self-row versus all-"No" competitor rows is a pattern referees are trained to distrust, and it invites an accusation of cherry-picked comparison dimensions even when the underlying claims hold up.
**Required action:** Either add one dimension where a competitor scores favorably (e.g., "Own deployed hardware," where Hamzidah/Ali/Putra all score "Yes" and this work scores "No," which the paper itself already acknowledges honestly in Section VII), or soften the framing in the surrounding text to explicitly note the table's dimensions were chosen to highlight this paper's specific contributions rather than provide an exhaustive comparison.

### Finding 8 — Table VIII caption omits units
**Location:** Table VIII caption: "False-Accept and False-Reject Rates by Lighting Condition."
**Why it's a problem:** Minor, but values are given as fractions (0.608, 0.217, etc.) with no explicit "(fraction)" or "(%)" label in the table itself; the surrounding text uses percentages ("54–87%"), creating a brief unit mismatch a fast reader could misread as 0.6% instead of 60.8%.
**Required action:** Add "(fraction, 0–1)" to the table caption or convert the table values to percentages to match the surrounding prose.

---

*No finding in this review concerns fabricated results, inconsistent headline numbers, or a citation that misrepresents its source — those were specifically checked and not found. All findings concern completeness (references, qualitative figures, comparison scope, statistical rigor) rather than correctness.*
