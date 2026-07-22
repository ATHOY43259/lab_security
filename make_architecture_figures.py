"""Generates two IEEE-style architecture figures for the paper:
  fig3_generator_architecture.png  -- internal FALEGenerator structure
  fig4_yolov8_customization.png    -- stock YOLOv8 -> fine-tuned face detector
Matches the plain black/white box-and-arrow style of the paper's existing
Fig. 1 / Fig. 2, built from the exact layer definitions in gan/model.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ablation_results"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "text.color": "black",
})


def box(ax, x, y, w, h, text, fontsize=9, fc="white", ec="black", lw=1.1, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                 fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4, linespacing=1.35)


def arrow(ax, xy1, xy2, style="-|>", lw=1.1, connectionstyle="arc3,rad=0.0", color="black"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=11,
                                  lw=lw, color=color, connectionstyle=connectionstyle, zorder=2))


# ============================================================
# Figure 3: FALEGenerator internal architecture
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 15.5)
ax.set_ylim(-5.4, 3.2)
ax.axis("off")

# Inputs (stacked, left)
box(ax, 1.0, 2.0, 1.8, 0.7, "Frame $I$\n(3 ch)", fontsize=8.5)
box(ax, 1.0, 1.0, 1.8, 0.7, "Face mask $M$\n(1 ch)", fontsize=8.5)
box(ax, 1.0, 0.0, 1.8, 0.7, "Brightness $b$\n(1 ch)", fontsize=8.5)
box(ax, 3.0, 1.0, 1.6, 1.1, "Concat\n(5 ch)", fontsize=9)
for y in (2.0, 1.0, 0.0):
    arrow(ax, (1.9, y), (2.2, 1.0), connectionstyle=f"arc3,rad={(1.0-y)*0.15}")

# Encoder row (top), 3x3 conv + ReLU, 32ch -- spacing 2.1, box width 1.7 (0.4 gap)
enc_x = [5.0, 7.1, 9.2, 11.3]
enc_labels = ["$e_1$\nConv 5$\\to$32\n3$\\times$3, ReLU", "$e_2$\nConv 32$\\to$32\n3$\\times$3, ReLU",
              "$e_3$\nConv 32$\\to$32\n3$\\times$3, ReLU", "$e_4$\nConv 32$\\to$32\n3$\\times$3, ReLU"]
for x, lab in zip(enc_x, enc_labels):
    box(ax, x, 1.0, 1.7, 1.3, lab, fontsize=7.8)
arrow(ax, (3.8, 1.0), (4.15, 1.0))
for i in range(3):
    arrow(ax, (enc_x[i] + 0.85, 1.0), (enc_x[i + 1] - 0.85, 1.0))

# Decoder row (bottom), mirrored x-positions under e1, e2, e3 -- box width 1.6 (0.5 gap)
dec_x = {"d3": enc_x[2], "d2": enc_x[1], "d1": enc_x[0]}
box(ax, dec_x["d3"], -1.5, 1.6, 1.3, "$d_3$\nConv 64$\\to$32\n3$\\times$3, ReLU", fontsize=7.8)
box(ax, dec_x["d2"], -1.5, 1.6, 1.3, "$d_2$\nConv 64$\\to$32\n3$\\times$3, ReLU", fontsize=7.8)
box(ax, dec_x["d1"], -1.5, 1.6, 1.3, "$d_1$\nConv 64$\\to$12\n3$\\times$3, tanh", fontsize=7.8)

# Skip connections (vertical, encoder -> decoder at same x)
arrow(ax, (enc_x[0], 0.35), (dec_x["d1"], -0.85), color="dimgray")
arrow(ax, (enc_x[1], 0.35), (dec_x["d2"], -0.85), color="dimgray")
arrow(ax, (enc_x[2], 0.35), (dec_x["d3"], -0.85), color="dimgray")
# Bottleneck e4 -> d3
arrow(ax, (enc_x[3], 0.35), (dec_x["d3"] + 0.55, -0.85), connectionstyle="arc3,rad=-0.25")
# decoder chain d3 -> d2 -> d1 (real gap now: 2.1 spacing - 1.6 width = 0.5 clear)
arrow(ax, (dec_x["d3"] - 0.8, -1.5), (dec_x["d2"] + 0.8, -1.5))
arrow(ax, (dec_x["d2"] - 0.8, -1.5), (dec_x["d1"] + 0.8, -1.5))

ax.text(9.2, 0.35, "skip connections (concat)", fontsize=7.5, color="dimgray", style="italic",
        ha="center", va="bottom")

# Bottom row: d1 output -> attention gate -> iterative refinement -> output
# Generous, non-overlapping spacing: centers at 5.0, 8.4, 12.2, 15.0-ish -- widened canvas to fit
gate_x, refine_x, out_x = 6.0, 10.0, 13.8
arrow(ax, (dec_x["d1"], -2.15), (dec_x["d1"], -2.6))
ax.text(dec_x["d1"] + 0.15, -2.4, "curves $A$\n(12 ch)", fontsize=7.3, color="dimgray", ha="left", va="center")
arrow(ax, (dec_x["d1"], -2.85), (gate_x - 1.35, -2.85))

box(ax, gate_x, -2.85, 2.5, 0.95, "Attention gate\n$A' = A \\cdot (1+\\alpha M)$", fontsize=7.8)
arrow(ax, (1.0, 0.35), (1.0, -2.55), connectionstyle="arc3,rad=0.5")
arrow(ax, (1.0, -2.55), (gate_x - 1.35, -3.2), connectionstyle="arc3,rad=-0.15")
ax.text(0.3, -1.3, "$M$", fontsize=8, color="dimgray")

box(ax, refine_x, -2.85, 3.1, 1.3,
    "Iterative curve refinement\nfor $k=1{\\ldots}4$:\n"
    "$\\hat I \\leftarrow \\hat I + a_k(\\hat I^2-\\hat I)$", fontsize=7.8)
arrow(ax, (gate_x + 1.25, -2.85), (refine_x - 1.55, -2.85))

# initial value path: I -> refinement block (long detour below everything else)
arrow(ax, (1.0, -0.35), (1.0, -4.6), connectionstyle="arc3,rad=-0.4")
arrow(ax, (1.0, -4.6), (refine_x - 1.55, -4.6))
arrow(ax, (refine_x - 1.55, -4.6), (refine_x - 1.55, -3.45))
ax.text(3.5, -4.85, "$\\hat I \\leftarrow I$  (initial value)", fontsize=7.5, color="dimgray", style="italic")

box(ax, out_x, -2.85, 1.9, 0.9, "Enhanced\nframe $I'$", fontsize=9)
arrow(ax, (refine_x + 1.55, -2.85), (out_x - 0.95, -2.85))

ax.text(7.7, 2.85, "Fig. 3: Internal architecture of the FALE-GAN generator "
        "(encoder-decoder with skip connections, face-attention gating,\n"
        "and iterative curve-based enhancement). Channel counts and layer "
        "order match the implementation exactly.",
        fontsize=8.3, ha="center", va="top")

fig.savefig(OUT / "fig3_generator_architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved fig3_generator_architecture.png")


# ============================================================
# Figure 4: YOLOv8 customization (stock -> fine-tuned face detector)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 12)
ax.set_ylim(-1, 7.6)
ax.axis("off")

# Stock model (top)
box(ax, 2.2, 6.6, 3.4, 0.8, "YOLOv8n backbone + neck\n(CSPDarknet, C2f, SPPF)", fontsize=8.3)
box(ax, 7.3, 6.6, 3.6, 0.8, "Detect head\n80-class COCO output", fontsize=8.3)
arrow(ax, (2.2 + 1.7, 6.6), (7.3 - 1.8, 6.6))
ax.text(0.2, 6.6, "Stock\n(pretrained\non COCO)", fontsize=8, ha="left", va="center", style="italic")

# transfer arrow down
arrow(ax, (2.2, 6.2), (2.2, 5.0), lw=1.4)
ax.text(2.85, 5.6, "weights transferred\n(319 / 355 tensors)", fontsize=7.6, color="dimgray", ha="left")

arrow(ax, (7.3, 6.2), (7.3, 5.0), lw=1.4, color="firebrick")
ax.text(8.0, 5.6, "classification layers\nreinitialized (nc: 80$\\to$1)", fontsize=7.6, color="firebrick", ha="left")

# Fine-tuned model (bottom)
box(ax, 2.2, 4.3, 3.4, 0.8, "YOLOv8n backbone + neck\n(same architecture, fine-tuned)", fontsize=8.3)
box(ax, 7.3, 4.3, 3.6, 0.8, "Detect head\nsingle-class \"face\" output", fontsize=8.3)
arrow(ax, (2.2 + 1.7, 4.3), (7.3 - 1.8, 4.3))
ax.text(0.2, 4.3, "Fine-tuned\n(this work)", fontsize=8, ha="left", va="center", style="italic")

# Training data box feeding into fine-tuning
box(ax, 4.75, 2.7, 4.9, 1.0,
    "DARK FACE: 45,474 face boxes\n5,400 train / 600 held-out images\n"
    "60 epochs (12+18+30), 320$\\times$320", fontsize=7.9)
arrow(ax, (2.2, 3.9), (3.6, 3.2), connectionstyle="arc3,rad=-0.2")
arrow(ax, (7.3, 3.9), (6.0, 3.2), connectionstyle="arc3,rad=0.2")

# Result box
box(ax, 4.75, 0.9, 5.6, 1.1,
    "Result (own validation split):\nPrecision 0.528  Recall 0.235  mAP$_{50}$ 0.229\n"
    "(standard IoU-based protocol; see Table V)", fontsize=7.9, fc="whitesmoke")
arrow(ax, (4.75, 2.2), (4.75, 1.45))

ax.text(6, -0.6,
        "Fig. 4: Face-detector customization. The backbone and neck retain "
        "their COCO-pretrained weights; only the Detect head's\n"
        "classification branch is reinitialized for the single \"face\" class, "
        "then the full network is fine-tuned end-to-end on DARK FACE.",
        fontsize=8.3, ha="center", va="top")

fig.savefig(OUT / "fig4_yolov8_customization.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved fig4_yolov8_customization.png")
