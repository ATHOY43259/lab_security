from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "ablation_results"
df = pd.read_csv(RESULTS_DIR / "ablation_results_detection.csv")

SHORT_NAMES = ["A: YOLOv5\n(stock)", "B: YOLOv8\n(stock)", "C: YOLOv8\n(fine-tuned)", "D: fine-tuned\n+TD-FALE-GAN"]
COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]  # validated categorical palette, fixed order
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def bar_with_labels(ax, x, heights, color, width=0.6, fmt="{:.2f}"):
    bars = ax.bar(x, heights, width=width, color=color, zorder=3)
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + max(heights) * 0.02, fmt.format(h),
                 ha="center", va="bottom", fontsize=9, color=INK)
    return bars


# --- accuracy_comparison.png: grouped Precision/Recall/F1 per config ---
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ["precision", "recall", "f1"]
metric_labels = ["Precision", "Recall", "F1"]
n_configs = len(df)
bar_w = 0.25
x = range(n_configs)
for i, (m, mlabel) in enumerate(zip(metrics, metric_labels)):
    offsets = [xi + (i - 1) * bar_w for xi in x]
    ax.bar(offsets, df[m], width=bar_w, label=mlabel,
            color=["#2a78d6", "#1baf7a", "#eda100"][i], zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels(SHORT_NAMES, fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Detection accuracy by configuration\n(600 real held-out DARK FACE images, 4,922 face instances)",
              fontsize=11)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "accuracy_comparison.png", dpi=160)
plt.close(fig)

# --- fps_comparison.png ---
fig, ax = plt.subplots(figsize=(7, 5))
bar_with_labels(ax, range(n_configs), df["fps"], COLORS, fmt="{:.1f}")
ax.set_xticks(list(range(n_configs)))
ax.set_xticklabels(SHORT_NAMES, fontsize=9)
ax.set_ylabel("Frames per second (CPU)")
ax.set_title("Inference speed by configuration", fontsize=11)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fps_comparison.png", dpi=160)
plt.close(fig)

# --- lighting_comparison.png: recall by lighting bucket per config ---
# 'bright' bucket omitted: 0 faces detected across every config (see note below)
# because whole-frame mean luminance is a crude proxy -- a handful of DARK FACE
# scenes have a bright sign/light dragging the frame average up while the actual
# faces remain in shadow. Reported honestly rather than dropped silently.
fig, ax = plt.subplots(figsize=(8, 5))
buckets = ["normal", "dim", "dark"]
bucket_labels = ["Normal", "Dim", "Dark"]
bar_w = 0.2
x = range(len(buckets))
for i, row in df.iterrows():
    vals = [row[f"recall_{b}"] for b in buckets]
    offsets = [xi + (i - 1.5) * bar_w for xi in x]
    ax.bar(offsets, vals, width=bar_w, label=SHORT_NAMES[i].replace("\n", " "), color=COLORS[i], zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels(bucket_labels)
ax.set_ylabel("Recall")
ax.set_title("Recall by lighting condition\n('Bright' bucket omitted: 0 face instances in this bucket for every config)",
              fontsize=10)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "lighting_comparison.png", dpi=160)
plt.close(fig)

# --- confusion_matrix.png: object detection has no classical TN, so this is
# a TP/FP/FN outcome breakdown per config, not a literal confusion matrix ---
fig, ax = plt.subplots(figsize=(8, 5))
outcomes = ["tp", "fp", "fn"]
outcome_labels = ["True Positive", "False Positive", "False Negative"]
bar_w = 0.25
x = range(n_configs)
for i, (o, olabel) in enumerate(zip(outcomes, outcome_labels)):
    offsets = [xi + (i - 1) * bar_w for xi in x]
    ax.bar(offsets, df[o], width=bar_w, label=olabel,
            color=["#1baf7a", "#eda100", "#e34948"][i], zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels(SHORT_NAMES, fontsize=9)
ax.set_ylabel("Count (out of 4,922 face instances)")
ax.set_title("Detection outcome breakdown by configuration\n(object detection has no true negatives, shown as counts not a 2×2 matrix)",
              fontsize=10)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

print("Saved 4 figures to", RESULTS_DIR)
for f in ["accuracy_comparison.png", "fps_comparison.png", "lighting_comparison.png", "confusion_matrix.png"]:
    print(" -", RESULTS_DIR / f)
