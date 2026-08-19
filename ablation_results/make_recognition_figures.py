from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "ablation_results"
df = pd.read_csv(RESULTS_DIR / "ablation_results_recognition.csv")

CONFIGS = ["Baseline (HOG + Euclidean-cosine)", "ArcFace (InsightFace)", "ArcFace + TD-FALE-GAN"]
SHORT_NAMES = ["HOG baseline", "ArcFace", "ArcFace +\nTD-FALE-GAN"]
LIGHTING_ORDER = ["bright", "normal", "dim", "dark"]
COLORS = ["#2a78d6", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

df["lighting"] = pd.Categorical(df["lighting"], categories=LIGHTING_ORDER, ordered=True)
df = df.sort_values(["lighting", "config"])

# --- recognition_accuracy_by_lighting.png ---
fig, ax = plt.subplots(figsize=(9, 5.5))
bar_w = 0.25
x = range(len(LIGHTING_ORDER))
for i, cfg in enumerate(CONFIGS):
    sub = df[df["config"] == cfg].set_index("lighting").reindex(LIGHTING_ORDER)
    offsets = [xi + (i - 1) * bar_w for xi in x]
    bars = ax.bar(offsets, sub["accuracy"], width=bar_w, label=SHORT_NAMES[i], color=COLORS[i], zorder=3)
    for b, v in zip(bars, sub["accuracy"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels([l.capitalize() for l in LIGHTING_ORDER])
ax.set_ylabel("Verification accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Face recognition accuracy by lighting condition\n(610 real LFW identities, per-method calibrated thresholds)", fontsize=11)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "recognition_accuracy_by_lighting.png", dpi=160)
plt.close(fig)

# --- recognition_far_frr.png: FAR and FRR side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, metric, title in zip(axes, ["far", "frr"], ["False Accept Rate (FAR)", "False Reject Rate (FRR)"]):
    for i, cfg in enumerate(CONFIGS):
        sub = df[df["config"] == cfg].set_index("lighting").reindex(LIGHTING_ORDER)
        offsets = [xi + (i - 1) * bar_w for xi in x]
        ax.bar(offsets, sub[metric], width=bar_w, label=SHORT_NAMES[i], color=COLORS[i], zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([l.capitalize() for l in LIGHTING_ORDER])
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=3)
fig.suptitle("Security-relevant error rates by lighting condition", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "recognition_far_frr.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved recognition figures to", RESULTS_DIR)
