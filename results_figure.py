"""
Results figure: box plot of 25 AUC values per mode + Wilcoxon signed-rank tests.
Reads AUC values directly from the training log file.
Usage: python3 results_figure.py [logfile]
"""

import sys
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "training_output_v10.log"
OUTPUT   = "results_figure.png"

# ── Parse log ─────────────────────────────────────────────────────────────────
aucs = {"multimodal": [], "image_only": [], "genomic_only": []}
current_mode = None

mode_re = re.compile(r"ABLATION MODE:\s+(\w+)", re.IGNORECASE)
fold_re = re.compile(r"Fold \d+ best AUC:\s+([\d.]+)")

with open(LOG_FILE) as f:
    for line in f:
        m = mode_re.search(line)
        if m:
            raw = m.group(1).lower()
            if "multi" in raw:   current_mode = "multimodal"
            elif "image" in raw: current_mode = "image_only"
            elif "genom" in raw: current_mode = "genomic_only"
        m = fold_re.search(line)
        if m and current_mode:
            aucs[current_mode].append(float(m.group(1)))

for mode, vals in aucs.items():
    print(f"{mode}: {len(vals)} AUCs  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

if not all(len(v) == 25 for v in aucs.values()):
    print("WARNING: expected 25 AUCs per mode — run may not be complete yet.")

# ── Stats ─────────────────────────────────────────────────────────────────────
mm  = np.array(aucs["multimodal"])
img = np.array(aucs["image_only"])
gen = np.array(aucs["genomic_only"])

n = min(len(mm), len(img), len(gen))
print(f"\nWilcoxon signed-rank tests (n={n} paired AUCs):")
if n >= 5:
    _, p_mm_img = wilcoxon(mm[:n], img[:n])
    _, p_mm_gen = wilcoxon(mm[:n], gen[:n])
    _, p_img_gen = wilcoxon(img[:n], gen[:n])
    print(f"  multimodal vs image_only:   p = {p_mm_img:.4f}  {'*' if p_mm_img < 0.05 else 'ns'}")
    print(f"  multimodal vs genomic_only: p = {p_mm_gen:.4f}  {'*' if p_mm_gen < 0.05 else 'ns'}")
    print(f"  image_only vs genomic_only: p = {p_img_gen:.4f}  {'*' if p_img_gen < 0.05 else 'ns'}")
else:
    print("  Not enough data yet.")
    p_mm_img = p_mm_gen = p_img_gen = 1.0

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = {
    "multimodal":   "#2ecc71",
    "image_only":   "#3498db",
    "genomic_only": "#e67e22",
}
LABELS = {
    "multimodal":   "Multimodal\n(CT + RNA-seq)",
    "image_only":   "CT Only",
    "genomic_only": "RNA-seq Only",
}

fig, ax = plt.subplots(figsize=(8, 6))

data   = [mm, img, gen]
keys   = ["multimodal", "image_only", "genomic_only"]
colors = [COLORS[k] for k in keys]
labels = [LABELS[k] for k in keys]

bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=5, alpha=0.5))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Overlay individual points
np.random.seed(42)
for i, (vals, color) in enumerate(zip(data, colors), start=1):
    jitter = np.random.uniform(-0.12, 0.12, len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals,
               color=color, edgecolors="white", s=28, zorder=5, linewidth=0.5)

# Significance brackets
def sig_bracket(ax, x1, x2, y, p):
    if p >= 0.05:
        label = "ns"
    elif p < 0.001:
        label = "***"
    elif p < 0.01:
        label = "**"
    else:
        label = "*"
    h = 0.012
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color="black")
    ax.text((x1+x2)/2, y+h+0.004, label, ha="center", va="bottom", fontsize=10)

y_base = max(mm.max(), img.max(), gen.max()) + 0.04
sig_bracket(ax, 1, 2, y_base,          p_mm_img)
sig_bracket(ax, 1, 3, y_base + 0.06,   p_mm_gen)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Val ROC-AUC", fontsize=12)
ax.set_ylim(0.2, y_base + 0.14)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random (AUC=0.5)")
ax.set_title(
    "Ablation Study: LUAD Staging Performance\n"
    f"5 seeds × 5-fold cohort-stratified CV  (n=160 patients, 3 cohorts)",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Mean ± std annotations inside boxes
for i, (vals, color) in enumerate(zip(data, colors), start=1):
    ax.text(i, np.mean(vals), f"{np.mean(vals):.3f}",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.85, linewidth=0))

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUTPUT}")
