"""
Gene loading analysis: which genes drive the PCA components used in staging prediction?

Fits PCA on the ComBat-corrected RNA-seq matrix (same setup as per-fold PCA in training,
but on the full dataset for visualization). Projects PC loadings back to gene space and
plots the top-weighted genes for PC1 and PC2, highlighting known LUAD driver genes.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config ────────────────────────────────────────────────────────────────────
COMBAT_MATRIX  = "combat_corrected_matrix.npy"
COMBAT_PIDS    = "combat_patient_ids.npy"
GENOMIC_MATRIX = "genomic_matrix.csv"
N_COMPONENTS   = 10        # compute top-10 PCs; plot PC1 and PC2
TOP_N_GENES    = 20        # genes to show per PC
OUTPUT_FILE    = "gene_loadings.png"

# Known LUAD driver genes to highlight (TCGA / COSMIC literature)
LUAD_DRIVERS = {
    "KRAS", "EGFR", "TP53", "STK11", "KEAP1", "NF1",
    "RBM10", "BRAF", "MET", "ERBB2", "PIK3CA", "RET",
    "ALK", "ROS1", "NTRK1", "CDKN2A", "SMAD4", "ARID1A",
    "U2AF1", "SETD2", "RB1", "PTEN",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading ComBat-corrected matrix …")
X      = np.load(COMBAT_MATRIX)          # (n_patients, n_genes)
pids   = np.load(COMBAT_PIDS, allow_pickle=True).tolist()

print("Loading gene names from genomic_matrix.csv …")
gene_df    = pd.read_csv(GENOMIC_MATRIX)
all_gene_cols = [c for c in gene_df.columns if c != "patient_id"]

# ComBat drops zero-variance genes before correction — reconstruct the same filter.
combat_pids_set = set(np.load(COMBAT_PIDS, allow_pickle=True).tolist())
matched = gene_df[gene_df["patient_id"].isin(combat_pids_set)]
X_raw   = matched[all_gene_cols].values
stds    = X_raw.std(axis=0)
gene_names = np.array(all_gene_cols)[stds > 0]

assert X.shape[1] == len(gene_names), (
    f"Mismatch after filtering: matrix has {X.shape[1]} cols but {len(gene_names)} gene names"
)
print(f"Matrix: {X.shape[0]} patients × {X.shape[1]} genes "
      f"({(stds == 0).sum()} zero-variance genes excluded)")

# ── PCA ───────────────────────────────────────────────────────────────────────
print(f"Fitting PCA (n_components={N_COMPONENTS}) …")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_COMPONENTS, random_state=42)
pca.fit(X_scaled)

var_exp = pca.explained_variance_ratio_ * 100
print("Variance explained per PC:")
for i, v in enumerate(var_exp):
    print(f"  PC{i+1}: {v:.2f}%")

gene_names = np.array(gene_names)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "Gene Loadings on Top PCA Components\n"
    "(ComBat-corrected LUAD RNA-seq, n=520 patients, 18 623 genes)",
    fontsize=13, fontweight="bold", y=1.01
)

DRIVER_COLOR  = "#e74c3c"   # red for driver genes
DEFAULT_COLOR = "#3498db"   # blue for others
DRIVER_LABEL  = "Known LUAD driver gene"
DEFAULT_LABEL = "Other gene"

for ax_idx, pc_idx in enumerate([0, 1]):
    loadings = pca.components_[pc_idx]          # signed loading per gene
    abs_load = np.abs(loadings)
    top_idx  = np.argsort(abs_load)[::-1][:TOP_N_GENES]

    top_genes = gene_names[top_idx]
    top_loads = loadings[top_idx]               # keep sign
    colors    = [DRIVER_COLOR if g in LUAD_DRIVERS else DEFAULT_COLOR
                 for g in top_genes]

    ax = axes[ax_idx]
    bars = ax.barh(range(TOP_N_GENES), top_loads[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.4)

    ax.set_yticks(range(TOP_N_GENES))
    ax.set_yticklabels(top_genes[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Loading weight", fontsize=10)
    ax.set_title(
        f"PC{pc_idx+1}  ({var_exp[pc_idx]:.1f}% variance)",
        fontsize=11, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bold + annotate known drivers
    for label_obj, gene in zip(ax.get_yticklabels(), top_genes[::-1]):
        if gene in LUAD_DRIVERS:
            label_obj.set_fontweight("bold")
            label_obj.set_color(DRIVER_COLOR)

# Legend
patches = [
    mpatches.Patch(color=DRIVER_COLOR, label=DRIVER_LABEL),
    mpatches.Patch(color=DEFAULT_COLOR, label=DEFAULT_LABEL),
]
fig.legend(handles=patches, loc="lower center", ncol=2,
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUTPUT_FILE}")

# ── Text summary ──────────────────────────────────────────────────────────────
print("\n── Top genes by PC ──")
for pc_idx in range(min(3, N_COMPONENTS)):
    loadings = pca.components_[pc_idx]
    top_idx  = np.argsort(np.abs(loadings))[::-1][:10]
    top_genes = gene_names[top_idx]
    driver_hits = [g for g in top_genes if g in LUAD_DRIVERS]
    print(f"PC{pc_idx+1} ({var_exp[pc_idx]:.1f}%): {', '.join(top_genes[:10])}")
    if driver_hits:
        print(f"  ★ LUAD drivers in top 10: {', '.join(driver_hits)}")
