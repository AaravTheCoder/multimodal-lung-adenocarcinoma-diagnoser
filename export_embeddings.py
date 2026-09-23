import os
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from neuroCombat import neuroCombat

print("Preparing 512-D patient embeddings (TCGA-LUAD + CPTAC-LUAD + Stanford)...")

# ── 1. Load genomic matrix (patient_id indexed) ───────────────────────────────
genomic_df = pd.read_csv("genomic_matrix.csv", index_col="patient_id")
print(f"Genomic matrix: {genomic_df.shape[0]} patients × {genomic_df.shape[1]} genes")

# ── 2. Load combined clinical labels ─────────────────────────────────────────
tcga_clinical     = pd.read_csv("clinical_manifest.csv")[["patient_id", "has_condition"]]
cptac_clinical    = pd.read_csv("cptac_luad_clinical.csv")[["patient_id", "has_condition"]]
stanford_clinical = pd.read_csv("stanford_clinical.csv")[["patient_id", "has_condition"]]
all_clinical   = pd.concat([tcga_clinical, cptac_clinical, stanford_clinical], ignore_index=True)
all_clinical   = all_clinical.drop_duplicates("patient_id").set_index("patient_id")
print(f"Clinical labels: {len(all_clinical)} patients")

# ── 3. Keep only patients present in BOTH genomic matrix AND clinical labels ──
common = genomic_df.index.intersection(all_clinical.index)
genomic_df    = genomic_df.loc[common]
all_clinical  = all_clinical.loc[common]
print(f"Matched patients (genomic ∩ clinical): {len(common)}")

label_counts = all_clinical["has_condition"].value_counts().sort_index()
print(f"  Stage I/II  (label=0): {label_counts.get(0, 0)}")
print(f"  Stage III/IV (label=1): {label_counts.get(1, 0)}")

# ── 4. Filter zero-variance genes, log-transform, scale ──────────────────────
X_raw = genomic_df.values.astype(np.float32)
variance_mask = np.std(X_raw, axis=0) > 0
X_filtered = X_raw[:, variance_mask]
print(f"\nGenes after variance filter: {X_filtered.shape[1]}")

X_log = np.log2(X_filtered + 1)

# ── 4b. ComBat batch correction across cohorts ────────────────────────────────
# Assigns each patient to their source cohort so neuroCombat can remove
# cohort-specific mean/variance shifts before PCA compression.
batch_labels = []
for pid in common:
    if str(pid).startswith("TCGA"):
        batch_labels.append(0)
    elif str(pid).startswith("C3"):
        batch_labels.append(1)
    else:  # R01 Stanford
        batch_labels.append(2)

batch_series = pd.Series(batch_labels, name="batch")
print(f"\nComBat batch sizes: TCGA={batch_labels.count(0)}, "
      f"CPTAC={batch_labels.count(1)}, Stanford={batch_labels.count(2)}")

# neuroCombat expects genes×patients (transpose), returns dict.
# Pass stage label as covariate so ComBat doesn't remove stage-correlated
# gene expression variation (supervised ComBat).
stage_labels = all_clinical.loc[common, "has_condition"].astype(int).tolist()
combat_out = neuroCombat(
    dat=X_log.T,
    covars=pd.DataFrame({"batch": batch_labels, "stage": stage_labels}),
    batch_col="batch",
    categorical_cols=["stage"],
)
X_corrected = combat_out["data"].T  # back to patients×genes
X_scaled = StandardScaler().fit_transform(X_corrected)
print("ComBat correction applied.")

# ── 5a. Save pre-PCA (post-ComBat, post-scaling) matrix for per-fold PCA ─────
# The main training script loads this to fit PCA on training patients only
# per fold, avoiding data leakage from validation patients into the PCA axes.
np.save("combat_corrected_matrix.npy", X_scaled)
np.save("combat_patient_ids.npy", np.array(list(common), dtype=str))
print(f"Saved combat_corrected_matrix.npy ({X_scaled.shape}) and combat_patient_ids.npy")

# ── 5b. PCA to 512 dimensions (global, for reference / legacy use) ────────────
_max_components = min(512, X_scaled.shape[0] - 1, X_scaled.shape[1])
n_components = (_max_components // 16) * 16  # must be divisible by NUM_GENOMIC_TOKENS=16
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance retained at 512-D: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

# ── 6. Export one .pt tensor per patient ─────────────────────────────────────
EMBED_DIR = "genomic_embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)

patient_rows = []
for i, patient_id in enumerate(common):
    tensor_path = os.path.join(EMBED_DIR, f"{patient_id}.pt")
    torch.save(torch.FloatTensor(X_pca[i]), tensor_path)
    patient_rows.append({
        "patient_id":    patient_id,
        "image_path":    "placeholder_image.png",
        "has_condition": int(all_clinical.loc[patient_id, "has_condition"]),
    })

master_csv = pd.DataFrame(patient_rows)
master_csv.to_csv("combined_patient_data.csv", index=False)

tcga_count     = sum(1 for p in common if str(p).startswith("TCGA"))
cptac_count    = sum(1 for p in common if str(p).startswith("C3"))
stanford_count = sum(1 for p in common if str(p).startswith("R01"))
print(f"\nEmbeddings saved: {len(patient_rows)} patients")
print(f"  TCGA-LUAD:  {tcga_count}")
print(f"  CPTAC-LUAD: {cptac_count}")
print(f"  Stanford:   {stanford_count}")
print(f"Saved combined_patient_data.csv")
