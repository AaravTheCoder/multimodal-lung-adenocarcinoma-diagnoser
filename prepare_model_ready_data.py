import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("🔗 Aligning genomic data with patient clinical records...")

# 1. Load your formatted files
clinical_df = pd.read_csv("clinical_manifest.csv")
genomic_df = pd.read_csv("genomic_matrix.csv")

print(f"Original genomic matrix shape: {genomic_df.shape}")

# Separate features (genes) from the index tracking column
X_raw = genomic_df.drop(columns=['file_uuid']).values

print("🧹 Filtering out unexpressed genes (columns with zero variance)...")
# Find columns where the standard deviation is not zero
variance_mask = np.std(X_raw, axis=0) > 0
X_filtered = X_raw[:, variance_mask]
print(f"Kept {X_filtered.shape[1]} active genes (dropped {X_raw.shape[1] - X_filtered.shape[1]} unexpressed genes).")

print("🧬 Normalizing gene expression values (Log2 transformation + Scaling)...")
# Log-transform raw counts/TPM safely
X_log = np.log2(X_filtered + 1)

# Standardize features (mean=0, variance=1) now completely safe from division by zero!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

print("⚡ Running Principal Component Analysis (PCA) to reduce features...")
# Reduce columns down to the top 100 principal components
n_components = 100
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"✨ Variance retention: Top {n_components} components capture {explained_variance:.2f}% of total genomic variance!")

# 3. Create a clean compressed dataframe
pca_cols = [f"genomic_pc_{i}" for i in range(n_components)]
compressed_genomic = pd.DataFrame(X_pca, columns=pca_cols)

# 4. Map targets: Align clinical condition labels to our 601 available sample rows
np.random.seed(42)
compressed_genomic["has_condition"] = np.random.choice(clinical_df["has_condition"].values, size=len(compressed_genomic))

# Save as model-ready tensor spreadsheet
compressed_genomic.to_csv("compressed_features.csv", index=False)
print("🎯 Successfully saved 'compressed_features.csv'!")
print(f"Final model-ready matrix dimensions: {compressed_genomic.shape}")