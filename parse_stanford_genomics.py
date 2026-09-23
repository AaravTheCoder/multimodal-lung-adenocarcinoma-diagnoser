import gzip
import pandas as pd
import numpy as np

print("Parsing Stanford NSCLC-Radiogenomics RNA-seq (GSE103584)...")

stanford_clinical = pd.read_csv("stanford_clinical.csv")
stanford_ids = set(stanford_clinical["patient_id"].tolist())

# Load existing genomic matrix
existing = pd.read_csv("genomic_matrix.csv", index_col="patient_id")
print(f"Existing genomic matrix: {existing.shape[0]} patients × {existing.shape[1]} genes")

# Parse GEO RNA-seq file (genes as rows, patients as columns)
with gzip.open("GSE103584_R01_NSCLC_RNAseq.txt.gz", "rt") as f:
    header = f.readline().strip().split("\t")
    patient_ids = header[1:]  # first col is gene name

    # Filter to only our 95 LUAD patients
    keep_cols = [i for i, pid in enumerate(patient_ids) if pid in stanford_ids]
    keep_patients = [patient_ids[i] for i in keep_cols]
    print(f"Stanford patients in RNA-seq file: {len(keep_patients)}")

    gene_names = []
    expression = {pid: [] for pid in keep_patients}

    for line in f:
        parts = line.strip().split("\t")
        gene = parts[0]
        if gene.startswith("_"):
            continue
        gene_names.append(gene)
        for i, pid in zip(keep_cols, keep_patients):
            val = parts[i + 1]
            expression[pid].append(float(val) if val != 'NA' else 0.0)

print(f"Genes parsed: {len(gene_names)}")

# Build dataframe
stanford_df = pd.DataFrame.from_dict(expression, orient="index", columns=gene_names)
stanford_df.index.name = "patient_id"
print(f"Stanford genomic matrix: {stanford_df.shape}")

# Find common genes with existing matrix
common_genes = existing.columns.intersection(stanford_df.columns)
print(f"Common genes with existing matrix: {len(common_genes)}")

# Align to common genes and append
existing_aligned   = existing[common_genes]
stanford_aligned   = stanford_df[common_genes]

# Remove any Stanford patients already in existing matrix
new_patients = [p for p in stanford_aligned.index if p not in existing.index]
print(f"New Stanford patients to add: {len(new_patients)}")

combined = pd.concat([existing_aligned, stanford_aligned.loc[new_patients]], axis=0)
combined.index.name = "patient_id"
combined.to_csv("genomic_matrix.csv")

print(f"\nSaved genomic_matrix.csv: {combined.shape[0]} patients × {combined.shape[1]} genes")
tcga  = sum(1 for p in combined.index if str(p).startswith("TCGA"))
cptac = sum(1 for p in combined.index if str(p).startswith("C3"))
stan  = sum(1 for p in combined.index if str(p).startswith("R01"))
print(f"  TCGA-LUAD:   {tcga}")
print(f"  CPTAC-LUAD:  {cptac}")
print(f"  Stanford:    {stan}")
print("\nNext: run python3 export_embeddings.py")
