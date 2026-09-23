import os
import gzip
import pandas as pd
import numpy as np

DATA_DIR   = "real_genomics"
OUTPUT_FILE = "genomic_matrix.csv"

print("Parsing GDC downloaded genomic files (TCGA-LUAD + CPTAC-LUAD)...")

# ── Build UUID → patient_id map from both cohorts ────────────────────────────

# TCGA: read the original GDC manifest + clinical manifest
tcga_manifest = pd.read_csv("gdc_manifest.2026-07-15.151322.txt", sep="\t")
tcga_clinical  = pd.read_csv("clinical_manifest.csv")

# Build UUID→patient_id from GDC manifest directly (positional alignment)
n = min(len(tcga_clinical), len(tcga_manifest))
tcga_uuid_map = dict(zip(tcga_manifest["id"].iloc[:n],
                          tcga_clinical["patient_id"].iloc[:n]))

# CPTAC: explicit uuid→patient map we built from GDC API
cptac_map_df   = pd.read_csv("cptac_uuid_to_patient.csv")
cptac_uuid_map = dict(zip(cptac_map_df["file_uuid"], cptac_map_df["patient_id"]))

# Combined map: UUID → patient_id
uuid_to_patient = {**tcga_uuid_map, **cptac_uuid_map}
print(f"UUID→patient map: {len(tcga_uuid_map)} TCGA + {len(cptac_uuid_map)} CPTAC = {len(uuid_to_patient)} total")

# ── Walk real_genomics/ and parse TSV files ──────────────────────────────────
expression_data = {}  # patient_id → numpy array
all_genes = []
files_processed = 0
skipped_no_map = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if not (file.endswith(".tsv") or file.endswith(".tsv.gz")):
            continue

        file_uuid = os.path.basename(root)
        if file_uuid == DATA_DIR:
            continue

        patient_id = uuid_to_patient.get(file_uuid)
        if patient_id is None:
            skipped_no_map += 1
            continue

        # Skip if we already have this patient (e.g. duplicate aliquots)
        if patient_id in expression_data:
            continue

        file_path = os.path.join(root, file)
        try:
            if file.endswith(".gz"):
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f, sep="\t", comment="#")
            else:
                df = pd.read_csv(file_path, sep="\t", comment="#")

            if "gene_id" not in df.columns:
                continue

            df = df[~df["gene_id"].str.startswith("_")]
            df = df.dropna(subset=["gene_name"])
            df = df[df["gene_name"].str.strip() != ""]

            if len(all_genes) == 0:
                all_genes = df["gene_name"].tolist()

            val_col = "tpm_unstranded" if "tpm_unstranded" in df.columns else df.columns[3]
            expression_data[patient_id] = df[val_col].values

            files_processed += 1
            if files_processed % 50 == 0:
                print(f"  Processed {files_processed} files...")

        except Exception:
            pass

print(f"\nParsed {files_processed} files  |  skipped {skipped_no_map} (no patient mapping)")
print(f"Patients with expression data: {len(expression_data)}")

if not expression_data:
    print("No valid genomic files found.")
    exit()

# ── Build patient-indexed matrix ─────────────────────────────────────────────
genomic_df = pd.DataFrame.from_dict(expression_data, orient='index', columns=all_genes)
genomic_df.index.name = "patient_id"

tcga_count  = sum(1 for p in genomic_df.index if p.startswith("TCGA"))
cptac_count = sum(1 for p in genomic_df.index if p.startswith("C3"))
print(f"  TCGA-LUAD patients: {tcga_count}")
print(f"  CPTAC-LUAD patients: {cptac_count}")

genomic_df.to_csv(OUTPUT_FILE)
print(f"\nSaved {OUTPUT_FILE}  shape: {genomic_df.shape}")
