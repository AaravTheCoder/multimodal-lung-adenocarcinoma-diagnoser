"""
Audits every possible matched multimodal patient across:
  - Genomic embeddings (518 patients in genomic_matrix.csv)
  - TCIA CT scans (from downloaded DICOM metadata)
  - Clinical labels (clinical_manifest.csv)

Outputs MULTIMODAL_PATIENTS.csv with a status column for each modality.
"""

import os
import pandas as pd
import requests
import json

print("=" * 60)
print("MULTIMODAL PATIENT AUDIT")
print("=" * 60)

# ── 1. Load genomic patients ──────────────────────────────────────
genomic_df = pd.read_csv("genomic_matrix.csv", usecols=[0])  # just the index column
genomic_patients = set(genomic_df.iloc[:, 0].tolist())
print(f"\n[Genomics] Patients in genomic_matrix.csv: {len(genomic_patients)}")

# Also check individual .pt embedding files
embed_dir = "genomic_embeddings"
embedded_patients = set()
for f in os.listdir(embed_dir):
    if f.endswith(".pt") and f.startswith("TCGA-"):
        embedded_patients.add(f.replace(".pt", ""))
print(f"[Genomics] .pt embedding files (TCGA patients): {len(embedded_patients)}")

# ── 2. Load TCIA CT patients ──────────────────────────────────────
meta = pd.read_csv("tcia_raw_dicom/manifest-1Rd7jPNd5199284876140322680/metadata.csv")
ct_only = meta[meta["Modality"] == "CT"]
tcia_patients = set(ct_only["Subject ID"].unique())
print(f"\n[Imaging] TCIA TCGA-LUAD CT patients downloaded: {len(tcia_patients)}")

# ── 3. Load clinical labels ───────────────────────────────────────
clinical_df = pd.read_csv("clinical_manifest.csv")
clinical_patients = set(clinical_df["patient_id"].tolist())
print(f"\n[Clinical] Patients with stage labels: {len(clinical_patients)}")

# ── 4. Query GDC for ALL TCGA-LUAD RNA-seq patients (not just downloaded ones) ──
print("\n[GDC] Querying ALL TCGA-LUAD RNA-seq patients via GDC API...")
gdc_payload = {
    "filters": {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD"]}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "RNA-Seq"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "access", "value": "open"}}
        ]
    },
    "fields": "cases.submitter_id,file_id,file_name,file_size",
    "format": "json",
    "size": 1000
}
try:
    resp = requests.post("https://api.gdc.cancer.gov/files", json=gdc_payload, timeout=30)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]
    gdc_rnaseq_patients = set()
    gdc_file_map = {}  # patient_id -> (file_id, file_name, file_size)
    for h in hits:
        cases = h.get("cases", [])
        if cases:
            pid = cases[0]["submitter_id"]
            gdc_rnaseq_patients.add(pid)
            gdc_file_map[pid] = (h["file_id"], h["file_name"], h.get("file_size", 0))
    print(f"[GDC] TCGA-LUAD open RNA-seq patients on GDC: {len(gdc_rnaseq_patients)}")
except Exception as e:
    print(f"[GDC] API call failed: {e}. Using local genomic data only.")
    gdc_rnaseq_patients = embedded_patients
    gdc_file_map = {}

# ── 5. Compute all intersections ──────────────────────────────────
all_patients = tcia_patients | gdc_rnaseq_patients | clinical_patients

rows = []
for pid in sorted(all_patients):
    has_ct       = pid in tcia_patients
    has_rnaseq   = pid in gdc_rnaseq_patients
    has_embedding = pid in embedded_patients
    has_label    = pid in clinical_patients

    # Already downloaded and processed?
    already_processed = os.path.exists(f"processed_ct_images/{pid}_slice0.png")

    # Usable = all three modalities + label
    usable = has_ct and (has_rnaseq or has_embedding) and has_label

    rows.append({
        "patient_id":      pid,
        "CT_scan":         "✅" if has_ct else "❌",
        "RNA_seq_GDC":     "✅" if has_rnaseq else "❌",
        "embedding_file":  "✅" if has_embedding else "❌",
        "stage_label":     "✅" if has_label else "❌",
        "already_in_model": "✅" if already_processed else "❌",
        "USABLE":          "✅" if usable else "❌",
        "_usable_bool":    usable,
        "_has_ct":         has_ct,
        "_has_rnaseq":     has_rnaseq,
        "_has_embedding":  has_embedding,
        "_has_label":      has_label,
        "_need_rnaseq_download": has_ct and has_label and has_rnaseq and not has_embedding,
    })

df_out = pd.DataFrame(rows)

# ── 6. Save and print ─────────────────────────────────────────────
display_cols = ["patient_id", "CT_scan", "RNA_seq_GDC", "embedding_file", "stage_label", "already_in_model", "USABLE"]
df_out[display_cols].to_csv("MULTIMODAL_PATIENTS.csv", index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total TCIA CT patients:                {len(tcia_patients)}")
print(f"Total GDC RNA-seq patients (open):     {len(gdc_rnaseq_patients)}")
print(f"Total with clinical labels:            {len(clinical_patients)}")
print(f"Currently in model (processed):        {len([r for r in rows if r['already_in_model'] == '✅'])}")
print()

usable = [r for r in rows if r["_usable_bool"]]
print(f"TOTAL USABLE (CT + genomic + label):   {len(usable)}")

need_download = [r for r in rows if r["_need_rnaseq_download"]]
print(f"New patients (need RNA-seq download):  {len(need_download)}")

if need_download:
    print("\nPatients that have CT + label but need RNA-seq download:")
    total_bytes = 0
    for r in need_download:
        pid = r["patient_id"]
        if pid in gdc_file_map:
            fid, fname, fsize = gdc_file_map[pid]
            total_bytes += fsize
            print(f"  {pid}  →  {fname}  ({fsize/1e6:.1f} MB)")
    if total_bytes > 0:
        print(f"\n  Estimated download size: {total_bytes/1e9:.2f} GB")

    # Generate a manifest for gdc-client
    with open("new_patients_manifest.txt", "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for r in need_download:
            pid = r["patient_id"]
            if pid in gdc_file_map:
                fid, fname, fsize = gdc_file_map[pid]
                f.write(f"{fid}\t{fname}\t\t{fsize}\tsubmitted\n")
    print(f"\nSaved new_patients_manifest.txt — ready to download with:")
    print(f"  ./gdc-client download -m new_patients_manifest.txt -d real_genomics/")

ct_no_genomic = [r for r in rows if r["_has_ct"] and not r["_has_rnaseq"] and not r["_has_embedding"]]
print(f"\nCT patients with NO open RNA-seq on GDC: {len(ct_no_genomic)}")
if ct_no_genomic:
    for r in ct_no_genomic[:5]:
        print(f"  {r['patient_id']}")
    if len(ct_no_genomic) > 5:
        print(f"  ... and {len(ct_no_genomic)-5} more")

print(f"\nSaved full audit to MULTIMODAL_PATIENTS.csv")
