"""
IDC TCGA-LUAD audit — metadata only, no downloads.

Produces the intersection counts:
  IDC CT patients ∩ GDC RNA-seq ∩ stage label

and identifies NEW patients beyond our existing 29.
"""
import os, json, requests, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── 1. IDC: enumerate TCGA-LUAD subjects that have CT ─────────────────────────
print("=" * 65)
print("STEP 1  IDC TCGA-LUAD — subjects with CT")
print("=" * 65)

from idc_index import index as idc_index

idc = idc_index.IDCClient()

# Query IDC for TCGA-LUAD collection, CT modality
idc_df = idc.get_filtered_idc_dataframe(
    collection_id="tcga_luad",
    modality_values=["CT"],
)

idc_subjects = set(idc_df["PatientID"].unique())
print(f"  IDC TCGA-LUAD subjects with CT: {len(idc_subjects)}")

# ── 2. GDC: patients with open-access RNA-seq (STAR Counts) ───────────────────
print("\n" + "=" * 65)
print("STEP 2  GDC — TCGA-LUAD open-access RNA-seq patients")
print("=" * 65)

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
params = {
    "filters": json.dumps({
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-LUAD"}},
            {"op": "=", "content": {"field": "data_category",            "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type",                "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "experimental_strategy",    "value": "RNA-Seq"}},
            {"op": "=", "content": {"field": "access",                   "value": "open"}},
        ]
    }),
    "fields": "cases.submitter_id",
    "size": "2000",
    "format": "json",
}

resp = requests.get(GDC_FILES_URL, params=params, timeout=60)
resp.raise_for_status()
hits = resp.json()["data"]["hits"]

gdc_patients = set()
for hit in hits:
    for case in hit.get("cases", []):
        sid = case.get("submitter_id", "")
        if sid:
            gdc_patients.add(sid)

print(f"  GDC TCGA-LUAD open RNA-seq patients: {len(gdc_patients)}")

# ── 3. Stage labels from our clinical manifest ────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3  Stage labels (from clinical_manifest.csv)")
print("=" * 65)

clin = pd.read_csv("clinical_manifest.csv")
# Normalize patient IDs to TCGA-XX-XXXX format (first 12 chars)
clin["case_id"] = clin["patient_id"].str[:12]
staged_patients = set(clin["case_id"].unique())
print(f"  Patients with stage labels: {len(staged_patients)}")

# ── 4. Intersections ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4  Intersection audit")
print("=" * 65)

# Normalize IDC and GDC IDs to TCGA-XX-XXXX (first 12 chars)
idc_norm  = set(s[:12] for s in idc_subjects)
gdc_norm  = set(s[:12] for s in gdc_patients)
stage_norm = staged_patients

ct_rna        = idc_norm & gdc_norm
ct_stage      = idc_norm & stage_norm
rna_stage     = gdc_norm & stage_norm
ct_rna_stage  = idc_norm & gdc_norm & stage_norm

print(f"  IDC CT patients:               {len(idc_norm)}")
print(f"  GDC RNA-seq patients:          {len(gdc_norm)}")
print(f"  Patients with stage labels:    {len(stage_norm)}")
print(f"  CT ∩ RNA-seq:                  {len(ct_rna)}")
print(f"  CT ∩ stage:                    {len(ct_stage)}")
print(f"  RNA-seq ∩ stage:               {len(rna_stage)}")
print(f"  CT ∩ RNA-seq ∩ stage:          {len(ct_rna_stage)}  ← multimodal candidates")

# ── 5. Compare against our existing 29 patients ───────────────────────────────
print("\n" + "=" * 65)
print("STEP 5  New patients beyond our existing 29")
print("=" * 65)

real = pd.read_csv("real_patient_data.csv")
existing_tcga = set(real[real["patient_id"].str.startswith("TCGA")]["patient_id"].str[:12].unique())
print(f"  Our existing TCGA CT patients: {len(existing_tcga)}")

new_candidates = ct_rna_stage - existing_tcga
print(f"  NEW multimodal candidates:     {len(new_candidates)}")

if new_candidates:
    new_list = sorted(new_candidates)
    print(f"\n  First 20 new patient IDs:")
    for pid in new_list[:20]:
        print(f"    {pid}")
    pd.Series(sorted(new_candidates)).to_csv("new_idc_candidates.csv", index=False, header=["patient_id"])
    print(f"\n  Full list saved → new_idc_candidates.csv")

# ── 6. Estimate download size for new patients ────────────────────────────────
if new_candidates and len(new_candidates) > 0:
    print("\n" + "=" * 65)
    print("STEP 6  Download size estimate for new patients")
    print("=" * 65)

    # Sample a few to get typical CT series size
    sample_ids = list(new_candidates)[:5]
    sample_df = idc_df[idc_df["PatientID"].str[:12].isin(sample_ids)]
    if "series_size_MB" in sample_df.columns:
        avg_mb = sample_df.groupby("PatientID")["series_size_MB"].sum().mean()
        total_gb = avg_mb * len(new_candidates) / 1024
        print(f"  Avg CT size per patient: {avg_mb:.0f} MB")
        print(f"  Estimated total for {len(new_candidates)} new patients: {total_gb:.1f} GB")
    else:
        print(f"  (Size column not available in this idc-index version)")
        print(f"  {len(new_candidates)} new patients — typical TCGA-LUAD CT ≈ 300–500 MB each")
        est_gb = len(new_candidates) * 400 / 1024
        print(f"  Rough estimate: ~{est_gb:.0f} GB")

print("\n" + "=" * 65)
print("AUDIT COMPLETE")
print("=" * 65)
