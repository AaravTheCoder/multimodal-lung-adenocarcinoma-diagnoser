"""
Fetches clinical stage labels and RNA-seq file UUIDs for the 38 TCGA-17 patients
that have CT scans but no genomic embeddings.
Outputs:
  - tcga17_clinical.csv   (patient_id, has_condition)
  - tcga17_manifest.txt   (gdc-client manifest for RNA-seq downloads)
"""

import requests
import json
import csv

GDC_API = "https://api.gdc.cancer.gov"

MISSING_PATIENTS = [
    "TCGA-17-Z011", "TCGA-17-Z013", "TCGA-17-Z015", "TCGA-17-Z016",
    "TCGA-17-Z017", "TCGA-17-Z018", "TCGA-17-Z019", "TCGA-17-Z020",
    "TCGA-17-Z021", "TCGA-17-Z023", "TCGA-17-Z024", "TCGA-17-Z027",
    "TCGA-17-Z028", "TCGA-17-Z029", "TCGA-17-Z030", "TCGA-17-Z031",
    "TCGA-17-Z032", "TCGA-17-Z033", "TCGA-17-Z034", "TCGA-17-Z035",
    "TCGA-17-Z036", "TCGA-17-Z038", "TCGA-17-Z039", "TCGA-17-Z042",
    "TCGA-17-Z043", "TCGA-17-Z045", "TCGA-17-Z048", "TCGA-17-Z050",
    "TCGA-17-Z051", "TCGA-17-Z052", "TCGA-17-Z053", "TCGA-17-Z054",
    "TCGA-17-Z056", "TCGA-17-Z058", "TCGA-17-Z059", "TCGA-17-Z060",
    "TCGA-17-Z061", "TCGA-17-Z062",
]

STAGE_MAP = {
    "stage i":   0, "stage ia": 0, "stage ib": 0,
    "stage ii":  0, "stage iia": 0, "stage iib": 0,
    "stage iii": 1, "stage iiia": 1, "stage iiib": 1,
    "stage iv":  1,
}

# ── Step 1: Fetch clinical staging ───────────────────────────────────────────
print("Querying GDC API for clinical staging data...")

clinical_payload = {
    "filters": {
        "op": "in",
        "content": {
            "field": "submitter_id",
            "value": MISSING_PATIENTS
        }
    },
    "fields": "submitter_id,diagnoses.ajcc_pathologic_stage,diagnoses.ajcc_pathologic_t",
    "format": "json",
    "size": 100
}

resp = requests.post(f"{GDC_API}/cases", json=clinical_payload, timeout=30)
resp.raise_for_status()
data = resp.json()["data"]["hits"]

clinical_rows = []
no_stage = []

for case in data:
    pid = case.get("submitter_id", "")
    stage_raw = ""
    diagnoses = case.get("diagnoses", [])
    if diagnoses:
        stage_raw = diagnoses[0].get("ajcc_pathologic_stage", "") or ""

    stage_key = stage_raw.lower().strip()
    label = STAGE_MAP.get(stage_key, None)

    if label is not None:
        clinical_rows.append({"patient_id": pid, "has_condition": label, "stage": stage_raw})
    else:
        no_stage.append((pid, stage_raw))

print(f"  Found staging for {len(clinical_rows)} patients")
if no_stage:
    print(f"  Could not map stage for {len(no_stage)} patients: {no_stage}")

# ── Step 2: Fetch RNA-seq file UUIDs ─────────────────────────────────────────
print("\nQuerying GDC API for RNA-seq file UUIDs...")

stageable_patients = [r["patient_id"] for r in clinical_rows]

files_payload = {
    "filters": {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": stageable_patients
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Gene Expression Quantification"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "experimental_strategy",
                    "value": "RNA-Seq"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": "STAR - Counts"
                }
            }
        ]
    },
    "fields": "file_id,file_name,cases.submitter_id,md5sum,file_size,state",
    "format": "json",
    "size": 200
}

resp2 = requests.post(f"{GDC_API}/files", json=files_payload, timeout=30)
resp2.raise_for_status()
files_data = resp2.json()["data"]["hits"]

print(f"  Found {len(files_data)} RNA-seq files")

# ── Step 3: Save clinical CSV ─────────────────────────────────────────────────
with open("tcga17_clinical.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["patient_id", "has_condition", "stage"])
    writer.writeheader()
    writer.writerows(clinical_rows)
print(f"\nSaved tcga17_clinical.csv ({len(clinical_rows)} rows)")

# ── Step 4: Save GDC manifest ─────────────────────────────────────────────────
if files_data:
    with open("tcga17_manifest.txt", "w", newline="") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for file in files_data:
            f.write(f"{file['file_id']}\t{file['file_name']}\t{file.get('md5sum','')}\t{file.get('file_size',0)}\t{file.get('state','')}\n")
    print(f"Saved tcga17_manifest.txt ({len(files_data)} files)")
    print("\nNext step: run this command to download the RNA-seq files:")
    print(f"  ./gdc-client download -m tcga17_manifest.txt -d real_genomics/")
else:
    print("WARNING: No RNA-seq files found — check if the patients are available in GDC open access.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Patients with CT scans + stage labels: {len(clinical_rows)}")
label_counts = {}
for r in clinical_rows:
    label_counts[r['has_condition']] = label_counts.get(r['has_condition'], 0) + 1
for label, count in sorted(label_counts.items()):
    name = "Stage I/II (0)" if label == 0 else "Stage III/IV (1)"
    print(f"  {name}: {count}")
