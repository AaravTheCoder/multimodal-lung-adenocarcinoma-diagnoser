"""
Processes CPTAC-LUAD DICOM CT scans into 224x224 PNGs (5 slices per patient).
Run this AFTER the NBIA download of cptac_tcia_raw_dicom/ is complete.
Appends new patients to real_patient_data.csv alongside existing TCGA patients.
"""

import os
import glob
import pydicom
import numpy as np
import pandas as pd
from PIL import Image

print("Processing CPTAC-LUAD CT DICOM files into 224x224 PNG slices...")

CPTAC_DICOM_ROOT = "cptac_tcia_raw_dicom"
OUTPUT_DIR       = "processed_ct_images"
COMBINED_CSV     = "combined_patient_data.csv"
OUTPUT_CSV       = "real_patient_data.csv"
SLICES_PER_PATIENT = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load patients that have genomic embeddings (only process those)
combined_df = pd.read_csv(COMBINED_CSV)
patients_with_embeddings = set(combined_df["patient_id"].tolist())
cptac_patients_needed = {p for p in patients_with_embeddings if p.startswith("C3")}
print(f"CPTAC patients with embeddings: {len(cptac_patients_needed)}")

# Load existing processed patients so we don't re-process TCGA ones
existing_csv = pd.read_csv(OUTPUT_CSV) if os.path.exists(OUTPUT_CSV) else pd.DataFrame()
already_processed = set(existing_csv["patient_id"].unique()) if len(existing_csv) > 0 else set()
print(f"Already processed patients: {len(already_processed)}")

def dicom_to_png(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    pixel_array = dcm.pixel_array.astype(np.float32)
    pixel_array = np.clip(pixel_array, -1000, 400)
    norm = (
        (pixel_array - pixel_array.min())
        / (pixel_array.max() - pixel_array.min() + 1e-8)
        * 255.0
    ).astype(np.uint8)
    return Image.fromarray(norm).convert("RGB").resize((224, 224))

def find_patient_id_from_path(path):
    """Extract CPTAC patient ID (C3L-XXXXX or C3N-XXXXX) from a directory path."""
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("C3L-") or part.startswith("C3N-"):
            return part
    return None

# Walk the CPTAC DICOM root and find CT series per patient
new_rows = []
processed_patients = set()
failed_patients = []

# NBIA "Descriptive Directory Name" layout:
# cptac_tcia_raw_dicom/CPTAC-LUAD/<PatientID>/<StudyDate>/<SeriesDescription>/
for patient_dir in sorted(os.listdir(CPTAC_DICOM_ROOT)):
    patient_path = os.path.join(CPTAC_DICOM_ROOT, patient_dir)
    if not os.path.isdir(patient_path):
        continue

    # patient_dir might be the collection name; recurse one more level
    # Try to find patient ID directories inside
    candidates = []
    for root, dirs, files in os.walk(patient_path):
        pid = find_patient_id_from_path(root)
        if pid and pid in cptac_patients_needed and pid not in already_processed:
            dcm_files = [f for f in glob.glob(os.path.join(root, "*.dcm"))]
            if len(dcm_files) >= 10:
                candidates.append((pid, root, dcm_files))

    # Per patient: pick the CT series with the most slices
    patient_series = {}
    for pid, series_dir, dcm_files in candidates:
        # Check if this series is CT modality
        try:
            sample = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            if getattr(sample, 'Modality', '') != 'CT':
                continue
        except Exception:
            continue
        if pid not in patient_series or len(dcm_files) > len(patient_series[pid][1]):
            patient_series[pid] = (series_dir, dcm_files)

    for pid, (series_dir, dcm_files) in patient_series.items():
        if pid in processed_patients:
            continue

        dcm_files = sorted(dcm_files)
        n = len(dcm_files)
        start = int(n * 0.10)
        end   = int(n * 0.90)
        indices = np.linspace(start, end, SLICES_PER_PATIENT, dtype=int)

        label = combined_df.loc[combined_df["patient_id"] == pid, "has_condition"].values[0]
        saved = []
        for i, idx in enumerate(indices):
            try:
                pil_img = dicom_to_png(dcm_files[idx])
                out_path = os.path.join(OUTPUT_DIR, f"{pid}_slice{i}.png")
                pil_img.save(out_path)
                saved.append(out_path)
                new_rows.append({"patient_id": pid, "image_path": out_path, "has_condition": label})
            except Exception as e:
                print(f"  [WARN] {pid} slice {idx}: {e}")

        processed_patients.add(pid)
        print(f"  [OK] {pid} — {len(saved)} slices (label={label}, total={n} CT slices)")

if not new_rows and not processed_patients:
    print("\nNo CPTAC CT series found. Check that cptac_tcia_raw_dicom/ is populated.")
    print("If the NBIA download is still running, wait for it to finish then re-run this script.")
else:
    # Merge with existing TCGA rows
    new_df = pd.DataFrame(new_rows)
    if len(existing_csv) > 0:
        combined = pd.concat([existing_csv, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nProcessed {len(processed_patients)} new CPTAC patients → {len(new_rows)} slices")
    print(f"Updated {OUTPUT_CSV}: {len(combined)} total rows")

    # Summary
    total_patients = combined["patient_id"].nunique()
    label_counts = combined.drop_duplicates("patient_id")["has_condition"].value_counts().sort_index()
    print(f"\nFinal dataset:")
    print(f"  Total patients: {total_patients}")
    print(f"  Stage I/II  (label=0): {label_counts.get(0, 0)}")
    print(f"  Stage III/IV (label=1): {label_counts.get(1, 0)}")
    print(f"  Total image rows (5 per patient): {len(combined)}")
    print(f"\nRun: python3 multimodal_sts_model.py")
