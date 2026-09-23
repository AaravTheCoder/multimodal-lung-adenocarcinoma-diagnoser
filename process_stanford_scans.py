import os
import glob
import pydicom
import numpy as np
import pandas as pd
from PIL import Image

print("Processing Stanford NSCLC-Radiogenomics CT DICOM files into 224x224 PNG slices...")

STANFORD_DICOM_ROOT = "stanford_tcia_raw_dicom/manifest-1787806518237/NSCLC Radiogenomics"
OUTPUT_DIR          = "processed_ct_images"
COMBINED_CSV        = "combined_patient_data.csv"
OUTPUT_CSV          = "real_patient_data.csv"
SLICES_PER_PATIENT  = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

stanford_clinical = pd.read_csv("stanford_clinical.csv")
stanford_patients_needed = set(stanford_clinical["patient_id"].tolist())

existing_csv = pd.read_csv(OUTPUT_CSV) if os.path.exists(OUTPUT_CSV) else pd.DataFrame()
already_processed = set(existing_csv["patient_id"].unique()) if len(existing_csv) > 0 else set()
print(f"Stanford patients needed: {len(stanford_patients_needed)}")
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

new_rows = []
processed_patients = set()

for patient_dir in sorted(os.listdir(STANFORD_DICOM_ROOT)):
    if not patient_dir.startswith("R01-"):
        continue
    pid = patient_dir
    if pid not in stanford_patients_needed or pid in already_processed:
        continue

    patient_path = os.path.join(STANFORD_DICOM_ROOT, patient_dir)

    # Find the CT series with the most slices
    best_series = None
    best_count = 0
    for root, dirs, files in os.walk(patient_path):
        dcm_files = glob.glob(os.path.join(root, "*.dcm"))
        if len(dcm_files) < 10:
            continue
        try:
            sample = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            if getattr(sample, 'Modality', '') != 'CT':
                continue
        except Exception:
            continue
        if len(dcm_files) > best_count:
            best_count = len(dcm_files)
            best_series = (root, dcm_files)

    if best_series is None:
        print(f"  [SKIP] {pid} — no CT series with ≥10 slices found")
        continue

    series_dir, dcm_files = best_series
    dcm_files = sorted(dcm_files)
    n = len(dcm_files)
    start = int(n * 0.10)
    end   = int(n * 0.90)
    indices = np.linspace(start, end, SLICES_PER_PATIENT, dtype=int)

    label = stanford_clinical.loc[stanford_clinical["patient_id"] == pid, "has_condition"].values[0]
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

if not new_rows:
    print("\nNo Stanford CT series processed.")
else:
    new_df = pd.DataFrame(new_rows)
    if len(existing_csv) > 0:
        combined = pd.concat([existing_csv, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nProcessed {len(processed_patients)} new Stanford patients → {len(new_rows)} slices")
    print(f"Updated {OUTPUT_CSV}: {len(combined)} total rows")

    total_patients = combined["patient_id"].nunique()
    label_counts = combined.drop_duplicates("patient_id")["has_condition"].value_counts().sort_index()
    print(f"\nFinal dataset:")
    print(f"  Total patients: {total_patients}")
    print(f"  Stage I/II  (label=0): {label_counts.get(0, 0)}")
    print(f"  Stage III/IV (label=1): {label_counts.get(1, 0)}")
    print(f"  Total image rows: {len(combined)}")
