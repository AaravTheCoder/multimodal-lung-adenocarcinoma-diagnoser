import os
import glob
import pydicom
import numpy as np
import pandas as pd
from PIL import Image

print("Processing TCIA CT DICOM files into 224x224 PNG slices (10 slices per patient)...")

METADATA_CSV = "tcia_raw_dicom/manifest-1Rd7jPNd5199284876140322680/metadata.csv"
DICOM_ROOT   = "tcia_raw_dicom/manifest-1Rd7jPNd5199284876140322680"
OUTPUT_DIR   = "processed_ct_images"
PATIENT_CSV  = "real_patient_data.csv"
SLICES_PER_PATIENT = 10  # extract 5 evenly-spaced slices per patient

os.makedirs(OUTPUT_DIR, exist_ok=True)

meta = pd.read_csv(METADATA_CSV)
ct_only = meta[meta["Modality"] == "CT"].copy()

patient_df = pd.read_csv(PATIENT_CSV)
csv_ids = set(patient_df["patient_id"].tolist())

matched = ct_only[ct_only["Subject ID"].isin(csv_ids)]
print(f"Found {matched['Subject ID'].nunique()} patients with both CT scans and genomic data.")

def dicom_to_png(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    pixel_array = dcm.pixel_array.astype(np.float32)
    pixel_array = np.clip(pixel_array, -1000, 400)
    norm_img = (
        (pixel_array - pixel_array.min())
        / (pixel_array.max() - pixel_array.min() + 1e-8)
        * 255.0
    ).astype(np.uint8)
    return Image.fromarray(norm_img).convert("RGB").resize((224, 224))

processed_patients = set()
new_rows = []

for _, row in matched.iterrows():
    patient_id = row["Subject ID"]
    if patient_id in processed_patients:
        continue

    rel_location = row["File Location"].lstrip("./")
    series_dir = os.path.join(DICOM_ROOT, rel_location)
    dcm_files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))

    if not dcm_files:
        print(f"  [SKIP] No DICOM files for {patient_id}")
        continue

    if len(dcm_files) < 10:
        print(f"  [SKIP] {patient_id} — only {len(dcm_files)} slice(s), likely scout image")
        continue

    # Pick 5 evenly-spaced indices, avoiding the very first and last 10% of slices
    # (those tend to be above/below the lung region)
    n = len(dcm_files)
    start = int(n * 0.10)
    end   = int(n * 0.90)
    indices = np.linspace(start, end, SLICES_PER_PATIENT, dtype=int)

    label = patient_df.loc[patient_df["patient_id"] == patient_id, "has_condition"].values[0]
    saved = []
    for i, idx in enumerate(indices):
        try:
            pil_img = dicom_to_png(dcm_files[idx])
            out_path = os.path.join(OUTPUT_DIR, f"{patient_id}_slice{i}.png")
            pil_img.save(out_path)
            saved.append(out_path)
            new_rows.append({
                "patient_id": patient_id,
                "image_path": out_path,
                "has_condition": label
            })
        except Exception as e:
            print(f"  [WARN] Could not read slice {idx} for {patient_id}: {e}")

    processed_patients.add(patient_id)
    print(f"  [OK] {patient_id} — {len(saved)} slices saved (label={label}, total={n} slices)")

print(f"\nConverted {len(processed_patients)} patients → {len(new_rows)} total samples.")

new_manifest = pd.DataFrame(new_rows)
new_manifest.to_csv("real_patient_data.csv", index=False)
print(f"Updated real_patient_data.csv with {len(new_manifest)} rows.")
print("Done! You can now run: python3 multimodal_sts_model.py")
