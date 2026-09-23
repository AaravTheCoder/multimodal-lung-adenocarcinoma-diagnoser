import os
import pandas as pd
import numpy as np
import torch
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

print("🚀 Generating Tri-Modal DICOM Sandbox Files for VS Code...")

os.makedirs("lung_images", exist_ok=True)
os.makedirs("real_genomics", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
num_patients = 20  # Keep it small and fast to test the code
data_index = []

for i in range(num_patients):
    pid = f"TCGA_PATIENT_{i:03d}"
    img_dir = f"lung_images/{pid}"
    os.makedirs(img_dir, exist_ok=True)
    img_path = f"{img_dir}/slice_0.dcm"  # Saving as true DICOM (.dcm)
    
    has_condition = np.random.choice([0, 1])
    
    # Save dummy 512 genomic array
    genomic_tensor = torch.randn(512)
    torch.save(genomic_tensor, f"real_genomics/{pid}.pt")
    
    # --- BUILD A VALID JUMBO DICOM DATASET STRUCTURE ---
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 166
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.3.4"

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # Structural visual frame payload matrix (224x224 grayscale)
    pixel_data = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    ds.Rows, ds.Columns = pixel_data.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_data.tobytes()
    
    # Force saving metadata headers cleanly
    pydicom.dcmwrite(img_path, ds, write_like_original=False)
    
    data_index.append([
        pid, 
        np.random.randint(45, 85),
        np.random.uniform(40, 95),
        np.random.choice([0, 1]),
        np.random.choice([0, 1, 2]),
        has_condition
    ])

df = pd.DataFrame(data_index, columns=[
    "patient_id", "Age", "Percent_Function", "Sex_encoded", "Smoking_encoded", "has_condition"
])
df.to_csv("clinical_manifest.csv", index=False)
print("✅ Sandbox DICOM data successfully written! Your model is ready to run.")