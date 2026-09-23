"""
Downloads the RadImageNet ResNet-18 pretrained weights from the official
BMEII-AI release (pretrained on 1.35M CT/MRI/ultrasound images from TCIA).

Reference: Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research
Dataset for Effective Transfer Learning," Radiology: AI 2022.
GitHub: https://github.com/BMEII-AI/RadImageNet

The weights are stored in a PyTorch state-dict format.
"""
import os
import urllib.request

# Direct link to the ResNet-18 checkpoint hosted on the RadImageNet GitHub release.
URL = "https://github.com/BMEII-AI/RadImageNet/raw/main/models/RadImageNet-ResNet18_notop_torch.pth"
DEST = "RadImageNet_resnet18.pth"

if os.path.exists(DEST):
    print(f"Weights already present: {DEST}")
else:
    print(f"Downloading RadImageNet ResNet-18 weights → {DEST}")
    print(f"Source: {URL}")
    try:
        def progress(count, block, total):
            if total > 0:
                pct = count * block / total * 100
                print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)

        urllib.request.urlretrieve(URL, DEST, reporthook=progress)
        print(f"\nDone. Saved {os.path.getsize(DEST) / 1e6:.1f} MB → {DEST}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Manual download instructions:")
        print("  1. Open https://github.com/BMEII-AI/RadImageNet/releases")
        print("  2. Download 'RadImageNet-ResNet18_notop_torch.pth'")
        print(f"  3. Place it in this directory as: {DEST}")
