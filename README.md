# Multimodal Deep Learning Model for Lung Adenocarcinoma Staging

**Regeneron Science Talent Search Project**

A multimodal AI pipeline for binary staging of lung adenocarcinoma (LUAD) — Stage I/II vs. Stage III/IV — by fusing CT scan imaging with RNA-seq genomic expression data across three clinical cohorts.

---

## Overview

| | |
|---|---|
| **Task** | Binary LUAD staging (Stage I/II vs III/IV) |
| **Modalities** | Chest CT scans + RNA-seq gene expression |
| **Cohorts** | TCGA-LUAD, CPTAC-LUAD, Stanford |
| **Patients** | ~160 patients, 1,600 CT slices |
| **Best AUC** | 0.636 ± 0.128 (genomic-only) |
| **Evaluation** | 5-seed × 5-fold cohort-stratified CV (25 AUCs per mode) |

---

## Architecture

**Image branch:** ResNet-18 pretrained on 1.9M radiology DICOM images (RadiologyNET), frozen as a feature extractor → 512-dim CT embedding via global average pooling

**Genomic branch:** 18,514-gene RNA-seq profiles → supervised ComBat batch correction across cohorts → per-fold PCA (480 dims, 99.9% variance retained, fitted on train split only to prevent leakage)

**Fusion:** Linear concat (992-D → 64 → 1), ~63K trainable parameters

**Explainability:** Grad-CAM on ResNet-18 layer4 + gradient×input attribution on genomic dimensions

---

## Key Finding

Genomic-only embeddings (AUC 0.636) outperformed both CT-only (AUC 0.576) and multimodal fusion (AUC 0.618), consistent with LUAD's known molecular heterogeneity. Single 2D axial CT slices carry limited staging signal compared to transcriptomic profiles capturing proliferation axes (CDC20, CCNB1, BIRC5/survivin) and chromatin remodeling signatures (SETD2).

---

## Ablation Results

| Mode | Mean AUC | Std |
|---|---|---|
| Genomic Only | **0.6355** | 0.1281 |
| Multimodal (CT + RNA-seq) | 0.6176 | 0.1443 |
| CT Only | 0.5763 | 0.1269 |

*5 seeds × 5-fold cohort-stratified cross-validation, n=25 AUCs per mode*

---

## Files

| File | Description |
|---|---|
| `multimodal_sts_model.py` | Main model: dataset, ResNet encoder, fusion net, trainer, XAI |
| `export_embeddings.py` | ComBat batch correction + PCA embedding export |
| `gene_loading_plot.py` | PCA gene loading analysis (which genes drive each PC) |
| `genomic_3d_pca.py` | 3D PCA visualization of patients in genomic space |
| `results_figure.py` | Ablation box plots + Wilcoxon significance tests |
| `gene_loadings.png` | Top genes by PC1/PC2 loading, LUAD drivers highlighted |
| `results_figure.png` | Final ablation study box plot |
| `genomic_3d_pca.png` | Static 3D PCA scatter colored by staging label |
| `genomic_3d_pca.html` | Interactive 3D PCA scatter (drag to rotate, hover for patient info) |
| `explanation.png` | XAI figure: CT scan, Grad-CAM, genomic attribution |

---

## Setup

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib scipy Pillow
```

Data requirements (not included — patient privacy):
- `real_patient_data.csv` — patient manifest with image paths and labels
- `combat_corrected_matrix.npy` — ComBat-corrected RNA-seq matrix
- `combat_patient_ids.npy` — patient ID index for the above
- `ResNet18.pth` — RadiologyNET pretrained weights

---

## Training

```bash
python3 multimodal_sts_model.py
```

Runs 5-seed × 5-fold cohort-stratified CV across three ablation modes (multimodal, image_only, genomic_only). Results are printed to stdout; pipe to a log file for `results_figure.py`.

```bash
python3 results_figure.py training_output.log
```

---

## Data Sources

- **TCGA-LUAD** — The Cancer Genome Atlas (NCI GDC)
- **CPTAC-LUAD** — Clinical Proteomic Tumor Analysis Consortium (TCIA)
- **Stanford** — NSCLC Radiogenomics dataset (TCIA)
