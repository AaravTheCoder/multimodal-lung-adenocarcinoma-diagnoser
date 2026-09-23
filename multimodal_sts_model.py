"""
================================================================================
Multimodal CT-Genomic Fusion Network for LUAD Staging — Regeneron STS
================================================================================
Purpose: Binary classification of lung adenocarcinoma (LUAD) pathological stage
         (Stage I/II vs Stage III/IV) by fusing:
         (a) a 512-dim RNA-seq genomic embedding (ComBat batch-corrected PCA),
             pre-extracted and stored per-patient as a .pt tensor, and
         (b) a 2D axial chest CT slice, processed through a ResNet-18 encoder
             pretrained on 1.9M DICOM radiology images (RadiologyNET).
         Fusion: cross-attention (v8) — 49 CT spatial tokens attend over 16 genomic
         tokens, fusing at the patch level before mean-pooling. Dropout=0.3 on
         attention weights regularises the 1.1M-param head for N≈160 patients.

Hardware target: Apple Silicon (MPS) with CPU fallback.
================================================================================
"""

import os
import copy
import warnings
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import matplotlib.pyplot as plt


# ==============================================================================
# 0. GLOBAL CONFIGURATION
# ==============================================================================
class Config:
    """
    Centralized hyperparameters and paths. Keeping this as a class (rather than
    loose module-level variables) makes the script trivially importable/testable
    and gives you one place to point to when judges ask about your setup.
    """
    CSV_PATH = "real_patient_data.csv"

    # --- Data contract (v2) ---------------------------------------------------
    # CSV is now expected to contain exactly these three columns:
    #   patient_id    -- unique identifier, used to locate the .pt embedding
    #   image_path    -- path to the fundus image file
    #   has_condition -- binary label (0/1)
    PATIENT_ID_COLUMN = "patient_id"
    IMAGE_PATH_COLUMN = "image_path"
    LABEL_COLUMN = "has_condition"

    # Directory containing one serialized tensor per patient, named
    # "<patient_id>.pt", each holding a 1D float tensor of shape (GENOMIC_DIM,).
    EMBEDDINGS_DIR = "genomic_embeddings"
    GENOMIC_DIM = 512

    IMAGE_SIZE = 224

    # --- Fusion / attention -----------------------------------------------------
    EMBED_DIM = 512          # must equal ResNet-18 layer4 output channels
    NUM_ATTENTION_HEADS = 8  # 512 / 8 = 64-dim per head, a clean split
    NUM_GENOMIC_TOKENS = 16  # genomic vector is chunked into 16 tokens of 32 dims
    # "attention" = cross-attention (1.1M params, needs large dataset)
    # "linear"    = concat + small MLP (~65K params, suited for <200 patients)
    FUSION_TYPE = "linear"

    # --- Backbone -----------------------------------------------------------
    FREEZE_EARLY_LAYERS = True   # freeze conv1/layer1/layer2
    FREEZE_ALL_RESNET = True     # freeze entire ResNet on small datasets (<100 patients)
    # RadiologyNET ResNet-18 weights (pretrained on 1.9M DICOM slices).
    # Set to None to use ImageNet weights.
    RADIOLOGY_WEIGHTS_PATH = "ResNet18.pth"

    BATCH_SIZE = 16
    NUM_WORKERS = 0  # 0 = main process only; avoids macOS shared-memory timeout with MPS
    VAL_SPLIT = 0.2
    NUM_EPOCHS = 25
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 5e-3
    SEED = 42

    # Probability of randomly zeroing one modality per batch during multimodal
    # training, forcing the fusion head to work with either modality alone.
    MODALITY_DROPOUT_P = 0.2

    # Seeds for repeated k-fold CV. Add more seeds (e.g. [42, 7, 13]) to get
    # more stable mean/std estimates at the cost of proportionally more runtime.
    CV_SEEDS = [42, 7, 13, 99, 2024]

    # Per-fold PCA: post-ComBat, pre-PCA matrix saved by export_embeddings.py
    COMBAT_MATRIX_PATH = "combat_corrected_matrix.npy"
    COMBAT_PIDS_PATH   = "combat_patient_ids.npy"
    N_PCA_COMPONENTS   = 512  # must be divisible by NUM_GENOMIC_TOKENS

    CHECKPOINT_PATH = "best_model.pt"


def set_seed(seed: int) -> None:
    """Reproducibility matters a great deal for a competition writeup."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ==============================================================================
# 1. DEVICE CONFIGURATION (Apple Silicon MPS with CPU fallback)
# ==============================================================================
def get_device() -> torch.device:
    """
    Explicitly targets Apple Silicon GPU acceleration via Metal Performance
    Shaders (MPS). Falls back gracefully to CPU if MPS is unavailable.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple Silicon GPU (MPS) detected. Using MPS backend.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[Device] CUDA GPU detected. Using CUDA backend.")
    else:
        device = torch.device("cpu")
        warnings.warn("[Device] No GPU backend found. Falling back to CPU. "
                       "Training will be significantly slower.")
    return device


# ==============================================================================
# 2. DATA PIPELINE
# ==============================================================================
class LUADMultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset pairing:
      - a 512-dim genomic embedding, loaded from '<EMBEDDINGS_DIR>/<patient_id>.pt'
      - a 2D retinal fundus image, loaded from the 'image_path' column
      - a binary label

    All validation (schema, file existence, tensor shape) happens once in
    __init__ so a bad row surfaces immediately rather than mid-training.
    """

    def __init__(self, csv_path: str, embeddings_dir: str, transform: T.Compose,
                 genomic_override: Optional[Dict[str, torch.Tensor]] = None,
                 genomic_dim: Optional[int] = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at '{csv_path}'. Check Config.CSV_PATH.")
        if genomic_override is None and not os.path.isdir(embeddings_dir):
            raise FileNotFoundError(
                f"Genomic embeddings directory not found at '{embeddings_dir}'. "
                f"Check Config.EMBEDDINGS_DIR."
            )

        self.df = pd.read_csv(csv_path)
        self.embeddings_dir = embeddings_dir
        self.transform = transform
        self.genomic_override = genomic_override  # patient_id → tensor, computed per fold
        # Store the expected dim as an instance attribute so DataLoader worker
        # processes (which re-import the module) use the correct value, not the
        # original Config.GENOMIC_DIM=512.
        self.genomic_dim = genomic_dim if genomic_dim is not None else Config.GENOMIC_DIM

        # --- Schema validation -------------------------------------------------
        required_cols = [Config.PATIENT_ID_COLUMN, Config.IMAGE_PATH_COLUMN, Config.LABEL_COLUMN]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {missing}. Expected columns: {required_cols}."
            )

        # --- Resolve embedding path per row and validate existence --------------
        self.df["_embedding_path"] = self.df[Config.PATIENT_ID_COLUMN].apply(
            lambda pid: os.path.join(self.embeddings_dir, f"{pid}.pt")
        )
        if self.genomic_override is not None:
            embedding_exists = self.df[Config.PATIENT_ID_COLUMN].apply(
                lambda pid: pid in self.genomic_override
            )
        else:
            embedding_exists = self.df["_embedding_path"].apply(os.path.exists)
        image_exists = self.df[Config.IMAGE_PATH_COLUMN].apply(os.path.exists)
        valid_mask = embedding_exists & image_exists

        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            warnings.warn(
                f"[Dataset] Dropping {n_dropped} row(s) with missing embedding "
                f"and/or image files."
            )
            self.df = self.df[valid_mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "Dataset is empty after validation. Check CSV paths, "
                "EMBEDDINGS_DIR, and that patient_id matches .pt filenames."
            )

        # --- Spot-check one embedding's shape up front (fail fast, not at epoch 5) ---
        first_pid = self.df[Config.PATIENT_ID_COLUMN].iloc[0]
        if self.genomic_override is not None:
            sample_tensor = self.genomic_override[first_pid]
        else:
            sample_tensor = torch.load(self.df["_embedding_path"].iloc[0], map_location="cpu")
        if sample_tensor.numel() != self.genomic_dim:
            raise ValueError(
                f"Genomic embedding at '{self.df['_embedding_path'].iloc[0]}' has "
                f"{sample_tensor.numel()} elements; expected genomic_dim={self.genomic_dim}."
            )

        # --- Label sanity check ---------------------------------------------------
        unique_labels = set(self.df[Config.LABEL_COLUMN].unique().tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"'{Config.LABEL_COLUMN}' must be binary (0/1). Found values: {unique_labels}"
            )

        print(f"[Dataset] Loaded {len(self.df)} valid samples "
              f"({self.df[Config.LABEL_COLUMN].mean()*100:.1f}% positive class).")

    def __len__(self) -> int:
        return len(self.df)

    def get_labels(self) -> np.ndarray:
        """Exposed so the Trainer can compute pos_weight without re-loading images."""
        return self.df[Config.LABEL_COLUMN].to_numpy()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # --- Genomic embedding: use per-fold override if available, else load .pt ---
        pid = row[Config.PATIENT_ID_COLUMN]
        if self.genomic_override is not None:
            genomic_tensor = self.genomic_override[pid].float().flatten()
        else:
            embedding_path = row["_embedding_path"]
            try:
                genomic_tensor = torch.load(embedding_path, map_location="cpu").float().flatten()
            except Exception as e:
                raise RuntimeError(f"Failed to load genomic embedding at '{embedding_path}': {e}")
        if genomic_tensor.numel() != self.genomic_dim:
            raise ValueError(
                f"Embedding for '{pid}' has shape {tuple(genomic_tensor.shape)}, "
                f"expected ({self.genomic_dim},)."
            )

        # --- Image ---
        img_path = row[Config.IMAGE_PATH_COLUMN]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open/decode image at '{img_path}': {e}")
        image_tensor = self.transform(image)  # shape: (3, 224, 224)

        # --- Label ---
        label = torch.tensor(row[Config.LABEL_COLUMN], dtype=torch.float32)

        return {
            "genomic": genomic_tensor,             # (496,)
            "image": image_tensor,                  # (3, 224, 224)
            "label": label,                          # scalar
            "patient_id": row[Config.PATIENT_ID_COLUMN],  # str, for patient-level aggregation
        }


def build_transforms(training: bool = False) -> T.Compose:
    """
    Standard torchvision preprocessing pipeline. ImageNet normalization stats
    are now MANDATORY (not just convenient) because the CNN encoder is a
    pretrained ResNet-18 -- feeding it inputs on a different numeric scale
    than it was trained on would badly degrade the transferred features.
    Training augmentations (flip, rotation, brightness) reduce overfitting
    on the small 29-patient cohort.
    """
    if training:
        return T.Compose([
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def _get_cohort(pid: str) -> str:
    if str(pid).startswith("TCGA"): return "TCGA"
    elif str(pid).startswith("C3"):  return "CPTAC"
    else:                             return "Stanford"


def build_kfold_splits(n_folds: int = 5, seed: int = Config.SEED):
    """Patient-level stratified k-fold splits, stratified by BOTH label AND cohort
    so every fold has a representative mix of TCGA/CPTAC/Stanford patients.
    Returns (splits, df) where splits is a list of (train_patient_set, val_patient_set)."""
    df = pd.read_csv(Config.CSV_PATH)

    label_counts = df.groupby(Config.PATIENT_ID_COLUMN)[Config.LABEL_COLUMN].nunique()
    conflicted = label_counts[label_counts > 1]
    if len(conflicted):
        raise ValueError(f"Patients with conflicting labels: {conflicted.index.tolist()}")

    patient_labels = df.groupby(Config.PATIENT_ID_COLUMN)[Config.LABEL_COLUMN].first()

    # Group patients by (cohort, label) — 6 strata — then distribute each
    # stratum evenly across folds. This prevents scanner-shift collapse where
    # one fold accidentally contains only patients from one cohort.
    strata: Dict[str, list] = {}
    for pid, label in patient_labels.items():
        key = f"{_get_cohort(pid)}_{int(label)}"
        strata.setdefault(key, []).append(pid)

    rng = np.random.default_rng(seed)
    for key in strata:
        rng.shuffle(strata[key])

    # Distribute each stratum into n_folds buckets round-robin
    fold_buckets: List[list] = [[] for _ in range(n_folds)]
    for key, pids in strata.items():
        sub_folds = [list(a) for a in np.array_split(pids, n_folds)]
        for i, sf in enumerate(sub_folds):
            fold_buckets[i].extend(sf)

    splits = []
    for i in range(n_folds):
        val_pats = set(fold_buckets[i])
        train_pats = set()
        for j in range(n_folds):
            if j != i:
                train_pats.update(fold_buckets[j])
        splits.append((train_pats, val_pats))
        n_pos_val = sum(1 for p in val_pats if patient_labels[p] == 1)
        n_neg_val = len(val_pats) - n_pos_val
        cohort_counts = {}
        for p in val_pats:
            c = _get_cohort(p)
            cohort_counts[c] = cohort_counts.get(c, 0) + 1
        print(f"  Fold {i+1}: train={len(train_pats)} patients | "
              f"val={len(val_pats)} ({n_pos_val} pos, {n_neg_val} neg) "
              f"cohorts={cohort_counts}")

    return splits, df


def compute_fold_genomics(train_pids, val_pids) -> Optional[Dict[str, torch.Tensor]]:
    """Fit PCA on all RNA-seq patients EXCEPT val CT patients; transform all.

    Fitting on ~490 patients (520 total minus ~30 val CT patients) gives ~480
    components — close to 512 but without leaking val gene expression into the
    PCA axes. Config.GENOMIC_DIM is updated in-place on the first call so the
    model is built with the correct input dimension.
    """
    if not os.path.exists(Config.COMBAT_MATRIX_PATH):
        warnings.warn(
            f"[Per-fold PCA] '{Config.COMBAT_MATRIX_PATH}' not found. "
            "Re-run export_embeddings.py to generate it. Falling back to pre-exported .pt files."
        )
        return None

    X_all    = np.load(Config.COMBAT_MATRIX_PATH)
    pids_all = np.load(Config.COMBAT_PIDS_PATH, allow_pickle=True).tolist()
    pid_to_idx = {p: i for i, p in enumerate(pids_all)}

    # Exclude val CT patients from PCA fitting; all other RNA-seq patients are safe.
    val_pid_set = set(val_pids)
    fit_idx = [pid_to_idx[p] for p in pids_all if p not in val_pid_set]
    if len(fit_idx) == 0:
        warnings.warn("[Per-fold PCA] No patients available for PCA fitting.")
        return None

    n_comp = min(Config.N_PCA_COMPONENTS, len(fit_idx) - 1, X_all.shape[1])
    n_comp = (n_comp // Config.NUM_GENOMIC_TOKENS) * Config.NUM_GENOMIC_TOKENS

    # Update Config.GENOMIC_DIM so models built in this run use the right dim.
    if Config.GENOMIC_DIM != n_comp:
        Config.GENOMIC_DIM = n_comp

    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_all[fit_idx])

    all_ct_pids = list(train_pids) + list(val_pids)
    override = {}
    for pid in all_ct_pids:
        if pid in pid_to_idx:
            x = pca.transform(X_all[[pid_to_idx[pid]]])[0]
            override[pid] = torch.FloatTensor(x)
        else:
            # Patient has CT but no RNA-seq — use zero vector (permanent modality dropout)
            override[pid] = torch.zeros(n_comp)

    var_retained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  [Per-fold PCA] {n_comp}-D, {var_retained:.1f}% variance retained "
          f"(fit on {len(fit_idx)} patients, {len(val_pids)} val patients excluded)")
    return override


def build_dataloaders() -> Tuple[DataLoader, DataLoader, Subset, Subset]:
    """Patient-level train/val split: all slices from a patient stay together
    in one split. This prevents data leakage from multi-slice patients.
    Training split uses augmentation; validation uses clean transforms only.
    """
    # Stratified patient-level split: ensure both classes appear in validation
    df = pd.read_csv(Config.CSV_PATH)
    patient_labels = df.groupby(Config.PATIENT_ID_COLUMN)[Config.LABEL_COLUMN].first()

    pos_patients = patient_labels[patient_labels == 1].index.tolist()
    neg_patients = patient_labels[patient_labels == 0].index.tolist()

    rng = np.random.default_rng(Config.SEED)
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    # Put at least 1 positive and proportional negatives in validation
    n_val_pos = max(1, int(len(pos_patients) * Config.VAL_SPLIT))
    n_val_neg = max(1, int(len(neg_patients) * Config.VAL_SPLIT))

    val_patients   = set(pos_patients[:n_val_pos] + neg_patients[:n_val_neg])
    train_patients = set(pos_patients[n_val_pos:] + neg_patients[n_val_neg:])

    train_indices = df.index[df[Config.PATIENT_ID_COLUMN].isin(train_patients)].tolist()
    val_indices   = df.index[df[Config.PATIENT_ID_COLUMN].isin(val_patients)].tolist()

    print(f"[Data] Train patients: {len(train_patients)} ({len(train_indices)} slices) | "
          f"Val patients: {len(val_patients)} ({len(val_indices)} slices)")

    train_dataset = LUADMultimodalDataset(Config.CSV_PATH, Config.EMBEDDINGS_DIR, build_transforms(training=True))
    val_dataset   = LUADMultimodalDataset(Config.CSV_PATH, Config.EMBEDDINGS_DIR, build_transforms(training=False))

    train_ds = Subset(train_dataset, train_indices)
    val_ds   = Subset(val_dataset,   val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, drop_last=False,
    )

    return train_loader, val_loader, train_ds, val_ds


def compute_pos_weight(train_subset: Subset) -> torch.Tensor:
    """
    Computes the positive-class weight for BCEWithLogitsLoss as
    (# negative samples / # positive samples), calculated STRICTLY on the
    training split to avoid any validation-set leakage into the loss
    function. A value > 1 up-weights the positive (presumably rarer) class.
    """
    full_dataset: LUADMultimodalDataset = train_subset.dataset
    all_labels = full_dataset.get_labels()
    train_labels = all_labels[train_subset.indices]

    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()

    if n_pos == 0:
        raise ValueError("Training split contains zero positive samples; cannot train a classifier.")

    pos_weight_value = n_neg / n_pos
    print(f"[Class Balance] Train split: {n_pos} positive / {n_neg} negative "
          f"-> pos_weight = {pos_weight_value:.3f}")
    return torch.tensor([pos_weight_value], dtype=torch.float32)


# Backward-compat alias so DataLoader multiprocessing workers can unpickle
# datasets that were created before the class rename.
RetinalGenomicDataset = LUADMultimodalDataset


# ==============================================================================
# 3. MODEL ARCHITECTURE
# ==============================================================================
class ResNetImageEncoder(nn.Module):
    """
    Feature Extractor A: pretrained ResNet-18 trunk, truncated before the
    global average pool / fc layer, so we retain the SPATIAL feature map
    (B, 512, 7, 7) at 224x224 input resolution. This spatial map serves two
    purposes:
      1. Flattened into 49 tokens of dim 512, it becomes the query sequence
         for cross-attention against the genomic tokens.
      2. It is the natural target for Grad-CAM, since Grad-CAM fundamentally
         requires a spatial (not globally-pooled) activation map.

    Early layers are optionally frozen (Config.FREEZE_EARLY_LAYERS) since
    low-level ImageNet features (edges, color blobs, textures) transfer well
    to medical images and freezing them is a standard, defensible way to
    reduce overfitting on a small labeled fundus dataset.
    """

    def __init__(self, freeze_early_layers: bool = True):
        super().__init__()
        radnet_path = Config.RADIOLOGY_WEIGHTS_PATH
        if radnet_path and os.path.exists(radnet_path):
            print(f"[Model] Loading RadiologyNET pretrained weights from '{radnet_path}'")
            backbone = resnet18(weights=None)
            ckpt = torch.load(radnet_path, map_location="cpu", weights_only=False)
            state = ckpt["model_state"]
            # Strip the 'model.' prefix added by their training wrapper
            state = {k.replace("model.", "", 1): v for k, v in state.items()}
            # Remove FC head — their 36-class output doesn't match torchvision's 1000
            state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            print(f"  [OK] Loaded. Only FC re-initialized (expected): {missing}")
        else:
            if radnet_path:
                print(f"[Model] '{radnet_path}' not found — using ImageNet weights.")
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Keep everything up through layer4; drop avgpool + fc (the original
        # 1000-way ImageNet classifier head, which we have no use for).
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4  # output: (B, 512, 7, 7) for 224x224 input

        if Config.FREEZE_ALL_RESNET:
            for param in backbone.parameters():
                param.requires_grad = False
            print("[Model] Froze entire ResNet-18 (feature extractor mode for small dataset). "
                  "Only fusion + classifier are trainable.")
        elif freeze_early_layers:
            for module in [self.stem, self.layer1, self.layer2]:
                for param in module.parameters():
                    param.requires_grad = False
            print("[Model] Froze ResNet-18 stem/layer1/layer2 (transfer-learning mode). "
                  "layer3 and layer4 remain trainable.")

        # Storage for Grad-CAM hooks.
        self.last_conv_activations: Optional[torch.Tensor] = None
        self.last_conv_gradients: Optional[torch.Tensor] = None

    def _save_activations_hook(self, module, input, output):
        self.last_conv_activations = output

    def _save_gradients_hook(self, module, grad_input, grad_output):
        self.last_conv_gradients = grad_output[0]

    def register_gradcam_hooks(self):
        """
        Attaches forward/backward hooks to the output of layer4 -- the final
        convolutional block of the ResNet-18 trunk -- for Grad-CAM saliency.
        This is the standard Grad-CAM target layer choice for ResNet
        architectures, since it's the last layer with meaningful spatial
        resolution before any pooling/flattening occurs.
        """
        self.layer4.register_forward_hook(self._save_activations_hook)
        self.layer4.register_full_backward_hook(self._save_gradients_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the raw spatial feature map (B, 512, 7, 7)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class GenomicTokenizer(nn.Module):
    """
    Converts a single 512-dim genomic embedding into a SEQUENCE of tokens,
    analogous to how a Vision Transformer splits an image into patches.

    Without this, a single 512-dim vector would be exactly one key/value
    token, and attention over a single token is mathematically trivial
    (softmax of one score is always 1.0) -- which produces no usable
    attention pattern for explainability. By chunking the vector into
    NUM_GENOMIC_TOKENS sub-embeddings and projecting each into EMBED_DIM
    space with a SHARED linear layer, we get a genuine (B, num_tokens,
    embed_dim) sequence that the image tokens can attend to non-trivially,
    and the resulting per-token attention weights are directly interpretable
    as "which chunk of the genomic embedding mattered for this image
    region."
    """

    def __init__(self, genomic_dim: int = None,
                 num_tokens: int = Config.NUM_GENOMIC_TOKENS,
                 embed_dim: int = Config.EMBED_DIM):
        if genomic_dim is None:
            genomic_dim = Config.GENOMIC_DIM
        super().__init__()
        if genomic_dim % num_tokens != 0:
            raise ValueError(
                f"GENOMIC_DIM ({genomic_dim}) must be divisible by "
                f"NUM_GENOMIC_TOKENS ({num_tokens})."
            )
        self.num_tokens = num_tokens
        self.chunk_size = genomic_dim // num_tokens

        # Shared projection applied independently to each chunk (weight-tying
        # keeps the parameter count small and mirrors ViT's patch embedding).
        self.token_proj = nn.Linear(self.chunk_size, embed_dim)

        # Learned positional embeddings so the model can distinguish "chunk 0"
        # (e.g. the first 32 dims of the genomic embedding) from "chunk 15",
        # since a Linear layer alone is permutation-agnostic across chunks.
        self.positional_embedding = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)

    def forward(self, genomic: torch.Tensor) -> torch.Tensor:
        """
        genomic: (B, genomic_dim)
        returns: (B, num_tokens, embed_dim)
        """
        B = genomic.shape[0]
        chunks = genomic.view(B, self.num_tokens, self.chunk_size)  # (B, 16, 32)
        tokens = self.token_proj(chunks)                              # (B, 16, embed_dim)
        tokens = tokens + self.positional_embedding
        return tokens


class CrossAttentionFusion(nn.Module):
    """
    Fusion Layer: a genuine multi-head cross-attention block built on
    nn.MultiheadAttention. Image spatial tokens (queries) attend over
    genomic sub-embedding tokens (keys/values), producing:
      - fused image tokens, informed by genomic context
      - a real (B, num_heads-averaged, 49, 16) attention weight tensor,
        which is a first-class explainability artifact: for each retinal
        patch, which genomic components did the model weight most heavily.

    A residual connection + LayerNorm around the attention block follows
    standard Transformer design, stabilizing training.
    """

    def __init__(self, embed_dim: int = Config.EMBED_DIM,
                 num_heads: int = Config.NUM_ATTENTION_HEADS):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.3, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_tokens: torch.Tensor, genomic_tokens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image_tokens:   (B, 49, embed_dim)  -- queries
        genomic_tokens: (B, 16, embed_dim)  -- keys and values

        Returns:
          fused_tokens:     (B, 49, embed_dim) -- attended, residual-added, normed
          attn_weights:      (B, 49, 16)        -- averaged over heads (interpretable)
        """
        attended, attn_weights = self.mha(
            query=image_tokens, key=genomic_tokens, value=genomic_tokens,
            need_weights=True, average_attn_weights=True,
        )
        fused_tokens = self.norm(image_tokens + attended)  # residual connection
        return fused_tokens, attn_weights


class MultimodalFusionNet(nn.Module):
    """
    Full hybrid architecture:
        Image  -> ResNetImageEncoder -> spatial feats (B,512,7,7)
                                      -> flatten -> image tokens (B,49,512)
        Genomic (512-d) -> GenomicTokenizer -> genomic tokens (B,16,512)
                                                          |
                                                          v
                        CrossAttentionFusion (MultiheadAttention) -> fused tokens (B,49,512)
                                                          v
                                     mean-pool over tokens -> (B,512)
                                                          v
                                   Classification MLP Head -> logit
    """

    def __init__(self,
                 genomic_dim: int = None,
                 embed_dim: int = Config.EMBED_DIM):
        super().__init__()
        if genomic_dim is None:
            genomic_dim = Config.GENOMIC_DIM  # read at construction time, not import time
        self.fusion_type = Config.FUSION_TYPE
        self.image_encoder = ResNetImageEncoder(freeze_early_layers=Config.FREEZE_EARLY_LAYERS)

        if self.fusion_type == "linear":
            # Simple concat fusion: mean-pooled image (512) + genomic (496) → MLP
            # ~65K trainable params — appropriate for datasets < 300 patients.
            concat_dim = embed_dim + genomic_dim  # 512 + 512 = 1024
            self.classifier = nn.Sequential(
                nn.Linear(concat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
            print(f"[Model] Fusion: LINEAR concat ({concat_dim}→64→1, ~{concat_dim*64+64+65:,} trainable params in head)")
        else:
            # Cross-attention fusion: powerful but needs > ~500 paired samples.
            self.genomic_tokenizer = GenomicTokenizer(genomic_dim=genomic_dim, embed_dim=embed_dim)
            self.fusion = CrossAttentionFusion(embed_dim=embed_dim)
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
            )
            print("[Model] Fusion: CROSS-ATTENTION (1.1M params — requires large dataset)")

        self._genomic_input_ref: Optional[torch.Tensor] = None

    def forward(self, image: torch.Tensor, genomic: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          logit:        (B, 1)   raw pre-sigmoid output
          attn_weights: (B, 49, 16) for cross-attention mode; None for linear mode
        """
        genomic = genomic.clone().requires_grad_(True)
        self._genomic_input_ref = genomic

        # --- Image branch: ResNet feature map → mean pool → (B, 512) ---
        feature_map = self.image_encoder(image)          # (B, 512, 7, 7)
        image_vec = feature_map.mean(dim=[2, 3])         # (B, 512)

        if self.fusion_type == "linear":
            # Concatenate image vector and raw genomic embedding, classify.
            fused = torch.cat([image_vec, genomic], dim=1)  # (B, 1024)
            logit = self.classifier(fused)                   # (B, 1)
            return logit, None
        else:
            B, C, H, W = feature_map.shape
            image_tokens = feature_map.flatten(2).transpose(1, 2)  # (B, 49, 512)
            genomic_tokens = self.genomic_tokenizer(genomic)        # (B, 16, 512)
            fused_tokens, attn_weights = self.fusion(image_tokens, genomic_tokens)
            pooled = fused_tokens.mean(dim=1)
            logit = self.classifier(pooled)
            return logit, attn_weights


# ==============================================================================
# 4. TRAINING / VALIDATION LOOP
# ==============================================================================
class Trainer:
    """
    Encapsulates the full training regimen: optimizer, loss (with explicit
    class-imbalance correction via pos_weight), per-epoch loops, metric
    tracking (loss, accuracy, ROC-AUC), and checkpointing of the best
    validation model.
    """

    def __init__(self, model: nn.Module, device: torch.device, pos_weight: torch.Tensor):
        self.model = model.to(device)
        self.device = device

        # pos_weight up-weights the positive class in the loss in proportion
        # to its rarity in the TRAINING split (computed by compute_pos_weight).
        # This directly counteracts class imbalance without resampling or
        # duplicating data, and keeps ROC-AUC/accuracy comparisons clean.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

        # Only pass parameters that actually require grad (relevant now that
        # early ResNet layers may be frozen) to the optimizer.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_auc": [], "val_auprc": [],
        }
        self.best_val_auc = -1.0
        self.best_state_dict = None

    def _run_epoch(self, loader: DataLoader, training: bool,
                   mode: str = "multimodal") -> Tuple[float, float, float, float]:
        self.model.train() if training else self.model.eval()

        from collections import defaultdict
        patient_logits: Dict[str, list] = defaultdict(list)
        patient_labels: Dict[str, float] = {}

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images    = batch["image"].to(self.device)
                genomics  = batch["genomic"].to(self.device)
                labels    = batch["label"].to(self.device)
                pids      = batch["patient_id"]

                if mode == "image_only":
                    genomics = torch.zeros_like(genomics)
                elif mode == "genomic_only":
                    images = torch.zeros_like(images)
                elif mode == "multimodal" and training:
                    # Modality dropout: randomly zero one modality per batch
                    r = torch.rand(1).item()
                    if r < Config.MODALITY_DROPOUT_P / 2:
                        genomics = torch.zeros_like(genomics)
                    elif r < Config.MODALITY_DROPOUT_P:
                        images = torch.zeros_like(images)

                logits, _ = self.model(images, genomics)  # (B, 1)
                logits_flat = logits.squeeze(1)            # (B,)

                for i, pid in enumerate(pids):
                    patient_logits[pid].append(logits_flat[i])
                    patient_labels[pid] = labels[i].item()

            # --- Patient-level loss & update (one gradient step per epoch) --------
            if training:
                self.optimizer.zero_grad()
                pat_logits_list, pat_labels_list = [], []
                for pid, logit_list in patient_logits.items():
                    pat_logits_list.append(torch.stack(logit_list).mean())
                    pat_labels_list.append(patient_labels[pid])
                pat_logits_t = torch.stack(pat_logits_list).unsqueeze(1)
                pat_labels_t = torch.tensor(pat_labels_list, dtype=torch.float32,
                                             device=self.device).unsqueeze(1)
                loss = self.criterion(pat_logits_t, pat_labels_t)
                loss.backward()
                self.optimizer.step()
            else:
                pat_logits_list, pat_labels_list = [], []
                for pid, logit_list in patient_logits.items():
                    pat_logits_list.append(torch.stack(logit_list).mean().item())
                    pat_labels_list.append(patient_labels[pid])
                dummy_logits = torch.tensor([[v] for v in pat_logits_list],
                                             dtype=torch.float32, device=self.device)
                dummy_labels = torch.tensor([[v] for v in pat_labels_list],
                                             dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    loss = self.criterion(dummy_logits, dummy_labels)

        # --- Metrics at patient level -------------------------------------------
        avg_loss = loss.item()
        all_probs  = [torch.sigmoid(torch.tensor(v)).item() for v in pat_logits_list] \
                     if not training else \
                     [torch.sigmoid(lv.detach()).item() for lv in pat_logits_list]
        all_labels_list = pat_labels_list
        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels_list, preds)

        try:
            auc = roc_auc_score(all_labels_list, all_probs)
        except ValueError:
            auc = float("nan")
            warnings.warn("ROC-AUC undefined for this epoch (only one class present in labels).")

        try:
            auprc = average_precision_score(all_labels_list, all_probs)
        except ValueError:
            auprc = float("nan")

        return avg_loss, acc, auc, auprc

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            mode: str = "multimodal") -> None:
        print(f"\n{'='*70}\nStarting training [{mode}] for up to {Config.NUM_EPOCHS} epochs on {self.device}\n{'='*70}")

        early_stop_patience = 5
        epochs_without_improvement = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss, train_acc, _, _ = self._run_epoch(train_loader, training=True, mode=mode)
            val_loss, val_acc, val_auc, val_auprc = self._run_epoch(val_loader, training=False, mode=mode)

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)
            self.history["val_auprc"].append(val_auprc)

            print(f"Epoch {epoch:2d}/{Config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                  f"AUC: {val_auc:.4f} AUPRC: {val_auprc:.4f}")

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state_dict, Config.CHECKPOINT_PATH)
                print(f"  -> New best model saved (Val AUC: {val_auc:.4f})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    print(f"  -> Early stopping triggered (no AUC improvement for {early_stop_patience} epochs)")
                    break

        print(f"\nTraining complete. Best Val ROC-AUC: {self.best_val_auc:.4f}")
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

    def plot_history(self, save_path: str = "training_curves.png") -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.history["train_loss"], label="Train")
        axes[0].plot(self.history["val_loss"], label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(self.history["train_acc"], label="Train")
        axes[1].plot(self.history["val_acc"], label="Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        axes[2].plot(self.history["val_auc"], label="Val ROC-AUC", color="green")
        axes[2].set_title("ROC-AUC")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Training curves saved to '{save_path}'")
        plt.close(fig)


# ==============================================================================
# 5. EXPLAINABILITY (XAI)
# ==============================================================================
class Explainer:
    """
    Produces three complementary explainability artifacts for a single sample:

    1. Grad-CAM saliency on the ResNet-18 layer4 feature map: which pixels in
       the retinal image most increased the model's predicted probability of
       the positive class.

    2. Genomic attribution (gradient x input) on the raw 512-dim embedding:
       which dimensions of the ORIGINAL genomic vector most influenced the
       logit, weighted by their own magnitude.

    3. The genuine multi-head cross-attention map (B, 49, 16): aggregated
       (mean over the 49 image-patch queries) into a (16,) distribution
       showing which genomic TOKENS (sub-embedding chunks) the retinal
       image attended to most heavily overall. Unlike the v1 scalar gate,
       this is a non-trivial, learned distribution.
    """

    def __init__(self, model: MultimodalFusionNet, device: torch.device):
        self.model = model
        self.device = device
        self.model.image_encoder.register_gradcam_hooks()

    def explain_sample(self, image: torch.Tensor, genomic: torch.Tensor
                        ) -> Dict[str, np.ndarray]:
        """
        image:   (1, 3, 224, 224) single-sample batch
        genomic: (1, 512) single-sample batch
        """
        self.model.eval()
        image = image.to(self.device)
        genomic = genomic.to(self.device)

        # Temporarily enable grad on layer4 so the backward hook fires even
        # when FREEZE_ALL_RESNET=True (frozen params have requires_grad=False,
        # which silences the backward hook on layer4).
        layer4_params = list(self.model.image_encoder.layer4.parameters())
        prev_requires_grad = [p.requires_grad for p in layer4_params]
        for p in layer4_params:
            p.requires_grad_(True)

        with torch.enable_grad():
            logit, attn_weights = self.model(image, genomic)
            prob = torch.sigmoid(logit)
            self.model.zero_grad()
            logit.backward()

        # Restore frozen state
        for p, prev in zip(layer4_params, prev_requires_grad):
            p.requires_grad_(prev)

        # --- 1. Grad-CAM for the image -----------------------------------------
        activations = self.model.image_encoder.last_conv_activations  # (1,512,7,7)
        gradients = self.model.image_encoder.last_conv_gradients      # (1,512,7,7)

        weights = gradients.mean(dim=(2, 3), keepdim=True)            # (1,512,1,1)
        cam = (weights * activations).sum(dim=1, keepdim=True)        # (1,1,7,7)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                             mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # --- 2. Genomic attribution (gradient x input) --------------------------
        genomic_grad = self.model._genomic_input_ref.grad.detach().cpu().numpy().flatten()
        genomic_value = genomic.detach().cpu().numpy().flatten()
        genomic_attribution = genomic_grad * genomic_value  # (512,)

        # --- 3. Token-level genomic attribution -----------------------------------
        # Cross-attention mode: use actual attention weights.
        # Linear fusion mode: group gradient×input attribution into NUM_GENOMIC_TOKENS
        # bins so panel 4 always shows a real signal (not zeros).
        if attn_weights is not None:
            token_attention = attn_weights.squeeze(0).mean(dim=0).detach().cpu().numpy()
        else:
            # Sum absolute attribution within each token-sized chunk (32 dims each)
            abs_attr = np.abs(genomic_attribution)
            chunk = Config.GENOMIC_DIM // Config.NUM_GENOMIC_TOKENS
            token_attention = abs_attr.reshape(Config.NUM_GENOMIC_TOKENS, chunk).sum(axis=1)
            if token_attention.max() > 0:
                token_attention = token_attention / token_attention.max()

        return {
            "predicted_probability": prob.item(),
            "image_saliency_map": cam,
            "genomic_attribution": genomic_attribution,
            "genomic_token_attention": token_attention,
        }

    def visualize(self, original_image_pil: Image.Image, explanation: Dict,
                   top_k_genes: int = 10, save_path: str = "explanation.png") -> None:
        """Renders a judge-friendly explainability figure with 4 panels."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(original_image_pil)
        axes[0].set_title("Input Chest CT Scan (LUAD)")
        axes[0].axis("off")

        axes[1].imshow(original_image_pil)
        axes[1].imshow(explanation["image_saliency_map"], cmap="jet", alpha=0.45)
        axes[1].set_title("Grad-CAM Saliency\n(ResNet-18 layer4)")
        axes[1].axis("off")

        attribution = explanation["genomic_attribution"]
        top_idx = np.argsort(np.abs(attribution))[::-1][:top_k_genes]
        top_vals = attribution[top_idx]
        colors = ["crimson" if v > 0 else "steelblue" for v in top_vals]
        axes[2].barh([f"dim {i}" for i in top_idx][::-1], top_vals[::-1], color=colors[::-1])
        axes[2].set_title(f"Top {top_k_genes} Genomic Dims\n(grad x input)")
        axes[2].set_xlabel("Attribution")

        token_attn = explanation["genomic_token_attention"]
        axes[3].bar(range(len(token_attn)), token_attn, color="darkorange")
        fusion_label = ("Cross-Attention Weight\nper Genomic Token"
                        if Config.FUSION_TYPE == "cross_attention"
                        else "Gradient Attribution\nper Genomic Token (normalized)")
        axes[3].set_title(fusion_label)
        axes[3].set_xlabel("Genomic token index (32 dims each)")
        axes[3].set_ylabel("Attribution (normalized)")

        fig.suptitle(f"Predicted P(has_condition) = {explanation['predicted_probability']:.3f}",
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"[XAI] Explanation figure saved to '{save_path}'")
        plt.close(fig)


# ==============================================================================
# 6. MAIN ENTRY POINT — 5-Fold Stratified CV + Ablation Study
# ==============================================================================
def run_cv_mode(mode: str, splits, df, device) -> List[float]:
    """Run 5-fold CV for one ablation mode. Returns list of per-fold val AUCs."""
    fold_aucs = []

    for fold_idx, (train_patients, val_patients) in enumerate(splits):
        print(f"\n--- Fold {fold_idx+1}/{len(splits)}  [{mode}] ---")

        # Per-fold PCA: fit on all RNA-seq patients except val CT patients.
        # Falls back to pre-exported .pt files if combat matrix not available.
        genomic_override = compute_fold_genomics(train_patients, val_patients)

        fold_genomic_dim = Config.GENOMIC_DIM  # captured after per-fold PCA updated it
        train_dataset = LUADMultimodalDataset(
            Config.CSV_PATH, Config.EMBEDDINGS_DIR, build_transforms(training=True),
            genomic_override=genomic_override, genomic_dim=fold_genomic_dim)
        val_dataset = LUADMultimodalDataset(
            Config.CSV_PATH, Config.EMBEDDINGS_DIR, build_transforms(training=False),
            genomic_override=genomic_override, genomic_dim=fold_genomic_dim)

        fdf = train_dataset.df
        train_indices = fdf.index[fdf[Config.PATIENT_ID_COLUMN].isin(train_patients)].tolist()
        val_indices   = fdf.index[fdf[Config.PATIENT_ID_COLUMN].isin(val_patients)].tolist()

        train_ds = Subset(train_dataset, train_indices)
        val_ds   = Subset(val_dataset,   val_indices)

        # pos_weight from training split only (no leakage)
        all_labels   = train_dataset.get_labels()
        train_labels = all_labels[train_indices]
        n_pos = int((train_labels == 1).sum())
        n_neg = int((train_labels == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        print(f"[Class Balance] Train: {n_pos} pos / {n_neg} neg → pos_weight={pos_weight.item():.2f}")

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                                   shuffle=True, num_workers=Config.NUM_WORKERS)
        val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                                   shuffle=False, num_workers=Config.NUM_WORKERS)

        model   = MultimodalFusionNet()
        trainer = Trainer(model, device, pos_weight)
        trainer.fit(train_loader, val_loader, mode=mode)

        fold_aucs.append(trainer.best_val_auc)
        print(f"Fold {fold_idx+1} best AUC: {trainer.best_val_auc:.4f}")

        # Save best model from fold 1 multimodal for explainability
        if fold_idx == 0 and mode == "multimodal":
            torch.save(trainer.model.state_dict(), Config.CHECKPOINT_PATH)
            # Run XAI on one validation sample
            try:
                explainer = Explainer(trainer.model, device)
                sample_batch = next(iter(val_loader))
                sample_image   = sample_batch["image"][0:1]
                sample_genomic = sample_batch["genomic"][0:1]
                explanation = explainer.explain_sample(sample_image, sample_genomic)
                unnorm = T.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225])
                original_pil = T.ToPILImage()(unnorm(sample_image.squeeze(0)).clamp(0, 1))
                explainer.visualize(original_pil, explanation)
                print(f"[XAI] P(Stage III/IV) = {explanation['predicted_probability']:.4f}")
            except Exception as e:
                print(f"[XAI] Skipped: {e}")

    return fold_aucs


def main():
    device = get_device()

    _model = MultimodalFusionNet()
    n_total     = sum(p.numel() for p in _model.parameters())
    n_trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {n_total:,} | Trainable: {n_trainable:,}")
    del _model

    # Repeated k-fold CV: run full ablation for each seed and aggregate.
    # CV_SEEDS = [42] gives one run (same as before). Add more seeds for
    # more stable estimates: [42, 7, 13] gives 15 AUC values per mode.
    all_seed_results: Dict[str, Dict] = {
        m: {"all_aucs": []} for m in ["multimodal", "image_only", "genomic_only"]
    }

    for seed_idx, seed in enumerate(Config.CV_SEEDS):
        set_seed(seed)
        print(f"\n{'#'*70}")
        print(f"  REPEATED CV  —  Seed {seed}  ({seed_idx+1}/{len(Config.CV_SEEDS)})")
        print(f"{'#'*70}")

        print(f"\n[CV] Building 5-fold cohort-stratified splits (seed={seed})...")
        splits, df = build_kfold_splits(n_folds=5, seed=seed)

        for mode in ["multimodal", "image_only", "genomic_only"]:
            print(f"\n{'='*70}")
            print(f"ABLATION MODE: {mode.upper()}  (seed={seed})")
            print(f"{'='*70}")
            aucs = run_cv_mode(mode, splits, df, device)
            all_seed_results[mode]["all_aucs"].extend(
                [a for a in aucs if not np.isnan(a)]
            )

    # ── Final summary across all seeds × folds ───────────────────────────────
    n_seeds = len(Config.CV_SEEDS)
    n_folds = 5
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY RESULTS ({n_seeds} seed(s) × {n_folds}-Fold Stratified CV)")
    print(f"{'='*70}")
    print(f"{'Mode':<15} {'Mean AUC':>10} {'Std':>8}  {'N AUCs':>7}")
    print(f"{'-'*70}")
    for mode, r in all_seed_results.items():
        aucs = r["all_aucs"]
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        std_auc  = float(np.std(aucs))  if aucs else float("nan")
        print(f"{mode:<15} {mean_auc:>10.4f} {std_auc:>8.4f}  {len(aucs):>7}")

    best_mode = max(all_seed_results, key=lambda m: np.mean(all_seed_results[m]["all_aucs"] or [0]))
    best_mean = np.mean(all_seed_results[best_mode]["all_aucs"])
    best_std  = np.std(all_seed_results[best_mode]["all_aucs"])
    print(f"\nBest mode: {best_mode} (AUC {best_mean:.4f} ± {best_std:.4f})")
    print("\nDone. Artifacts: best_model.pt, explanation.png")


if __name__ == "__main__":
    main()
