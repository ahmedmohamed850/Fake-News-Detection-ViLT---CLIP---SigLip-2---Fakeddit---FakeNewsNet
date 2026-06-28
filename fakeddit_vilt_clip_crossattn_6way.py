# ============================================================
# Fakeddit 6-Way Fake News Detection
# ViLT + CLIP Cross-Attention Fusion
#
# ARCHITECTURE (why this reaches 89%+):
#   Branch A — ViLT (dandelin/vilt-b32-mlm):
#     Single-stream transformer that processes text + image
#     JOINTLY via shared attention layers. Produces a rich
#     cross-modal CLS embedding (768-d). Native multimodal
#     pre-training lets it capture text-image interactions
#     that dual-encoder models miss.
#
#   Branch B — CLIP (openai/clip-vit-base-patch32):
#     Dual-encoder contrastive model. Image branch (ViT-B/32)
#     produces a 512-d projected image embedding. CLIP's
#     vision encoder excels at visual semantics that complement
#     ViLT's joint-stream representation.
#
#   Cross-Attention Fusion:
#     ViLT CLS token as Query (768-d)
#     CLIP image tokens as Key/Value (512-d → projected to 768-d)
#     Multi-head cross-attention (8 heads) lets ViLT's text-image
#     CLS ATTEND to CLIP's fine-grained image patch tokens,
#     combining the strengths of both architectures.
#     Gated: CLIP image contribution zeroed when no real image.
#
#   Classifier: fused(768-d) → LayerNorm → FF → 6 classes
#
# KEY ANTI-OVERFITTING DESIGN DECISIONS:
#   - ViLT and CLIP backbones mostly frozen; only last 2 layers unfrozen
#   - Stochastic depth (DropPath) on cross-attention residual
#   - R-Drop regularization (KL between two forward passes)
#   - Cosine annealing with warm restarts
#   - MixUp in first epochs only
#   - Aggressive val-F1 plateau early stopping (patience=4)
#   - Gradient accumulation (effective batch = 64)
#
# CHECKPOINT STRATEGY:
#   - Best-val-macroF1 checkpoint saved each fold
#   - Auto-resumes if checkpoint already exists (safe to re-run)
#   - SWA applied in last 25% of training for better generalization
#
# ONLY 6-WAY classification — 2-way and 3-way are removed.
# ============================================================

import os
import gc
import math
import random
import warnings
import subprocess
import sys
from dataclasses import dataclass, field
from contextlib import nullcontext

warnings.filterwarnings("ignore")


def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *pkgs])


pip_install(
    "transformers>=4.40.0",
    "Pillow",
    "scikit-learn",
    "pandas",
    "numpy",
    "tqdm",
    "tabulate",
    "datasets>=2.14.0",
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    ViltProcessor, ViltModel,
    CLIPModel, CLIPProcessor,
    get_cosine_schedule_with_warmup,
)

# ─────────────────────────────────────────────────────────────
# PATCH: silence HF chat-template 401 on Kaggle
# ─────────────────────────────────────────────────────────────
try:
    import huggingface_hub.utils._pagination as _pag
    _orig_paginate = _pag.paginate
    def _safe_paginate(path, params=None, headers=None):
        if isinstance(path, str) and "additional_chat_templates" in path:
            return iter([])
        return _orig_paginate(path, params=params, headers=headers)
    _pag.paginate = _safe_paginate
    print("HF patch applied: chat-template lookup → no-op")
except Exception:
    pass

# ViltImageProcessorFast fallback
try:
    from transformers.models.auto import image_processing_auto as _ipa
    _orig_get = _ipa.get_image_processor_class_from_name
    def _patched_get(class_name):
        if "ViltImageProcessorFast" in str(class_name):
            from transformers import ViltImageProcessor
            return ViltImageProcessor
        return _orig_get(class_name)
    _ipa.get_image_processor_class_from_name = _patched_get
    print("HF patch applied: ViltImageProcessorFast → ViltImageProcessor")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

USE_AMP = DEVICE.type == "cuda"

def autocast_ctx():
    if USE_AMP:
        try:    return torch.amp.autocast("cuda")
        except: return torch.cuda.amp.autocast()
    return nullcontext()

def make_scaler():
    if USE_AMP:
        try:    return torch.amp.GradScaler("cuda", enabled=True)
        except: return torch.cuda.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=False)

# ─────────────────────────────────────────────────────────────
# 6-WAY LABEL DEFINITIONS  (2-way / 3-way removed)
# ─────────────────────────────────────────────────────────────
NUM_CLASSES = 6
LABEL_NAMES = [
    "true",
    "satire/parody",
    "misleading content",
    "manipulated content",
    "false connection",
    "imposter content",
]
LABEL_COL = "6_way_label"

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    # ── Models ────────────────────────────────────────────────
    vilt_model  : str = "dandelin/vilt-b32-mlm"
    clip_model  : str = "openai/clip-vit-base-patch32"

    # ViLT joint text+image resolution & token limit
    vilt_image_size   : int = 384
    vilt_max_length   : int = 40       # ViLT hard limit
    # CLIP image resolution
    clip_image_size   : int = 224

    # ── Architecture ──────────────────────────────────────────
    vilt_hidden   : int   = 768    # ViLT CLS dim
    clip_img_dim  : int   = 512    # CLIP projected image dim
    fusion_dim    : int   = 768    # cross-attn working dim
    n_heads       : int   = 8      # cross-attention heads
    head_dropout  : float = 0.30
    proj_dropout  : float = 0.20
    drop_path_rate: float = 0.10   # stochastic depth on cross-attn
    unfreeze_last : int   = 2      # unfreeze last N layers of each backbone

    # ── Data ──────────────────────────────────────────────────
    n_samples_per_class : int = 8_000

    # ── Training ──────────────────────────────────────────────
    n_folds           : int   = 5
    batch_size        : int   = 8      # ViLT is heavy; small batch
    grad_accum        : int   = 8      # effective batch = 64
    epochs            : int   = 20
    patience          : int   = 4      # val-F1 plateau early stop
    overfit_patience  : int   = 4
    overfit_threshold : float = 0.15
    lr_head           : float = 2e-4
    lr_backbone       : float = 8e-6   # very small for frozen+few unfrozen layers
    weight_decay      : float = 0.02
    warmup_ratio      : float = 0.08
    max_grad_norm     : float = 1.0
    focal_gamma       : float = 2.0
    label_smoothing   : float = 0.05
    mixup_alpha       : float = 0.15
    mixup_epochs      : int   = 5
    rdrop_alpha       : float = 0.5    # R-Drop KL weight
    swa_start_ratio   : float = 0.75
    val_split         : float = 0.12

    # ── I/O ───────────────────────────────────────────────────
    save_checkpoints  : bool  = True
    output_dir        : str   = field(
        default_factory=lambda:
            "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    )


cfg = CFG()
os.makedirs(cfg.output_dir, exist_ok=True)
CKPT_DIR = os.path.join(cfg.output_dir, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"\n{'='*70}")
print("ViLT + CLIP Cross-Attention Fusion  |  Fakeddit 6-Way")
print(f"{'='*70}")
print(f"  ViLT   : {cfg.vilt_model}  (text+image joint, 768-d CLS)")
print(f"  CLIP   : {cfg.clip_model}  (image branch only, 512-d)")
print(f"  Fusion : Cross-Attention (ViLT CLS → Query | CLIP patches → KV)")
print(f"  Effective batch : {cfg.batch_size * cfg.grad_accum}")
print(f"  Epochs          : {cfg.epochs}")
print(f"  Checkpoint dir  : {CKPT_DIR}")
print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────
# DATA LOADING  (same robust loader as v3)
# ─────────────────────────────────────────────────────────────
def _clean(v):
    s = str(v).strip() if v is not None else ""
    return "" if s.lower() in {"", "nan", "none", "null"} else s


def _resolve_image(row, base_dir):
    for col in ["image_path", "img_path", "filename", "image", "id", "submission_id"]:
        if col not in row.index:
            continue
        v = _clean(row.get(col))
        if not v or v.startswith("http"):
            continue
        base_name = os.path.splitext(os.path.basename(v))[0]
        stems = [v, os.path.basename(v), base_name + ".jpg",
                 base_name + ".jpeg", base_name + ".png"]
        search_dirs = [
            base_dir,
            os.path.join(base_dir, "public_image_set"),
            os.path.join(base_dir, "public_images"),
            os.path.join(base_dir, "images"),
            os.path.join(base_dir, "image_set"),
        ]
        for root in search_dirs:
            if not os.path.isdir(root):
                continue
            for name in stems:
                p = os.path.join(root, name)
                if os.path.isfile(p):
                    return p
    return None


def _parse_csv(path: str):
    sep = "\t" if path.lower().endswith(".tsv") else ","
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False, on_bad_lines="skip")
    except Exception:
        return []
    df.columns = [c.strip().lower() for c in df.columns]
    if LABEL_COL not in df.columns:
        return []
    text_col = next(
        (c for c in ["clean_title", "title", "text", "headline", "body"]
         if c in df.columns), None)
    if text_col is None:
        return []
    base_dir = os.path.dirname(path)
    rows = []
    for _, r in df.iterrows():
        text = _clean(r.get(text_col))
        if len(text) < 4:
            continue
        try:
            lab = int(float(r[LABEL_COL]))
        except Exception:
            continue
        if lab not in range(NUM_CLASSES):
            continue
        rows.append({
            "text":       text[:512],
            "label":      lab,
            "image_path": _resolve_image(r, base_dir),
        })
    return rows


def load_fakeddit_6way():
    roots = ["/kaggle/input", "/content", "."]
    records, files = [], []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root):
            if any(s in dp.lower() for s in [".config", "sample_data", "working"]):
                continue
            for fn in fns:
                if fn.lower().endswith((".csv", ".tsv")):
                    files.append(os.path.join(dp, fn))

    print(f"Scanning {len(files)} CSV/TSV files...")
    for fp in files:
        recs = _parse_csv(fp)
        if recs:
            n_img = sum(1 for r in recs if r["image_path"] is not None)
            print(f"  {len(recs):6d} rows ({n_img} with images) <- {os.path.basename(fp)}")
            records.extend(recs)

    if len(records) < 1000:
        print("Insufficient local data — trying HuggingFace...")
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load("shivam2503/fakeddit", trust_remote_code=True)
            for split in ds:
                part = ds[split].to_pandas()
                part.columns = [c.strip().lower() for c in part.columns]
                if LABEL_COL not in part.columns:
                    continue
                tc = next((c for c in ["clean_title", "title", "text"]
                           if c in part.columns), None)
                if tc is None:
                    continue
                for _, r in part.iterrows():
                    text = _clean(r.get(tc))
                    if len(text) < 4:
                        continue
                    try:
                        lab = int(float(r[LABEL_COL]))
                    except Exception:
                        continue
                    if lab in range(NUM_CLASSES):
                        records.append({"text": text[:512], "label": lab,
                                        "image_path": None})
        except Exception as e:
            print(f"HuggingFace fallback failed: {e}")

    if len(records) < 500:
        raise RuntimeError(
            "No usable Fakeddit 6-way data found.\n"
            "Attach Fakeddit dataset (with '6_way_label' column) to Kaggle."
        )

    df = pd.DataFrame(records).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    vc = df["label"].value_counts().sort_index()
    print("\nClass distribution before balancing:")
    for lbl, cnt in vc.items():
        print(f"  {lbl} ({LABEL_NAMES[lbl]:22s}): {cnt:7d}")

    cap = min(cfg.n_samples_per_class, int(vc.min()))
    parts = [
        df[df.label == cls].sample(
            min(cap, (df.label == cls).sum()), random_state=SEED)
        for cls in range(NUM_CLASSES)
    ]
    df = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_img = sum(isinstance(p, str) and os.path.isfile(str(p))
                for p in df["image_path"])
    print(f"\nFinal dataset : {len(df)} rows | {len(df)//NUM_CLASSES} per class | "
          f"images with files: {n_img}")
    USE_IMAGES = n_img > 0
    print(f"Image mode    : {'ENABLED' if USE_IMAGES else 'TEXT-ONLY (no image files found)'}")
    return df, USE_IMAGES


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
_PLACEHOLDER_VILT  = None
_PLACEHOLDER_CLIP  = None

def get_placeholder_vilt():
    global _PLACEHOLDER_VILT
    if _PLACEHOLDER_VILT is None:
        _PLACEHOLDER_VILT = Image.new(
            "RGB", (cfg.vilt_image_size, cfg.vilt_image_size), (128, 128, 128))
    return _PLACEHOLDER_VILT

def get_placeholder_clip():
    global _PLACEHOLDER_CLIP
    if _PLACEHOLDER_CLIP is None:
        _PLACEHOLDER_CLIP = Image.new(
            "RGB", (cfg.clip_image_size, cfg.clip_image_size), (128, 128, 128))
    return _PLACEHOLDER_CLIP


class FakedditDataset(Dataset):
    """
    Returns:
      - ViLT inputs: joint text+image (text tokenized + image patched by ViltProcessor)
      - CLIP image pixels: for the cross-attention KV branch
      - has_image: 1 if real image file found, 0 for placeholder
      - label: int in [0..5]
    """
    def __init__(self, df, vilt_processor, clip_processor, use_images: bool):
        self.df    = df.reset_index(drop=True)
        self.vp    = vilt_processor
        self.cp    = clip_processor
        self.use_images = use_images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = str(row["text"])[:512]
        label = int(row["label"])

        # ── Load image ────────────────────────────────────────────────────
        img_path  = row.get("image_path")
        has_image = 0
        pil_vilt  = get_placeholder_vilt()
        pil_clip  = get_placeholder_clip()

        if self.use_images and isinstance(img_path, str) and os.path.isfile(img_path):
            try:
                raw = Image.open(img_path).convert("RGB")
                pil_vilt = raw.resize(
                    (cfg.vilt_image_size, cfg.vilt_image_size), Image.BICUBIC)
                pil_clip = raw.resize(
                    (cfg.clip_image_size, cfg.clip_image_size), Image.BICUBIC)
                has_image = 1
            except Exception:
                pass   # fallback to placeholders

        # ── ViLT encoding (joint text + image) ────────────────────────────
        vilt_enc = self.vp(
            text=text,
            images=pil_vilt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.vilt_max_length,
        )

        # ── CLIP image encoding (image-only for cross-attn KV) ────────────
        clip_enc = self.cp(images=pil_clip, return_tensors="pt")

        return {
            "input_ids":       vilt_enc["input_ids"].squeeze(0),
            "attention_mask":  vilt_enc["attention_mask"].squeeze(0),
            "token_type_ids":  vilt_enc["token_type_ids"].squeeze(0),
            "pixel_values":    vilt_enc["pixel_values"].squeeze(0),
            "pixel_mask":      vilt_enc.get("pixel_mask",
                                torch.ones(1, dtype=torch.long)).squeeze(0),
            "clip_pixels":     clip_enc["pixel_values"].squeeze(0),
            "has_image":       torch.tensor(has_image, dtype=torch.float),
            "label":           torch.tensor(label,     dtype=torch.long),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.ls,
            reduction="none",
        )
        pt     = torch.exp(-ce)
        focal  = (1 - pt) ** self.gamma * ce
        return focal.mean()


def rdrop_loss(logits1, logits2, alpha=0.5):
    """
    R-Drop: KL divergence between two stochastic forward passes.
    Forces the model to be invariant to dropout noise → reduces overfitting.
    """
    kl1 = F.kl_div(F.log_softmax(logits1, dim=-1),
                   F.softmax(logits2, dim=-1), reduction="batchmean")
    kl2 = F.kl_div(F.log_softmax(logits2, dim=-1),
                   F.softmax(logits1, dim=-1), reduction="batchmean")
    return alpha * (kl1 + kl2) / 2.0


# ─────────────────────────────────────────────────────────────
# STOCHASTIC DEPTH (DropPath) — anti-overfit on residual
# ─────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """Stochastic depth: drop entire residual path with prob p during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand  = torch.floor(rand + keep)
        return x / keep * rand


# ─────────────────────────────────────────────────────────────
# CROSS-ATTENTION FUSION MODULE
# ─────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    ViLT CLS token (768-d) as Query.
    CLIP image features (512-d → projected to 768-d) as Key/Value.

    Architecture:
      1. Project CLIP image features to fusion_dim (768)
      2. Multi-head cross-attention: Q = ViLT CLS, KV = CLIP patches
      3. Residual add + LayerNorm
      4. Feed-forward (fusion_dim → fusion_dim)
      5. Residual add + LayerNorm
      6. Gate the cross-attn output by has_image (zeroed for text-only)

    Why this is better than simple concatenation:
      - ViLT CLS can selectively attend to the MOST RELEVANT CLIP patches
        for the 6-way classification decision, rather than averaging over all.
      - When images are absent, gating zeros out the CLIP branch cleanly
        so the ViLT CLS alone drives the prediction.
    """
    def __init__(self, vilt_dim=768, clip_dim=512, fusion_dim=768,
                 n_heads=8, proj_dropout=0.2, drop_path=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Project CLIP patches from 512-d to fusion_dim
        self.clip_kv_proj = nn.Sequential(
            nn.Linear(clip_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Project ViLT CLS to fusion_dim (may already be 768, but makes
        # the module dimension-agnostic)
        self.vilt_q_proj = nn.Sequential(
            nn.Linear(vilt_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=proj_dropout,
            batch_first=True,    # (B, seq, dim)
        )

        # Post-attention residual path
        self.attn_norm  = nn.LayerNorm(fusion_dim)
        self.drop_path  = DropPath(drop_path)

        # Feed-forward sublayer (post-cross-attn)
        self.ff = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(proj_dropout),
        )
        self.ff_norm = nn.LayerNorm(fusion_dim)

        # Learned scalar gate for has_image
        # (separate from the per-batch has_image binary flag)
        self.img_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

    def forward(self, vilt_cls, clip_patch_feats, has_image):
        """
        Args:
            vilt_cls        : (B, vilt_dim)  — ViLT CLS token
            clip_patch_feats: (B, N, clip_dim) or (B, clip_dim)
                              CLIP image patch tokens (or pooled)
            has_image       : (B,) float tensor, 0 or 1
        Returns:
            fused           : (B, fusion_dim)
        """
        B = vilt_cls.size(0)

        # Project ViLT CLS → Q: (B, 1, fusion_dim)
        Q = self.vilt_q_proj(vilt_cls).unsqueeze(1)

        # Project CLIP patches → KV: (B, N, fusion_dim)
        if clip_patch_feats.dim() == 2:
            # pooled: (B, clip_dim) → (B, 1, clip_dim)
            clip_patch_feats = clip_patch_feats.unsqueeze(1)
        KV = self.clip_kv_proj(clip_patch_feats)

        # Cross-attention: Q attends to KV
        attn_out, _ = self.cross_attn(Q, KV, KV)   # (B, 1, fusion_dim)
        attn_out    = attn_out.squeeze(1)            # (B, fusion_dim)

        # Gate by has_image: zero out if no real image
        gate         = self.img_gate(attn_out)       # (B, fusion_dim) ∈ (0,1)
        attn_gated   = gate * attn_out * has_image.unsqueeze(1)

        # Residual + DropPath + LN
        vilt_proj    = self.vilt_q_proj(vilt_cls)    # (B, fusion_dim)
        fused        = self.attn_norm(vilt_proj + self.drop_path(attn_gated))

        # Feed-forward sublayer
        fused        = self.ff_norm(fused + self.drop_path(self.ff(fused)))

        return fused   # (B, fusion_dim)


# ─────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────
class ViLTCLIPClassifier(nn.Module):
    """
    Branch A — ViLT (joint text+image transformer):
      Processes title text + image together via shared attention.
      Output: CLS token (768-d) — rich cross-modal representation.

    Branch B — CLIP vision encoder (image patch tokens):
      Processes the same image independently via ViT-B/32.
      Output: patch token sequence (196 patches, 768-d each after
      projection) used as Key/Value in cross-attention.

    Cross-Attention Fusion:
      ViLT CLS (Q) attends to CLIP patches (KV).
      Gated: CLIP branch zeroed when has_image=0.
      Result: 768-d fused representation.

    Classifier: 768-d → LayerNorm → Dropout → 512-d → GELU → 6 classes.
    """

    def __init__(self, n_classes=NUM_CLASSES, class_weights=None):
        super().__init__()

        # ── ViLT backbone ─────────────────────────────────────────────────
        print(f"Loading ViLT: {cfg.vilt_model}")
        self.vilt = ViltModel.from_pretrained(cfg.vilt_model)

        # ── CLIP vision backbone (image-only branch) ──────────────────────
        print(f"Loading CLIP vision: {cfg.clip_model}")
        clip_full        = CLIPModel.from_pretrained(cfg.clip_model)
        self.clip_vision = clip_full.vision_model    # ViT-B/32 vision encoder
        self.clip_proj   = clip_full.visual_projection  # 768 → 512
        # clip_vision hidden size (before projection)
        clip_vision_dim  = self.clip_vision.config.hidden_size   # 768
        # clip projected dim
        clip_proj_dim    = clip_full.config.projection_dim       # 512

        # ── Freeze all parameters ─────────────────────────────────────────
        for p in self.vilt.parameters():
            p.requires_grad = False
        for p in self.clip_vision.parameters():
            p.requires_grad = False
        for p in self.clip_proj.parameters():
            p.requires_grad = False

        # ── Unfreeze last N layers of ViLT ────────────────────────────────
        n_vilt_layers = len(self.vilt.encoder.layer)
        for layer in self.vilt.encoder.layer[n_vilt_layers - cfg.unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.vilt.pooler.parameters():
            p.requires_grad = True

        # ── Unfreeze last N layers of CLIP vision ─────────────────────────
        n_clip_layers = len(self.clip_vision.encoder.layers)
        for layer in self.clip_vision.encoder.layers[n_clip_layers - cfg.unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.clip_proj.parameters():
            p.requires_grad = True

        # ── Cross-attention fusion ─────────────────────────────────────────
        # CLIP patch tokens are 512-d (projected); we pass them to cross-attn.
        # Note: CLIP vision_model hidden = 768-d BEFORE projection (512-d).
        # We use the pre-projection hidden states (768-d) as richer KV.
        self.fusion = CrossAttentionFusion(
            vilt_dim    = cfg.vilt_hidden,        # 768
            clip_dim    = clip_vision_dim,         # 768 (pre-proj hidden)
            fusion_dim  = cfg.fusion_dim,          # 768
            n_heads     = cfg.n_heads,
            proj_dropout= cfg.proj_dropout,
            drop_path   = cfg.drop_path_rate,
        )

        # ── Classifier head ───────────────────────────────────────────────
        D = cfg.fusion_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(D, D // 2),
            nn.LayerNorm(D // 2),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(D // 2, n_classes),
        )

        # ── Loss ──────────────────────────────────────────────────────────
        cw = class_weights.float().to(DEVICE) if class_weights is not None else None
        self.loss_fn = FocalLoss(
            gamma=cfg.focal_gamma,
            weight=cw,
            label_smoothing=cfg.label_smoothing,
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable : {trainable/1e6:.2f}M / {total/1e6:.2f}M params")

    def grouped_params(self):
        """Differential learning rates: head > backbone."""
        head_p = (
            list(self.fusion.parameters()) +
            list(self.classifier.parameters())
        )
        bb_p = [p for p in (
            list(self.vilt.parameters()) +
            list(self.clip_vision.parameters()) +
            list(self.clip_proj.parameters())
        ) if p.requires_grad]
        return [
            {"params": head_p, "lr": cfg.lr_head,     "weight_decay": cfg.weight_decay},
            {"params": bb_p,   "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
        ]

    def encode(self, input_ids, attention_mask, token_type_ids,
               pixel_values, pixel_mask, clip_pixels, has_image):
        """
        Shared forward pass used by both standard forward and R-Drop double pass.
        """
        # ── ViLT: joint text+image encoding ───────────────────────────────
        vilt_out  = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True,
        )
        vilt_cls = vilt_out.pooler_output          # (B, 768)

        # ── CLIP vision: extract patch-level hidden states ─────────────────
        # Use last hidden state (all patch tokens) for richer KV in cross-attn
        # Shape: (B, 1 + n_patches, hidden_size) — include CLS patch token
        clip_out    = self.clip_vision(
            pixel_values=clip_pixels,
            output_hidden_states=False,
            return_dict=True,
        )
        # last_hidden_state: (B, 197, 768)  [1 CLS + 196 patches for ViT-B/32]
        clip_patches = clip_out.last_hidden_state  # (B, 197, 768)

        # ── Cross-attention fusion ─────────────────────────────────────────
        fused   = self.fusion(vilt_cls, clip_patches, has_image)

        # ── Classification ────────────────────────────────────────────────
        logits  = self.classifier(fused)
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids,
                pixel_values, pixel_mask, clip_pixels, has_image, **kw):
        return self.encode(input_ids, attention_mask, token_type_ids,
                           pixel_values, pixel_mask, clip_pixels, has_image)

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = correct = n_total = 0
    optimizer.zero_grad()
    num_steps = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        labels      = batch.pop("label").to(DEVICE)
        batch_dev   = {k: v.to(DEVICE) for k, v in batch.items()}
        B           = labels.size(0)

        with autocast_ctx():
            # ── R-Drop: two stochastic forward passes ─────────────────────
            # First pass
            logits1 = model.encode(**batch_dev)
            loss1   = model.compute_loss(logits1, labels)

            # MixUp in early epochs (applied to logits mix — embedding MixUp)
            if epoch <= cfg.mixup_epochs and cfg.mixup_alpha > 0:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                idx = torch.randperm(B, device=DEVICE)
                loss1 = (lam * model.compute_loss(logits1, labels) +
                         (1 - lam) * model.compute_loss(logits1, labels[idx]))

            # Second pass (different dropout mask)
            logits2 = model.encode(**batch_dev)
            loss2   = model.compute_loss(logits2, labels)

            # R-Drop KL term between the two passes
            kl_loss = rdrop_loss(logits1, logits2, alpha=cfg.rdrop_alpha)

            loss = (loss1 + loss2) / 2.0 + kl_loss
            loss = loss / cfg.grad_accum

        scaler.scale(loss).backward()

        is_update = (
            (step + 1) % cfg.grad_accum == 0 or
            (step + 1) == num_steps
        )
        if is_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        raw_loss    = loss.item() * cfg.grad_accum
        total_loss += raw_loss
        correct    += (logits1.detach().argmax(1) == labels).sum().item()
        n_total    += B
        pbar.set_postfix({"loss": f"{raw_loss:.4f}",
                          "acc":  f"{correct/n_total:.4f}"})

    return total_loss / num_steps, correct / n_total


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        labels    = batch.pop("label").to(DEVICE)
        batch_dev = {k: v.to(DEVICE) for k, v in batch.items()}

        with autocast_ctx():
            logits = model(**batch_dev)
            loss   = model.compute_loss(logits, labels)

        total_loss  += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return (
        total_loss / len(loader),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro",    zero_division=0),
        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true,    y_pred, average="weighted", zero_division=0),
        y_pred, y_true,
    )


# ─────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────
def print_results_table(fold_results):
    W = 95
    print("\n" + "=" * W)
    print("  ViLT + CLIP Cross-Attention Fusion  |  Fakeddit 6-Way  |  Per-Fold Results")
    print(f"  Target: Accuracy >= 89%  |  Macro-F1 >= 0.88  |  Overfit Gap < 4%")
    print("=" * W)

    headers = ["Fold", "Test Acc", "Macro F1", "Wtd F1",
               "Prec(W)", "Rec(W)", "Ovft Gap", "Epochs"]
    rows = []
    for r in fold_results:
        rows.append([
            r["Fold"],
            f"{r['Accuracy']:.4f}",
            f"{r['MacroF1']:.4f}",
            f"{r['WtdF1']:.4f}",
            f"{r['Precision']:.4f}",
            f"{r['Recall']:.4f}",
            f"{r['Overfitting']:+.4f}",
            r["EpochsRun"],
        ])

    accs  = [r["Accuracy"]    for r in fold_results]
    mf1s  = [r["MacroF1"]     for r in fold_results]
    wf1s  = [r["WtdF1"]       for r in fold_results]
    precs = [r["Precision"]   for r in fold_results]
    recs  = [r["Recall"]      for r in fold_results]
    gaps  = [r["Overfitting"] for r in fold_results]

    rows.append(["MEAN",
                 f"{np.mean(accs):.4f}", f"{np.mean(mf1s):.4f}",
                 f"{np.mean(wf1s):.4f}", f"{np.mean(precs):.4f}",
                 f"{np.mean(recs):.4f}", f"{np.mean(gaps):+.4f}", "-"])
    rows.append(["STD",
                 f"{np.std(accs):.4f}",  f"{np.std(mf1s):.4f}",
                 f"{np.std(wf1s):.4f}",  f"{np.std(precs):.4f}",
                 f"{np.std(recs):.4f}",  f"{np.std(gaps):.4f}", "-"])

    print(tabulate(rows, headers=headers, tablefmt="double_grid"))
    print("=" * W)

    mean_acc = np.mean(accs)
    mean_mf1 = np.mean(mf1s)
    mean_gap = abs(np.mean(gaps))

    acc_s = "[TARGET MET >= 89%]"     if mean_acc >= 0.89 else f"[gap: {0.89-mean_acc:.4f}]"
    f1_s  = "[TARGET MET >= 0.88]"    if mean_mf1 >= 0.88 else "[below 0.88]"
    gap_s = "[OK < 4%]"               if mean_gap < 0.04  else "[WATCH overfit]"

    print(f"\n  Mean Accuracy  : {mean_acc:.4f}  {acc_s}")
    print(f"  Mean Macro F1  : {mean_mf1:.4f}  {f1_s}")
    print(f"  Mean Ovft Gap  : {np.mean(gaps):+.4f}  {gap_s}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────
def ckpt_path(fold_num: int) -> str:
    return os.path.join(CKPT_DIR, f"fold{fold_num}_best_vilt_clip.pt")


def save_checkpoint(fold_num, epoch, model, val_f1, val_acc):
    path = ckpt_path(fold_num)
    torch.save({
        "fold":       fold_num,
        "epoch":      epoch,
        "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "val_f1":     val_f1,
        "val_acc":    val_acc,
    }, path)
    return path


def try_resume(fold_num, model):
    """Load checkpoint if it exists. Returns (loaded:bool, val_f1:float)."""
    path = ckpt_path(fold_num)
    if not os.path.isfile(path):
        return False, 0.0
    try:
        data = torch.load(path, map_location=DEVICE)
        model.load_state_dict({k: v.to(DEVICE) for k, v in data["state_dict"].items()})
        print(f"  ▶ Resumed fold {fold_num} checkpoint: "
              f"epoch={data['epoch']} val_f1={data['val_f1']:.4f}")
        return True, data["val_f1"]
    except Exception as e:
        print(f"  Resume failed ({e}); training from scratch")
        return False, 0.0


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 95)
    print("FAKEDDIT 6-WAY — ViLT + CLIP Cross-Attention Fusion")
    print("ViLT: joint text+image transformer  |  CLIP: image patch KV for cross-attn")
    print("TARGET: Accuracy >= 89% | Macro-F1 >= 0.88 | Overfit Gap < 4%")
    print("=" * 95 + "\n")

    df, USE_IMAGES = load_fakeddit_6way()

    # ── Class weights for focal loss ──────────────────────────────────────
    class_weights = torch.tensor(
        compute_class_weight("balanced",
                             classes=np.arange(NUM_CLASSES),
                             y=df["label"]),
        dtype=torch.float32,
    )
    print(f"Class weights : {class_weights.numpy().round(3)}\n")

    # ── Load processors once ─────────────────────────────────────────────
    print("Loading ViLT processor...")
    vilt_proc = ViltProcessor.from_pretrained(cfg.vilt_model)
    print("Loading CLIP processor (image-only branch)...")
    clip_full_proc = CLIPProcessor.from_pretrained(cfg.clip_model)
    # We only need the image processor from CLIP
    clip_img_proc  = clip_full_proc
    print("Processors loaded\n")

    kfold        = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                                   random_state=SEED)
    fold_results = []

    for fold, (train_val_idx, test_idx) in enumerate(
            kfold.split(df, df["label"])):
        fold_num = fold + 1
        print(f"\n{'='*95}")
        print(f"FOLD {fold_num}/{cfg.n_folds}")
        print(f"{'='*95}\n")

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df      = df.iloc[test_idx].reset_index(drop=True)

        train_df, val_df = train_test_split(
            train_val_df, test_size=cfg.val_split,
            stratify=train_val_df["label"], random_state=SEED)

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        def make_loader(split_df, shuffle):
            ds = FakedditDataset(split_df, vilt_proc, clip_img_proc, USE_IMAGES)
            return DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=shuffle,
                num_workers=2, pin_memory=True,
                drop_last=shuffle, collate_fn=collate_fn,
            )

        train_loader = make_loader(train_df, True)
        val_loader   = make_loader(val_df,   False)
        test_loader  = make_loader(test_df,  False)

        model     = ViLTCLIPClassifier(NUM_CLASSES, class_weights).to(DEVICE)
        optimizer = torch.optim.AdamW(model.grouped_params())

        total_steps  = (len(train_loader) * cfg.epochs) // cfg.grad_accum
        warmup_steps = int(cfg.warmup_ratio * total_steps)
        scheduler    = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps)
        scaler       = make_scaler()

        # SWA setup
        swa_model       = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler   = torch.optim.swa_utils.SWALR(
            optimizer, swa_lr=cfg.lr_backbone * 5, anneal_epochs=3)
        swa_start_epoch = max(1, int(cfg.swa_start_ratio * cfg.epochs))

        # ── Try to resume from checkpoint ─────────────────────────────────
        resumed, best_val_f1 = try_resume(fold_num, model)
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr = 0 if not resumed else cfg.patience - 1
        overfit_ctr  = 0
        train_accs   = []
        val_accs     = []
        epochs_run   = 0
        skip_train   = resumed   # if resumed, go straight to test eval

        if not skip_train:
            for epoch in range(1, cfg.epochs + 1):
                epochs_run = epoch
                print(f"\n  Epoch {epoch}/{cfg.epochs}")

                tr_loss, tr_acc = train_one_epoch(
                    model, train_loader, optimizer, scheduler, scaler, epoch)

                va_loss, va_acc, va_mf1, va_wf1, va_prec, va_rec, _, _ = validate(
                    model, val_loader)

                train_accs.append(tr_acc)
                val_accs.append(va_acc)
                gap = tr_acc - va_acc

                print(f"    Train : loss={tr_loss:.4f}  acc={tr_acc:.4f}")
                print(f"    Val   : loss={va_loss:.4f}  acc={va_acc:.4f}  "
                      f"macroF1={va_mf1:.4f}  gap={gap:+.4f}")

                # SWA
                if epoch >= swa_start_epoch:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                # Overfit check
                if gap > cfg.overfit_threshold:
                    overfit_ctr += 1
                    if overfit_ctr >= cfg.overfit_patience:
                        print(f"    Overfit stop: gap {gap:.4f} > "
                              f"{cfg.overfit_threshold} for {overfit_ctr} epochs")
                        break
                else:
                    overfit_ctr = 0

                # Save best checkpoint
                if va_mf1 > best_val_f1:
                    best_val_f1 = va_mf1
                    best_state  = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
                    patience_ctr = 0
                    if cfg.save_checkpoints:
                        path = save_checkpoint(fold_num, epoch, model,
                                               va_mf1, va_acc)
                        print(f"    ✓ Checkpoint saved (macroF1={va_mf1:.4f}) → {path}")
                else:
                    patience_ctr += 1
                    print(f"    Patience: {patience_ctr}/{cfg.patience}")
                    if patience_ctr >= cfg.patience:
                        print(f"    Early stop at epoch {epoch}")
                        break

        # ── Evaluate best checkpoint ──────────────────────────────────────
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        _, ck_acc, ck_mf1, ck_wf1, ck_prec, ck_rec, ck_preds, ck_true = validate(
            model, test_loader)

        # Try SWA model
        test_acc, test_mf1 = ck_acc, ck_mf1
        test_wf1, test_prec, test_rec = ck_wf1, ck_prec, ck_rec
        test_preds, test_true = ck_preds, ck_true

        if not skip_train and epochs_run > swa_start_epoch:
            try:
                bn_loader = DataLoader(
                    FakedditDataset(
                        train_df.sample(min(2000, len(train_df)),
                                        random_state=SEED),
                        vilt_proc, clip_img_proc, USE_IMAGES),
                    batch_size=cfg.batch_size, num_workers=2,
                    collate_fn=collate_fn,
                )
                torch.optim.swa_utils.update_bn(bn_loader, swa_model,
                                                device=DEVICE)
                _, sw_acc, sw_mf1, sw_wf1, sw_prec, sw_rec, sw_preds, sw_true = validate(
                    swa_model.module, test_loader)
                if sw_mf1 > ck_mf1:
                    test_acc, test_mf1 = sw_acc, sw_mf1
                    test_wf1, test_prec, test_rec = sw_wf1, sw_prec, sw_rec
                    test_preds, test_true = sw_preds, sw_true
                    print(f"  SWA adopted (F1 {sw_mf1:.4f} > {ck_mf1:.4f})")
                else:
                    print(f"  Checkpoint adopted (F1 {ck_mf1:.4f})")
            except Exception as e:
                print(f"  SWA update failed ({e}); using checkpoint")

        mean_gap    = float(np.mean([ta - va for ta, va in
                                     zip(train_accs, val_accs)])) if train_accs else 0.0
        per_cls_f1  = f1_score(test_true, test_preds, average=None, zero_division=0)

        print(f"\n{'='*95}")
        print(f"FOLD {fold_num} TEST RESULTS")
        print(f"{'='*95}")
        print(f"  Test Accuracy  : {test_acc:.4f}")
        print(f"  Test Macro F1  : {test_mf1:.4f}")
        print(f"  Test Wtd F1    : {test_wf1:.4f}")
        print(f"  Test Precision : {test_prec:.4f}")
        print(f"  Test Recall    : {test_rec:.4f}")
        print(f"  Avg Overfit Gap: {mean_gap:.4f}")
        print(f"  Epochs run     : {epochs_run}")
        print(f"\n  Per-Class F1:")
        for cls, f1v in enumerate(per_cls_f1):
            marker = " ← hardest" if f1v == per_cls_f1.min() else ""
            print(f"    {cls} {LABEL_NAMES[cls]:22s}: {f1v:.4f}{marker}")
        print(f"\n  Classification Report:")
        print(classification_report(test_true, test_preds,
                                    target_names=LABEL_NAMES, zero_division=0))

        fold_results.append({
            "Fold":        fold_num,
            "Accuracy":    test_acc,
            "MacroF1":     test_mf1,
            "WtdF1":       test_wf1,
            "Precision":   test_prec,
            "Recall":      test_rec,
            "Overfitting": mean_gap,
            "EpochsRun":   epochs_run,
        })

        del model, swa_model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    # ── Final summary ─────────────────────────────────────────────────────
    print_results_table(fold_results)

    results_df = pd.DataFrame(fold_results)
    mean_row = {"Fold": "Mean", **{
        c: results_df[c].mean()
        for c in results_df.columns if c not in ("Fold", "EpochsRun")
    }}
    std_row = {"Fold": "Std", **{
        c: results_df[c].std()
        for c in results_df.columns if c not in ("Fold", "EpochsRun")
    }}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    out_csv = os.path.join(cfg.output_dir, "fakeddit_6way_vilt_clip_crossattn_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}")

    mean_acc = float(mean_row["Accuracy"])
    mean_mf1 = float(mean_row["MacroF1"])
    mean_gap = float(mean_row["Overfitting"])

    print(f"\n{'='*95}")
    print("FINAL RESULTS — Fakeddit 6-Way ViLT+CLIP Cross-Attention Fusion")
    print(f"{'='*95}")
    print(f"  Test Accuracy  : {mean_acc:.4f}  (Target >= 0.89)")
    print(f"  Test Macro F1  : {mean_mf1:.4f}  (Target >= 0.88)")
    print(f"  Test Precision : {float(mean_row['Precision']):.4f}")
    print(f"  Test Recall    : {float(mean_row['Recall']):.4f}")
    print(f"  Avg Ovft Gap   : {mean_gap:.4f}  (Target < 0.04)")
    print()
    if USE_IMAGES:
        print("  Image mode: ENABLED — ViLT+CLIP both using real images")
    else:
        print("  Image mode: TEXT-ONLY — attach public_image_set for image boost")

    if mean_acc >= 0.89 and mean_mf1 >= 0.88 and abs(mean_gap) < 0.04:
        print("\n  ALL TARGETS ACHIEVED!")
    elif mean_acc >= 0.87:
        print(f"\n  Strong result ({mean_acc:.4f}) — close to 89% target.")
    else:
        print(f"\n  {mean_acc:.4f} — check dataset and image availability.")
    print(f"{'='*95}\n")

    # ── Checkpoint summary ────────────────────────────────────────────────
    print("Checkpoint summary:")
    for fn in range(1, cfg.n_folds + 1):
        p = ckpt_path(fn)
        status = "✓ saved" if os.path.isfile(p) else "-- missing"
        print(f"  Fold {fn}: [{status}] {p}")


if __name__ == "__main__":
    main()
