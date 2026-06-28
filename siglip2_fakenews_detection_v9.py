# ============================================================
#  SigLIP-2  ×  FAKEDDIT  —  Fake News Detection  v5
#  Binary Classification | 5-Fold CV | Kaggle-Ready
#  Target: ≥ 90% accuracy | Overfit gap < 5%
# ============================================================
#
#  OVERFITTING FIXES vs v3 (gap was +24%):
#  ─────────────────────────────────────────────────────────
#  FIX 1  freeze_layers 6 → 10   (freeze almost all backbone)
#  FIX 2  lr 2e-5 → 5e-6         (4× smaller, less memorisation)
#         head gets 10× higher LR via param groups
#  FIX 3  Mixup applied to pixel values ONLY (not token IDs —
#         integers cannot be linearly interpolated)
#  FIX 4  dropout 0.30 → 0.50 in head
#  FIX 5  grad_accum 2→4  (effective batch 16→64, smoother grad)
#  FIX 6  StochasticDepth (LayerDrop) on transformer blocks
#  FIX 7  label_smoothing 0.10 → 0.15
#  FIX 8  early_stop patience 4 → 3, per-epoch LR scheduler
#  FIX 9  Dataset: real FAKEDDIT rows parsed correctly from
#         Uni-Fakeddit-55k single "text" column format
#  FIX 10 Added R-Drop regularisation (KL between two forward
#         passes with different dropout masks)
#
#  HOW TO RUN ON KAGGLE
#  ─────────────────────
#  1. New Notebook → Accelerator: GPU T4 x2
#  2. Settings → Internet: ON
#  3. (Optional) Add-ons → Secrets → key=HF_TOKEN
#  4. Paste entire script into one cell → Run All
# ============================================================

# ── Cell 0: Installs ────────────────────────────────────────
import subprocess, sys

def pip(*args):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *args])

pip("transformers>=4.41.0", "datasets", "accelerate",
    "scikit-learn", "pandas", "Pillow", "requests")

# ── Imports ─────────────────────────────────────────────────
import os, gc, io, re, random, warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoProcessor, AutoModel,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  0.  HF TOKEN
# ══════════════════════════════════════════════════════════════
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("  ✔ HF token from Kaggle secrets")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    print("  ✔ HF token from env" if HF_TOKEN
          else "  ℹ No HF token — public access only")

# ══════════════════════════════════════════════════════════════
#  1.  CONFIG  (all anti-overfit values annotated)
# ══════════════════════════════════════════════════════════════
# ── v9 tuning rationale ────────────────────────────────────────
#  v8 achieved only 68% — over-regularised, backbone too frozen.
#  Root causes:
#   • freeze_layers=10/12 → only 2 layers trainable; model starved
#   • dropout=0.50 on a 1536-dim input → too destructive
#   • rdrop_alpha=1.0 + mixup=0.40 simultaneously → over-smoothed
#   • backbone_lr=5e-6 far too low for only 2 unfrozen layers
#   • label_smoothing=0.15 too high for binary task
#   • early_stop_patience=3 stops before convergence
#   • n_samples=5000 fine but text max_length=64 truncates headlines
#  Fix: unfreeze 6 layers, stronger LR, lighter regularisation,
#       more epochs/patience, longer text, more samples.
# ──────────────────────────────────────────────────────────────
CFG = dict(
    model_name          = "google/siglip2-base-patch16-224",

    # Data — more samples → better generalisation
    n_samples           = 8000,     # ↑ was 5000
    n_folds             = 5,
    train_ratio         = 0.80,
    val_ratio           = 0.10,

    # Training
    epochs              = 15,       # ↑ was 10 — need more steps
    batch_size          = 16,
    grad_accum          = 4,        # effective batch = 64
    backbone_lr         = 2e-5,     # ↑ was 5e-6; more layers trainable
    head_lr             = 1e-4,     # ↑ was 5e-5
    weight_decay        = 1e-2,     # ↓ was 5e-2; less L2 squeeze
    label_smoothing     = 0.05,     # ↓ was 0.15; binary task needs crisp signal
    warmup_ratio        = 0.10,
    max_grad_norm       = 1.0,      # ↑ was 0.5; allow larger updates

    # Regularisation — lighter so model can actually learn
    dropout             = 0.30,     # ↓ was 0.50
    freeze_layers       = 6,        # ↓ was 10; unfreeze top 6 layers
    stoch_depth_prob    = 0.05,     # ↓ was 0.10; lighter drop

    # Anti-overfit: training — balanced, not stacked
    mixup_alpha         = 0.20,     # ↓ was 0.40; lighter mixup
    rdrop_alpha         = 0.50,     # ↓ was 1.0; lighter R-Drop
    early_stop_patience = 5,        # ↑ was 3; wait longer

    # Misc
    seed                = 42,
    num_workers         = 2,
    image_size          = 224,
    max_text_len        = 128,      # ↑ was 64; capture full headline
    device              = "cuda" if torch.cuda.is_available() else "cpu",
    fp16                = torch.cuda.is_available(),
)

print(f"  Device : {CFG['device']}  |  FP16 : {CFG['fp16']}")
print(f"  Effective batch : {CFG['batch_size'] * CFG['grad_accum']}")


# ══════════════════════════════════════════════════════════════
#  2.  SEED
# ══════════════════════════════════════════════════════════════
def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(CFG["seed"])


# ══════════════════════════════════════════════════════════════
#  3.  DATASET LOADING  (3-tier fallback, fixed column parsing)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Loading FAKEDDIT …")
print("=" * 60)

# ── parser for Uni-Fakeddit-55k single-column format ─────────
#  Format: "[TEXT] headline [OBJECTS] obj1,obj2 [LABEL] 1"
_TEXT_RE  = re.compile(r'\[TEXT\]\s*(.*?)(?=\[OBJECTS\]|\[LABEL\]|$)',
                       re.S)
_LABEL_RE = re.compile(r'\[LABEL\]\s*([01])')

def parse_row(s: str):
    tm = _TEXT_RE.search(s)
    lm = _LABEL_RE.search(s)
    if tm and lm:
        return tm.group(1).strip(), int(lm.group(1))
    return None, None

df = None

# ── Tier 1: HF mirror ───────────────────────────────────────
try:
    from datasets import load_dataset
    raw  = load_dataset("ArkaMukherjee/Uni-Fakeddit-55k",
                        split="train")
    col  = raw.column_names[0]          # "text"
    rows = []
    for row in raw:
        h, lbl = parse_row(str(row[col]))
        if h is not None:
            rows.append({"text": h, "image_url": "", "label": lbl})
    df = pd.DataFrame(rows)
    print(f"  ✔ HF mirror: {len(df)} rows parsed")
except Exception as e:
    print(f"  ✗ HF mirror: {e}")

# ── Tier 2: GitHub TSVs ─────────────────────────────────────
if df is None or len(df) < CFG["n_samples"]:
    import requests as req
    TSV_URLS = [
        ("https://raw.githubusercontent.com/entitize/Fakeddit/"
         "master/data/multimodal_only_samples/multimodal_train.tsv"),
        ("https://raw.githubusercontent.com/entitize/Fakeddit/"
         "master/data/all_samples/multimodal_train.tsv"),
    ]
    for url in TSV_URLS:
        try:
            r = req.get(url, timeout=20); r.raise_for_status()
            from io import StringIO
            tmp = pd.read_csv(StringIO(r.text), sep="\t",
                              on_bad_lines="skip")
            tc = next((c for c in tmp.columns if "title" in c.lower()), None)
            lc = next((c for c in tmp.columns if "2_way" in c.lower()), None)
            ic = next((c for c in tmp.columns if "image" in c.lower()), None)
            if tc and lc:
                df = pd.DataFrame({
                    "text"     : tmp[tc].astype(str),
                    "label"    : tmp[lc].astype(int),
                    "image_url": tmp[ic].astype(str) if ic else "",
                }).dropna()
                print(f"  ✔ TSV: {len(df)} rows"); break
        except Exception as e:
            print(f"  ✗ TSV {url.split('/')[-1]}: {e}")

# ── Tier 3: synthetic (architecture demo) ───────────────────
if df is None or len(df) < 100:
    print("  ⚠  Synthetic data — accuracy targets are indicative only")
    FAKE = [
        "BREAKING: Scientists discover miracle cure suppressed by pharmaceutical companies",
        "Government secretly adding mind-control chemicals to tap water",
        "Leaked documents prove the moon landing was filmed on a Hollywood set",
        "Billionaire funds secret underground city for global elites after collapse",
        "New pill lets you eat anything you want and still burn fat overnight",
        "Doctor reveals the one weird trick Big Pharma does not want you to know",
        "Alien spacecraft spotted over Area 51 witnesses claim military cover-up",
        "Exclusive: president signs secret executive order banning free speech online",
        "Scientists confirm 5G towers designed to track population movements globally",
        "Major bank prepares to confiscate all customer savings next month report says",
    ]
    REAL = [
        "Senate passes bipartisan infrastructure bill after months of negotiation",
        "Federal Reserve raises benchmark interest rate by a quarter percentage point",
        "City council approves affordable housing development in downtown district",
        "Mediterranean diet associated with lower risk of cardiovascular disease",
        "Technology company reports record revenue growth in quarterly earnings call",
        "Scientists sequence genome of newly discovered deep sea organism",
        "Local school district expands mental health counselling services for students",
        "Transportation department releases updated highway construction safety rules",
        "University researchers develop new battery technology with longer lifespan",
        "Health officials report decline in seasonal influenza cases this year",
    ]
    n = CFG["n_samples"]
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append({"text"     : FAKE[i % len(FAKE)] + f" [ref-{i}]",
                         "image_url": "", "label": 1})
        else:
            rows.append({"text"     : REAL[i % len(REAL)] + f" [ref-{i}]",
                         "image_url": "", "label": 0})
    df = pd.DataFrame(rows)

# ── Stratified subsample ─────────────────────────────────────
half = CFG["n_samples"] // 2
df   = pd.concat([
    resample(df[df["label"]==0],
             n_samples=min(half, (df["label"]==0).sum()),
             random_state=CFG["seed"]),
    resample(df[df["label"]==1],
             n_samples=min(half, (df["label"]==1).sum()),
             random_state=CFG["seed"]),
]).sample(frac=1, random_state=CFG["seed"]).reset_index(drop=True)

print(f"\n  Samples : {len(df)}")
print(f"  Labels  :\n{df['label'].value_counts().to_string()}\n")


# ══════════════════════════════════════════════════════════════
#  4.  IMAGE HELPER
# ══════════════════════════════════════════════════════════════
import requests as _req

def safe_image(url: str, sz: int = 224) -> Image.Image:
    blank = Image.new("RGB", (sz, sz), (128, 128, 128))
    if not isinstance(url, str) or not url.strip():
        return blank
    try:
        r = _req.get(url, timeout=4)
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return blank


# ══════════════════════════════════════════════════════════════
#  5.  PROCESSOR
# ══════════════════════════════════════════════════════════════
print("  Loading SigLIP-2 processor …")
processor = AutoProcessor.from_pretrained(
    CFG["model_name"], token=HF_TOKEN)
print("  ✔ Processor ready\n")


# ══════════════════════════════════════════════════════════════
#  6.  DATASET
#
#  KEY NOTES:
#  • SigLIP-2 processor does NOT return attention_mask  → skip it
#  • Augmentations on PIL image before processor
#  • text kept ≤ 128 chars; processor pads to max_length=64 tokens
# ══════════════════════════════════════════════════════════════
from torchvision import transforms as T

TRAIN_AUG = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.RandomAffine(degrees=15, translate=(0.10, 0.10),
                   scale=(0.85, 1.15)),
    T.ColorJitter(brightness=0.30, contrast=0.30,
                  saturation=0.30, hue=0.10),
    T.RandomGrayscale(p=0.10),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.RandomPerspective(distortion_scale=0.15, p=0.3),
])

class FakedditDataset(Dataset):
    def __init__(self, dataframe, processor, augment=False):
        self.df        = dataframe.reset_index(drop=True)
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
        text  = str(row["text"])[:128]
        img   = safe_image(str(row.get("image_url", "")),
                           CFG["image_size"])
        if self.augment:
            img = TRAIN_AUG(img)

        enc = self.processor(
            text           = [text],
            images         = [img],
            return_tensors = "pt",
            padding        = "max_length",
            truncation     = True,
            max_length     = CFG.get("max_text_len", 64),
        )
        item = {
            "input_ids"   : enc["input_ids"].squeeze(0),
            "pixel_values": enc["pixel_values"].squeeze(0),
            "label"       : torch.tensor(label, dtype=torch.long),
        }
        if "attention_mask" in enc:
            item["attention_mask"] = enc["attention_mask"].squeeze(0)
        return item


# ══════════════════════════════════════════════════════════════
#  7.  STOCHASTIC DEPTH — implemented via forward hooks
#
#  FIX (v8): Previous wrapper-class approaches broke the SigLIP
#  Transformers encoder loop because the loop calls each layer and
#  unpacks its output as  hidden_states = layer_output[0].
#  Replacing a layer with a wrapper class changes the call
#  signature in subtle ways that differ between vision and text
#  encoders (different argument lists, output_attentions handling,
#  etc.) and causes layer_norm to receive a tuple instead of a
#  tensor.
#
#  The hook approach is completely transparent: the real layer runs
#  normally (correct I/O types guaranteed), and the hook
#  ONLY scales the residual delta post-hoc.  No wrapper class,
#  no signature changes, no tuple confusion.
#
#  How it works:
#    output_tensor = layer(input_tensor, ...)
#    residual      = output_tensor[0] - input_tensor   ← the delta
#    if drop:  output_tensor[0] = input_tensor          ← zero delta
#    else:     output_tensor[0] = input_tensor + residual  ← keep
#  This is mathematically equivalent to stochastic depth.
# ══════════════════════════════════════════════════════════════
def _register_stoch_depth_hooks(model, frozen_up_to: int,
                                p: float, handles: list):
    """
    Attach forward hooks to unfrozen encoder layers in both
    vision_model and text_model.  Each hook randomly drops the
    residual connection of that layer during training.
    All hooks are stored in `handles` so they can be removed later.
    """
    def make_hook(p_drop):
        def hook(module, args, output):
            # Only active during training
            if not module.training:
                return output
            if torch.rand(1).item() >= p_drop:
                return output          # keep this layer normally

            # Recover the layer input from the hook args.
            # SigLIP encoder layers receive hidden_states as
            # the first positional argument.
            layer_input = args[0]
            if isinstance(layer_input, tuple):
                layer_input = layer_input[0]

            # output is a tuple; index 0 is the hidden_states tensor
            if isinstance(output, tuple):
                # Zero the residual: replace output[0] with input
                return (layer_input,) + output[1:]
            else:
                return layer_input
        return hook

    for attr in ["vision_model", "text_model"]:
        tower = getattr(model, attr, None)
        if tower is None:
            continue
        enc = getattr(tower, "encoder", None)
        if enc is None:
            continue
        layers = getattr(enc, "layers", None)
        if layers is None:
            continue
        for i in range(frozen_up_to, len(layers)):
            h = layers[i].register_forward_hook(make_hook(p))
            handles.append(h)


# ══════════════════════════════════════════════════════════════
#  8.  MODEL
# ══════════════════════════════════════════════════════════════
class SigLIP2Classifier(nn.Module):
    """
    Dual-encoder SigLIP-2 + regularised classification head.

    Anti-overfit design:
    • freeze_layers=10  → top 2 blocks + head fine-tune only
    • StochasticDepth on unfrozen transformer blocks
    • Dropout(0.50) in head
    • Separate LR groups: backbone_lr << head_lr
    • image_embeds and text_embeds are already L2-normed by SigLIP-2
    """
    def __init__(self, model_name, num_classes=2,
                 dropout=0.50, freeze_layers=10,
                 stoch_depth_p=0.10, token=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name, token=token)

        self._freeze(freeze_layers)
        self._add_stoch_depth(freeze_layers, stoch_depth_p)

        # FIX (v8): get_image_features() on some transformers versions
        # returns BaseModelOutputWithPooling, not a tensor.
        # Use pooler_output from vision_model directly — it is the
        # post_layernorm-normalised CLS/mean-pool vector and is always
        # a plain tensor.  D = vision hidden_size.
        try:
            D = self.backbone.config.vision_config.hidden_size
        except AttributeError:
            D = 768

        # Gated fusion: learn how much to trust image vs text
        self.gate = nn.Sequential(
            nn.Linear(D * 2, 2),
            nn.Softmax(dim=-1),
        )
        # Classifier head — slightly wider, batch-normed for stability
        self.head = nn.Sequential(
            nn.LayerNorm(D * 2),
            nn.Dropout(dropout),
            nn.Linear(D * 2, 768),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(256, num_classes),
        )

    def _freeze(self, n):
        """Freeze first n layers of both towers + their embeddings."""
        for attr in ["vision_model", "text_model"]:
            tower = getattr(self.backbone, attr, None)
            if tower is None:
                continue
            # Freeze embedding layers entirely
            for sub in ["embeddings", "patch_embedding"]:
                emb = getattr(tower, sub, None)
                if emb is not None:
                    for p in emb.parameters():
                        p.requires_grad = False
            enc    = getattr(tower, "encoder", None)
            if enc is None:
                continue
            layers = getattr(enc, "layers", [])
            for i, layer in enumerate(layers):
                if i < n:
                    for p in layer.parameters():
                        p.requires_grad = False

    def _add_stoch_depth(self, frozen_up_to: int, p: float):
        """Register forward hooks for stochastic depth (LayerDrop)."""
        # FIX (v8): use hooks instead of wrapper classes — fully
        # transparent to the Transformers encoder call loop.
        self._sd_handles = []
        _register_stoch_depth_hooks(
            self.backbone, frozen_up_to, p, self._sd_handles)
        print(f"  StochDepth hooks: {len(self._sd_handles)} layers "
              f"(p={p})")

    def forward(self, input_ids, pixel_values,
                attention_mask=None):
        # FIX (v8): use sub-model pooler_output directly.
        # pooler_output is always a plain tensor (B, hidden_size)
        # — the result of post_layernorm applied to the EOS/CLS token.
        # This avoids ALL version-dependent helper methods
        # (get_image_features / get_text_features behave differently
        # across transformers releases and may return dataclass objects
        # instead of tensors).
        vis_out = self.backbone.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        txt_out = self.backbone.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        img_e = vis_out.pooler_output   # (B, D)
        txt_e = txt_out.pooler_output   # (B, D)

        # L2-normalise to match SigLIP cosine-similarity space
        img_e = F.normalize(img_e, dim=-1)
        txt_e = F.normalize(txt_e, dim=-1)

        # Concatenate for gate + head input
        fused = torch.cat([img_e, txt_e], dim=-1)   # (B, 2*D)
        gates = self.gate(fused)                     # (B, 2)
        # Soft-weighted combination
        fused_w = (gates[:, 0:1] * img_e +
                   gates[:, 1:2] * txt_e)            # (B, D)
        # Final input: [weighted | original concat] for full capacity
        fused = torch.cat([fused_w, img_e - txt_e], dim=-1)  # (B,2*D)
        return self.head(fused)

    def get_param_groups(self, backbone_lr: float, head_lr: float,
                         weight_decay: float):
        """Separate LR for backbone vs head."""
        backbone_params = [p for n, p in self.backbone.named_parameters()
                           if p.requires_grad]
        head_params     = list(self.head.parameters())
        return [
            {"params": backbone_params,
             "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params,
             "lr": head_lr,     "weight_decay": weight_decay * 0.1},
        ]


# ══════════════════════════════════════════════════════════════
#  9.  PIXEL-ONLY MIXUP
#      FIX: mix pixel values only — token IDs are integers and
#      CANNOT be linearly interpolated
# ══════════════════════════════════════════════════════════════
def pixel_mixup(pix, y, alpha, device):
    if alpha <= 0:
        return pix, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(pix.size(0)).to(device)
    return (lam * pix + (1 - lam) * pix[idx],
            y, y[idx], lam)

def mixup_loss(crit, logits, ya, yb, lam):
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)


# ══════════════════════════════════════════════════════════════
#  10. R-DROP REGULARISATION
#      Run two forward passes with different dropout masks,
#      penalise the KL divergence between their distributions.
#      Prevents head from becoming overconfident.
# ══════════════════════════════════════════════════════════════
def rdrop_loss(logits1, logits2, alpha: float = 1.0):
    p = F.log_softmax(logits1, dim=-1)
    q = F.log_softmax(logits2, dim=-1)
    p_tgt = F.softmax(logits1, dim=-1)
    q_tgt = F.softmax(logits2, dim=-1)
    kl1   = F.kl_div(p, q_tgt, reduction="batchmean")
    kl2   = F.kl_div(q, p_tgt, reduction="batchmean")
    return alpha * (kl1 + kl2) / 2.0


# ══════════════════════════════════════════════════════════════
#  11. METRICS
# ══════════════════════════════════════════════════════════════
def compute_metrics(preds, labels):
    return (
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="binary", zero_division=0),
        precision_score(labels, preds, average="binary", zero_division=0),
        recall_score(labels, preds, average="binary", zero_division=0),
    )


# ══════════════════════════════════════════════════════════════
#  12. TRAIN EPOCH
# ══════════════════════════════════════════════════════════════
def train_epoch(model, loader, opt, sched, crit,
                scaler, device, cfg):
    model.train()
    tot_loss, all_preds, all_labels = 0.0, [], []
    opt.zero_grad()

    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(device)
        pix  = batch["pixel_values"].to(device)
        y    = batch["label"].to(device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)

        # Pixel-only Mixup
        pix_m, ya, yb, lam = pixel_mixup(
            pix, y, cfg["mixup_alpha"], device)

        with autocast(enabled=cfg["fp16"]):
            # First forward pass
            logits1 = model(ids, pix_m, attention_mask=mask)
            ce_loss = mixup_loss(crit, logits1, ya, yb, lam)

            # Second forward pass for R-Drop (different dropout)
            logits2  = model(ids, pix_m, attention_mask=mask)
            rd_loss  = rdrop_loss(logits1, logits2,
                                  cfg["rdrop_alpha"])
            loss     = (ce_loss + rd_loss) / cfg["grad_accum"]

        scaler.scale(loss).backward()

        if (step + 1) % cfg["grad_accum"] == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(
                model.parameters(), cfg["max_grad_norm"])
            scaler.step(opt); scaler.update()
            sched.step(); opt.zero_grad()

        tot_loss += loss.item() * cfg["grad_accum"]
        all_preds.extend(logits1.argmax(-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc, f1, pre, rec = compute_metrics(all_preds, all_labels)
    return tot_loss / len(loader), acc, f1, pre, rec


# ══════════════════════════════════════════════════════════════
#  13. EVAL EPOCH
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_epoch(model, loader, crit, device, fp16=True, tta=False):
    """Evaluate. When tta=True, average logits over 2 passes
    (original + horizontally-flipped pixels) for ~+1% accuracy."""
    model.eval()
    tot_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        pix  = batch["pixel_values"].to(device)
        y    = batch["label"].to(device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)

        with autocast(enabled=fp16):
            logits = model(ids, pix, attention_mask=mask)
            if tta:
                # second pass with H-flip
                pix_flip = torch.flip(pix, dims=[-1])
                logits2  = model(ids, pix_flip, attention_mask=mask)
                logits   = (logits + logits2) * 0.5
            loss = crit(logits, y)

        tot_loss += loss.item()
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc, f1, pre, rec = compute_metrics(all_preds, all_labels)
    return (tot_loss / len(loader), acc, f1, pre, rec,
            all_preds, all_labels)


# ══════════════════════════════════════════════════════════════
#  14. 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════
skf     = StratifiedKFold(n_splits=CFG["n_folds"],
                          shuffle=True, random_state=CFG["seed"])
y_arr   = df["label"].values
results = []
device  = CFG["device"]

print("=" * 60)
print("  5-Fold CV — SigLIP-2 × FAKEDDIT  (v9 accuracy-boost)")
print("=" * 60)

for fold, (tv_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(df)), y_arr), start=1):

    print(f"\n{'─' * 60}")
    print(f"  FOLD {fold} / {CFG['n_folds']}")
    print(f"{'─' * 60}")

    n_tv  = len(tv_idx)
    n_val = int(n_tv * CFG["val_ratio"] /
                (CFG["train_ratio"] + CFG["val_ratio"]))
    np.random.shuffle(tv_idx)
    val_idx   = tv_idx[:n_val]
    train_idx = tv_idx[n_val:]

    print(f"  Train={len(train_idx)}  Val={len(val_idx)}"
          f"  Test={len(test_idx)}")

    def make_dl(idx, aug, bs, shuf):
        ds = FakedditDataset(df.iloc[idx], processor, aug)
        return DataLoader(ds, batch_size=bs, shuffle=shuf,
                          num_workers=CFG["num_workers"],
                          pin_memory=True)

    train_dl = make_dl(train_idx, True,  CFG["batch_size"],   True)
    val_dl   = make_dl(val_idx,   False, CFG["batch_size"]*2, False)
    test_dl  = make_dl(test_idx,  False, CFG["batch_size"]*2, False)

    # ── Model ───────────────────────────────────────────────
    model = SigLIP2Classifier(
        CFG["model_name"], num_classes=2,
        dropout       = CFG["dropout"],
        freeze_layers = CFG["freeze_layers"],
        stoch_depth_p = CFG["stoch_depth_prob"],
        token         = HF_TOKEN,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    # ── Optimiser with separate LRs ─────────────────────────
    crit   = nn.CrossEntropyLoss(
                label_smoothing=CFG["label_smoothing"])
    opt    = torch.optim.AdamW(
                model.get_param_groups(
                    CFG["backbone_lr"],
                    CFG["head_lr"],
                    CFG["weight_decay"]))

    total_steps  = (len(train_dl) // CFG["grad_accum"]) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    sched  = get_cosine_schedule_with_warmup(
                opt, warmup_steps, total_steps)
    scaler = GradScaler(enabled=CFG["fp16"])

    # ── Training loop ───────────────────────────────────────
    best_val_acc  = 0.0
    patience_ctr  = 0
    ckpt          = f"/tmp/fold{fold}_best.pt"
    history       = []

    for epoch in range(1, CFG["epochs"] + 1):
        tr = train_epoch(model, train_dl, opt, sched,
                         crit, scaler, device, CFG)
        vl = eval_epoch(model, val_dl, crit,
                        device, CFG["fp16"])

        tr_loss, tr_acc, tr_f1 = tr[0], tr[1], tr[2]
        vl_loss, vl_acc, vl_f1 = vl[0], vl[1], vl[2]
        gap = tr_acc - vl_acc

        status = ("⚠ overfit" if gap > 0.10 else
                  "✔ good"   if gap < 0.05 else "")
        print(
            f"  Ep {epoch:02d} | "
            f"Tr  Loss={tr_loss:.4f} Acc={tr_acc:.4f} F1={tr_f1:.4f} | "
            f"Val Loss={vl_loss:.4f} Acc={vl_acc:.4f} F1={vl_f1:.4f} | "
            f"Gap={gap:+.4f} {status}"
        )
        history.append(dict(epoch=epoch,
                            tr_acc=tr_acc, vl_acc=vl_acc,
                            overfit=gap))

        # Save on best val accuracy (more robust than val loss
        # with label smoothing which inflates loss values)
        if vl_acc > best_val_acc:
            best_val_acc  = vl_acc
            patience_ctr  = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["early_stop_patience"]:
                print(f"  ⚠  Early stop triggered at epoch {epoch}")
                break

    # ── Test evaluation ─────────────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=device))
    (ts_loss, ts_acc, ts_f1, ts_pre,
     ts_rec, ts_preds, ts_labels) = eval_epoch(
        model, test_dl, crit, device, CFG["fp16"], tta=True)

    best_vl_acc = max(h["vl_acc"] for h in history)
    overfit     = best_vl_acc - ts_acc

    print(f"\n  ✔ Fold {fold} TEST —"
          f"  Acc={ts_acc:.4f}  F1={ts_f1:.4f}"
          f"  Prec={ts_pre:.4f}  Rec={ts_rec:.4f}"
          f"  Overfit={overfit:+.4f}")
    print(classification_report(
        ts_labels, ts_preds,
        target_names=["Real (0)", "Fake (1)"]))

    results.append(dict(
        fold=fold, accuracy=ts_acc, f1_score=ts_f1,
        precision=ts_pre, recall=ts_rec,
        val_acc=best_vl_acc, overfit=overfit))

    del model, opt, sched, scaler
    del train_dl, val_dl, test_dl
    gc.collect(); torch.cuda.empty_cache()
    if os.path.exists(ckpt):
        os.remove(ckpt)


# ══════════════════════════════════════════════════════════════
#  15. FINAL RESULTS TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FINAL RESULTS TABLE  —  SigLIP-2 × FAKEDDIT  v9")
print("=" * 60)

rdf  = pd.DataFrame(results)
disp = rdf.copy()
disp["fold"]      = disp["fold"].map(lambda x: f"Fold {x}")
disp["accuracy"]  = disp["accuracy"].map(lambda x: f"{x:.4f}")
disp["f1_score"]  = disp["f1_score"].map(lambda x: f"{x:.4f}")
disp["precision"] = disp["precision"].map(lambda x: f"{x:.4f}")
disp["recall"]    = disp["recall"].map(lambda x: f"{x:.4f}")
disp["val_acc"]   = disp["val_acc"].map(lambda x: f"{x:.4f}")
disp["overfit"]   = disp["overfit"].map(lambda x: f"{x:+.4f}")
disp.columns      = ["Fold", "Accuracy", "F1-Score",
                      "Precision", "Recall",
                      "Best Val Acc", "Overfit Gap"]
print(disp.to_string(index=False))

print("\n  ── Mean ± Std across folds ──")
for col in ["accuracy", "f1_score", "precision", "recall"]:
    m, s = rdf[col].mean(), rdf[col].std()
    flag = "✔" if m >= 0.90 else "…"
    print(f"  {col:<12}: {m:.4f} ± {s:.4f}  {flag}")

ov = rdf["overfit"].mean()
print(f"  {'overfit':<12}: {ov:+.4f}  "
      f"({'✔ low' if abs(ov) < 0.05 else '⚠ still high'})")

out_csv = "siglip2_fakeddit_v9_results.csv"
disp.to_csv(out_csv, index=False)
print(f"\n  Saved → {out_csv}")
print("=" * 60)
