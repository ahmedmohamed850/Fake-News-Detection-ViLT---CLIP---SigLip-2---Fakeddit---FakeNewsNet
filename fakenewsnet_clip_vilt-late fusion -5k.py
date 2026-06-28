# ============================================================
# Late Fusion: CLIP + ViLT for FakeNewsNet 2-Way Classification
# Kaggle Notebook — 5-Fold CV | 80/10/10 Split | 10 Epochs
# Datasets: GossipCop + PolitiFact (fake / real)
# ============================================================

# ── Cell 1: Install dependencies ──────────────────────────
# !pip install transformers datasets Pillow scikit-learn pandas numpy torch torchvision tqdm

# ── Cell 2: Imports ───────────────────────────────────────
import os, random, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    CLIPProcessor, CLIPModel,
    ViltProcessor, ViltForImageAndTextRetrieval,
    ViltModel,
    get_linear_schedule_with_warmup,
)
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Cell 3: Reproducibility ───────────────────────────────
SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES    = 2             # 2-way: 0 = real, 1 = fake
NUM_EPOCHS     = 10
BATCH_SIZE     = 32            # larger batch — backbones frozen
LR             = 2e-4          # higher LR for head-only training
N_FOLDS        = 5
MAX_SAMPLES    = 5000          # max per class (balanced)
# FakeNewsNet labels are derived from folder names (fake/real) → label_enc
# Sources: gossipcop + politifact
FREEZE_BACKBONES = True        # freeze CLIP & ViLT — only train projection + classifier
USE_AMP        = True          # mixed-precision for speed

print(f"Device: {DEVICE}")

# ── Cell 4: Load FakeNewsNet ──────────────────────────────
# FakeNewsNet structure on Kaggle:
#   /kaggle/input/<dataset>/gossipcop/fake/<article_id>/news content.json
#                                          + images/<img>.jpg
#                            real/<article_id>/news content.json
#                                              + images/<img>.jpg
#   /kaggle/input/<dataset>/politifact/fake/...
#                            real/...
#
# Each "news content.json" has keys: "title", "text", "images" (list of URLs)
# We use the title (+ first 128 chars of text) as the text input and
# the first available local image under images/ as the image input.
# If no image exists we fall back to a grey placeholder.

import glob as _glob
import json as _json
from tqdm import tqdm as _tqdm

# ── 4a. Locate dataset root ───────────────────────────────
_roots = []
for _pattern in [
    "/kaggle/input/**/gossipcop",
    "/kaggle/input/**/politifact",
]:
    _roots += _glob.glob(_pattern, recursive=True)

# Deduplicate parent dirs
_dataset_roots = list({os.path.dirname(p) for p in _roots})
if not _dataset_roots:
    raise FileNotFoundError(
        "FakeNewsNet directory not found under /kaggle/input/.\n"
        "Add the dataset: Notebook → Add data → search 'FakeNewsNet'.\n"
        "Expected structure: .../gossipcop/fake/<id>/news content.json\n"
        "                    .../gossipcop/real/<id>/news content.json"
    )
DATASET_ROOT = sorted(_dataset_roots)[0]
print(f"Dataset root: {DATASET_ROOT}")

# ── 4b. Crawl articles ────────────────────────────────────
_SOURCES  = ["gossipcop", "politifact"]
_LABELS   = {"fake": 1, "real": 0}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

def _find_first_image(article_dir):
    """Return path to the first valid image under <article_dir>/images/, or None."""
    img_folder = os.path.join(article_dir, "images")
    if not os.path.isdir(img_folder):
        return None
    for fname in sorted(os.listdir(img_folder)):
        if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
            fpath = os.path.join(img_folder, fname)
            if os.path.getsize(fpath) > 500:
                return fpath
    return None

def _load_article(article_dir):
    """Load title + text from 'news content.json'. Returns (text, img_path)."""
    json_path = os.path.join(article_dir, "news content.json")
    if not os.path.isfile(json_path):
        # Some datasets use a slightly different name
        candidates = _glob.glob(os.path.join(article_dir, "*.json"))
        if not candidates:
            return None, None
        json_path = candidates[0]
    try:
        with open(json_path, "r", encoding="utf-8", errors="ignore") as fh:
            data = _json.load(fh)
        title = str(data.get("title", "")).strip()
        body  = str(data.get("text", "")).strip()[:256]   # first 256 chars
        text  = (title + " " + body).strip() if body else title
        if not text:
            return None, None
        img_path = _find_first_image(article_dir)
        return text, img_path
    except Exception:
        return None, None

records = []
print("Crawling FakeNewsNet articles…")
for source in _SOURCES:
    src_dir = os.path.join(DATASET_ROOT, source)
    if not os.path.isdir(src_dir):
        print(f"  ⚠  {source} not found, skipping.")
        continue
    for split, label_int in _LABELS.items():
        split_dir = os.path.join(src_dir, split)
        if not os.path.isdir(split_dir):
            continue
        article_ids = os.listdir(split_dir)
        for art_id in _tqdm(article_ids, desc=f"{source}/{split}", leave=False):
            art_dir = os.path.join(split_dir, art_id)
            if not os.path.isdir(art_dir):
                continue
            text, img_path = _load_article(art_dir)
            if text is None:
                continue
            records.append({
                "id":        f"{source}_{split}_{art_id}",
                "source":    source,
                "text":      text,
                "img_path":  img_path,      # None if no image found
                "label_enc": label_int,
            })

if not records:
    raise RuntimeError(
        "No articles loaded from FakeNewsNet. Check the dataset structure.\n"
        "Expected: .../gossipcop/fake/<id>/news content.json"
    )

df = pd.DataFrame(records)
print(f"\nLoaded {len(df)} articles  |  "
      f"with image: {df['img_path'].notna().sum()}  |  "
      f"label dist: {df['label_enc'].value_counts().to_dict()}")

# ── 4c. Balanced sample ───────────────────────────────────
_per_class = min(df["label_enc"].value_counts().min(), MAX_SAMPLES // 2)
df = pd.concat([
    df[df["label_enc"] == c].sample(_per_class, random_state=SEED)
    for c in [0, 1]
]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Final dataset: {len(df)} rows  "
      f"| class dist: {df['label_enc'].value_counts().to_dict()}")

# ── Cell 5: Dataset ───────────────────────────────────────
_GREY_IMAGE = Image.new("RGB", (224, 224), (128, 128, 128))   # shared placeholder

class FakeNewsNetDataset(Dataset):
    """
    Loads FakeNewsNet articles.
    - text  : article title (+ first 256 chars of body)
    - image : first image found in <article>/images/, else grey placeholder
    - label : 0 = real, 1 = fake
    """
    def __init__(self, df, clip_processor, vilt_processor):
        self.df             = df.reset_index(drop=True)
        self.clip_processor = clip_processor
        self.vilt_processor = vilt_processor

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = str(row["text"])[:512]      # cap length for tokeniser
        label = int(row["label_enc"])

        # Load image (fall back to grey if missing / corrupt)
        img_path = row.get("img_path")
        try:
            if img_path and os.path.isfile(str(img_path)):
                image = Image.open(img_path).convert("RGB").resize((224, 224))
            else:
                image = _GREY_IMAGE.copy()
        except Exception:
            image = _GREY_IMAGE.copy()

        # CLIP inputs  (max_length=77 is CLIP's native limit)
        clip_in = self.clip_processor(
            text=[text], images=image,
            return_tensors="pt", padding="max_length", truncation=True, max_length=77
        )
        clip_in = {k: v.squeeze(0) for k, v in clip_in.items()}

        # ViLT inputs  (keep at 40 to stay within ViLT's 40-token limit)
        vilt_in = self.vilt_processor(
            text=[text], images=image,
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=40
        )
        vilt_in = {k: v.squeeze(0) for k, v in vilt_in.items()}

        return clip_in, vilt_in, label


def collate_fn(batch):
    clip_batch, vilt_batch, labels = zip(*batch)

    def stack_dict(lst):
        keys = lst[0].keys()
        return {k: torch.stack([d[k] for d in lst]) for k in keys}

    return stack_dict(clip_batch), stack_dict(vilt_batch), torch.tensor(labels)


# ── Cell 6: Late Fusion Model ─────────────────────────────
class CLIPFeatureExtractor(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        d = clip_model.config.projection_dim          # typically 512
        self.fc = nn.Sequential(
            nn.Linear(d * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, clip_inputs):
        text_out  = self.clip.text_model(
            input_ids=clip_inputs["input_ids"],
            attention_mask=clip_inputs["attention_mask"],
        )
        vision_out = self.clip.vision_model(
            pixel_values=clip_inputs["pixel_values"],
        )
        text_emb  = self.clip.text_projection(text_out.pooler_output)
        image_emb = self.clip.visual_projection(vision_out.pooler_output)
        text_emb  = F.normalize(text_emb, dim=-1)
        image_emb = F.normalize(image_emb, dim=-1)
        return self.fc(torch.cat([text_emb, image_emb], dim=-1))   # [B, 512]


class ViLTFeatureExtractor(nn.Module):
    def __init__(self, vilt_model):
        super().__init__()
        self.vilt = vilt_model
        d = vilt_model.config.hidden_size              # typically 768
        self.fc = nn.Sequential(
            nn.Linear(d, 512),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, vilt_inputs):
        out = self.vilt(
            input_ids=vilt_inputs["input_ids"],
            attention_mask=vilt_inputs["attention_mask"],
            token_type_ids=vilt_inputs.get("token_type_ids"),
            pixel_values=vilt_inputs["pixel_values"],
            pixel_mask=vilt_inputs.get("pixel_mask"),
        )
        cls_token = out.last_hidden_state[:, 0, :]     # [B, 768]
        return self.fc(cls_token)                       # [B, 512]


class LateFusionClassifier(nn.Module):
    def __init__(self, clip_model, vilt_model, num_classes=2, freeze_backbones=True):
        super().__init__()
        self.clip_fe = CLIPFeatureExtractor(clip_model)
        self.vilt_fe = ViLTFeatureExtractor(vilt_model)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        if freeze_backbones:
            for p in self.clip_fe.clip.parameters():
                p.requires_grad = False
            for p in self.vilt_fe.vilt.parameters():
                p.requires_grad = False

    def forward(self, clip_inputs, vilt_inputs):
        clip_feat = self.clip_fe(clip_inputs)   # [B, 512]
        vilt_feat = self.vilt_fe(vilt_inputs)   # [B, 512]
        fused     = torch.cat([clip_feat, vilt_feat], dim=-1)  # [B, 1024]
        return self.classifier(fused)           # [B, num_classes]


# ── Cell 7: Training & Evaluation helpers ─────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for clip_in, vilt_in, labels in loader:
        clip_in  = {k: v.to(DEVICE) for k, v in clip_in.items()}
        vilt_in  = {k: v.to(DEVICE) for k, v in vilt_in.items()}
        labels   = labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(clip_in, vilt_in)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss  += loss.item() * labels.size(0)
        preds_all   += logits.argmax(dim=1).cpu().tolist()
        labels_all  += labels.cpu().tolist()

    n    = len(labels_all)
    acc  = accuracy_score(labels_all, preds_all)
    return total_loss / n, acc


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []
    for clip_in, vilt_in, labels in loader:
        clip_in  = {k: v.to(DEVICE) for k, v in clip_in.items()}
        vilt_in  = {k: v.to(DEVICE) for k, v in vilt_in.items()}
        labels   = labels.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(clip_in, vilt_in)
            loss   = criterion(logits, labels)

        total_loss  += loss.item() * labels.size(0)
        preds_all   += logits.argmax(dim=1).cpu().tolist()
        labels_all  += labels.cpu().tolist()

    n   = len(labels_all)
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="binary", zero_division=0)
    pr  = precision_score(labels_all, preds_all, average="binary", zero_division=0)
    rc  = recall_score(labels_all, preds_all, average="binary", zero_division=0)
    return total_loss / n, acc, f1, pr, rc, preds_all, labels_all


def compute_overfitting_gap(train_acc, val_acc):
    """Simple gap metric — positive means train > val (overfitting signal)."""
    return round(train_acc - val_acc, 4)


# ── Cell 8: 5-Fold Cross-Validation ───────────────────────
print("\n📥 Loading processors…")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
labels_arr  = df["label_enc"].values
fold_results = []

CLASS_NAMES = ["real", "fake"]   # 2-way

print("=" * 70)
print(f"  Late Fusion CLIP + ViLT  |  FakeNewsNet 2-Way  |  {N_FOLDS}-Fold CV")
print("=" * 70)

for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, labels_arr)):
    print(f"\n{'─'*70}")
    print(f"  FOLD {fold+1}/{N_FOLDS}")
    print(f"{'─'*70}")

    # --- 80 / 10 / 10 split within this fold ---
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df      = df.iloc[test_idx].reset_index(drop=True)

    # Split train_val → 80% train, 10% val  (relative to full fold)
    val_size   = int(0.1 * len(df))
    train_size = len(train_val_df) - val_size
    perm       = np.random.permutation(len(train_val_df))
    train_idx2 = perm[:train_size]
    val_idx2   = perm[train_size:train_size + val_size]

    train_df = train_val_df.iloc[train_idx2].reset_index(drop=True)
    val_df   = train_val_df.iloc[val_idx2].reset_index(drop=True)

    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # --- Datasets / Loaders ---
    full_ds    = FakeNewsNetDataset(df, clip_processor, vilt_processor)
    train_loader = DataLoader(
        Subset(full_ds, train_val_idx[:train_size]),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_ds, train_val_idx[train_size:train_size+val_size]),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(full_ds, test_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # --- Build fresh model for each fold ---
    clip_model_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    vilt_model_base = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
    model = LateFusionClassifier(
        clip_model_base, vilt_model_base,
        num_classes=NUM_CLASSES,
        freeze_backbones=FREEZE_BACKBONES
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_p/1e6:.1f}M total | {trainable/1e6:.1f}M trainable")

    # Class-weighted loss to combat imbalance
    class_counts = np.bincount(df.iloc[train_val_idx[:train_size]]["label_enc"].values,
                                minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE),
        label_smoothing=0.1
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    optimizer   = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-2
    )
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_acc = 0.0
    best_state   = None
    epoch_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc                              = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
        vl_loss, vl_acc, vl_f1, vl_pr, vl_rc, _, _ = eval_epoch(model, val_loader, criterion)
        gap = compute_overfitting_gap(tr_acc, vl_acc)

        epoch_history.append({
            "epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
            "vl_loss": vl_loss, "vl_acc": vl_acc, "overfit_gap": gap
        })

        print(f"  Ep {epoch:02d}/{NUM_EPOCHS}  "
              f"TR-Loss={tr_loss:.4f}  TR-Acc={tr_acc:.4f}  |  "
              f"VL-Loss={vl_loss:.4f}  VL-Acc={vl_acc:.4f}  "
              f"F1={vl_f1:.4f}  Overfit-Gap={gap:+.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- Evaluate best model on test set ---
    model.load_state_dict(best_state)
    model.to(DEVICE)
    _, ts_acc, ts_f1, ts_pr, ts_rc, ts_preds, ts_labels = eval_epoch(model, test_loader, criterion)
    final_gap = compute_overfitting_gap(
        epoch_history[-1]["tr_acc"], epoch_history[-1]["vl_acc"]
    )

    print(f"\n  ✅ Fold {fold+1} TEST  Acc={ts_acc:.4f}  F1={ts_f1:.4f}  "
          f"Prec={ts_pr:.4f}  Rec={ts_rc:.4f}  Overfit-Gap={final_gap:+.4f}")
    print("\n  Classification Report:\n",
          classification_report(ts_labels, ts_preds,
                                target_names=CLASS_NAMES,
                                zero_division=0))

    fold_results.append({
        "Fold":          fold + 1,
        "Test Acc":      round(ts_acc, 4),
        "Test F1":       round(ts_f1, 4),
        "Precision":     round(ts_pr, 4),
        "Recall":        round(ts_rc, 4),
        "Best Val Acc":  round(best_val_acc, 4),
        "Overfit Gap":   round(final_gap, 4),
    })

    del model, clip_model_base, vilt_model_base
    torch.cuda.empty_cache()


# ── Cell 9: Summary Table ─────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL 5-FOLD RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(fold_results)

# Append mean & std rows
mean_row = results_df.drop(columns=["Fold"]).mean().round(4).to_dict()
std_row  = results_df.drop(columns=["Fold"]).std().round(4).to_dict()
mean_row["Fold"] = "Mean"
std_row["Fold"]  = "Std"

summary_df = pd.concat(
    [results_df, pd.DataFrame([mean_row, std_row])],
    ignore_index=True
)

print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv("/kaggle/working/fold_results.csv", index=False)
print("\n📄 Results saved to /kaggle/working/fold_results.csv")

# Pretty display (Kaggle notebook)
try:
    from IPython.display import display
    display(summary_df.style
        .set_caption("Late Fusion CLIP+ViLT — FakeNewsNet 5-Fold 2-Way Classification Results")
        .background_gradient(subset=["Test Acc","Test F1","Precision","Recall"], cmap="YlGn")
        .applymap(lambda v: "color:red" if isinstance(v, float) and v > 0.05
                  else ("color:green" if isinstance(v, float) and v <= 0.02 else ""),
                  subset=["Overfit Gap"])
        .format(precision=4)
    )
except Exception:
    pass

mean_acc = results_df["Test Acc"].mean()
mean_f1  = results_df["Test F1"].mean()
print(f"\n🏆  Average Test Accuracy : {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"🏆  Average Test F1-Score : {mean_f1:.4f} ({mean_f1*100:.2f}%)")