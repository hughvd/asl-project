"""Build the MS4 main notebook (cs1090b_ms4_main.ipynb).

The output is the single deliverable notebook: training code, evaluation,
external-dataset generalization, and the AC209B interpretability suite all
self-contained so it can be run on Colab.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path('cs1090b_ms4_main.ipynb')

cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# =====================================================================
# 0. TITLE + TOC
# =====================================================================
md("""# CS1090B · MS4 — ASL Sign Recognition: Final Modeling & Interpretability

**Canvas project:** #11
**Group members:** Zachary Donnini, Ayush Gupta, Hugh Van Deventer

This notebook is the main MS4 deliverable. It trains stronger supervised
baselines (ResNet-50, ViT-B/16, DINOv2-ViT-S/14), evaluates them on held-out
signers (P9/P10) and an external Kaggle dataset, and runs an interpretability
suite — linear probing, TCAV, Grad-CAM, and counterfactual perturbation —
to characterize *what* the models actually learned about hand shape vs.
signer identity.

The interpretability suite is the AC209B "method not covered in class"
component.

## Table of Contents

1. **Setup** — environment, paths, dependencies
2. **Data**
   - 2.1 Primary dataset (subject-independent split P1-P10)
   - 2.2 External generalization dataset (Kaggle ASL hand-landmark + gesture)
3. **Models** — ResNet-18 / ResNet-50 / ViT-B/16 / DINOv2-ViT-S/14
4. **Training** — config-driven loop, AMP, cosine LR, label smoothing, RandAug
5. **Evaluation** — metrics on held-out signers, per-signer breakdown,
   external-dataset transfer, robustness sweep, test-time augmentation
6. **Interpretability** *(AC209B novel-method component)*
   - 6.1 Layer-by-layer linear probing — *where does the model factor out
     signer identity vs crystallize sign identity?*
   - 6.2 TCAV (concept activation vectors) — *does the model represent
     hand-shape concepts internally?*
   - 6.3 Saliency / Grad-CAM / Integrated Gradients
   - 6.4 Counterfactual perturbation on the W→4/6 case
   - 6.5 ViT attention head specialization
7. **Final summary, references, AI-assistance disclosure**
""")

# =====================================================================
# 1. SETUP
# =====================================================================
md("""## 1. Setup

The cell below installs `timm`, `captum`, and `kagglehub`. Skip it if
the environment already has them.""")

code("""%pip install -q timm captum kagglehub seaborn
""")

code("""import os, sys, json, time, math, copy, random, itertools, zipfile, shutil
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

import timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, top_k_accuracy_score,
                             confusion_matrix, classification_report)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

IS_COLAB = 'google.colab' in sys.modules
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cpu')
)
print(f'IS_COLAB={IS_COLAB}  DEVICE={DEVICE}  torch={torch.__version__}')

plt.rcParams['figure.dpi'] = 110
sns.set_context('notebook')
""")

md("""**Project root.** If running on Colab with a Drive mount, set
`PROJECT_ROOT` to the Drive folder containing `data/asl_dataset/`. Otherwise
the notebook assumes the working directory is the project root and `data/`
sits next to it.""")

code("""# --- Edit this if your data lives elsewhere ---
if IS_COLAB:
    # Option A: data already in /content (uploaded directly to the Colab VM)
    # Option B: mount Google Drive and point at a folder there
    try:
        from google.colab import drive
        if not Path('/content/drive').exists():
            drive.mount('/content/drive')
        candidate = Path('/content/drive/MyDrive/asl_project')
        PROJECT_ROOT = candidate if candidate.exists() else Path('/content')
    except Exception:
        PROJECT_ROOT = Path('/content')
else:
    PROJECT_ROOT = Path.cwd()

DATA_DIR     = PROJECT_ROOT / 'data'
RAW_DIR      = DATA_DIR / 'asl_dataset'
RUNS_DIR     = PROJECT_ROOT / 'runs'
FIGS_DIR     = PROJECT_ROOT / 'figs_ms4'
EXTERNAL_DIR = DATA_DIR / 'external'

for p in (RUNS_DIR, FIGS_DIR, EXTERNAL_DIR):
    p.mkdir(parents=True, exist_ok=True)

print('PROJECT_ROOT:', PROJECT_ROOT)
print('Has raw images:', RAW_DIR.exists())
""")

# =====================================================================
# 2. DATA
# =====================================================================
md("""## 2. Data

### 2.1 Primary dataset — subject-independent split

10 signers (P1-P10), 36 classes, 100 images per (signer, class). MS3 EDA
revealed that the dataset's default 80/20 split mixes signers across both
halves; we instead build a **subject-independent split**: P1-P6 → train,
P7-P8 → val, P9-P10 → test. Every metric in this notebook is reported on
P9/P10 — signers the model has **never seen** during training.""")

code("""TRAIN_SUBJECTS = {f'P{i}' for i in range(1, 7)}
VAL_SUBJECTS   = {'P7', 'P8'}
TEST_SUBJECTS  = {'P9', 'P10'}

def build_splits(raw_dir: Path = RAW_DIR,
                 out_csv: Path = DATA_DIR / 'splits.csv',
                 force: bool = False) -> pd.DataFrame:
    \"\"\"Construct (or load) the subject-independent split CSV.\"\"\"
    if out_csv.exists() and not force:
        return pd.read_csv(out_csv)
    classes = sorted(d for d in os.listdir(raw_dir) if (raw_dir / d).is_dir())
    rows = []
    for cls in classes:
        for f in os.listdir(raw_dir / cls):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            subj = f.split('_')[0]
            split = ('train' if subj in TRAIN_SUBJECTS else
                     'val'   if subj in VAL_SUBJECTS   else
                     'test'  if subj in TEST_SUBJECTS  else None)
            if split is None: continue
            rows.append({'path': str(raw_dir / cls / f), 'class': cls,
                         'subject': subj, 'split': split})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

splits_df = build_splits()
CLASSES = sorted(splits_df['class'].unique().tolist())
N_CLASSES = len(CLASSES)
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
print(splits_df.groupby('split').size())
print('Classes:', N_CLASSES)
""")

code("""IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class AslDataset(Dataset):
    \"\"\"Loads ASL images from a (path, class)-row dataframe.\"\"\"
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row['path']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[row['class']]

def make_transforms(image_size: int, training: bool, strength: str = 'strong'):
    \"\"\"Build a torchvision transform pipeline.

    strength: 'strong' (RandAugment + RandomErasing + jitter), 'mild' (jitter
    + small rotation), 'none' (just resize+normalize).
    \"\"\"
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not training or strength == 'none':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), norm,
        ])
    if strength == 'mild':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), norm,
        ])
    # strong
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        transforms.RandomCrop(image_size),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.25),
    ])

def make_loaders(image_size: int, batch: int = 64,
                 strength: str = 'strong', num_workers: int = 2):
    train_tf = make_transforms(image_size, training=True,  strength=strength)
    eval_tf  = make_transforms(image_size, training=False)
    tr_ds = AslDataset(splits_df[splits_df['split'] == 'train'], class_to_idx, train_tf)
    va_ds = AslDataset(splits_df[splits_df['split'] == 'val'],   class_to_idx, eval_tf)
    te_ds = AslDataset(splits_df[splits_df['split'] == 'test'],  class_to_idx, eval_tf)
    tr = DataLoader(tr_ds, batch_size=batch, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(va_ds, batch_size=batch, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    te = DataLoader(te_ds, batch_size=batch, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tr, va, te
""")

md("""### 2.2 External generalization dataset

Source: [Kaggle — ASL Hand Landmark and Gesture Dataset (avinashkr090502)](https://www.kaggle.com/datasets/iamavinashkr090502/asl-hand-landmark-and-gesture-dataset).

This dataset was collected by a *different team* under different lighting,
backgrounds, and signers. Evaluating on it is a stronger generalization
test than P9/P10 alone — it stresses the model on covariate shifts our
training distribution never saw. We map its class names onto our 36-way
taxonomy (case-insensitive, with simple normalization).""")

code("""def download_kaggle_external(force: bool = False) -> Path:
    \"\"\"Download the external ASL dataset via kagglehub. Returns its root path.\"\"\"
    try:
        import kagglehub
    except ImportError:
        os.system('pip install -q kagglehub')
        import kagglehub
    target = EXTERNAL_DIR / 'kaggle_avinashkr'
    marker = target / '.downloaded'
    if marker.exists() and not force:
        return target
    src = Path(kagglehub.dataset_download(
        'iamavinashkr090502/asl-hand-landmark-and-gesture-dataset'))
    target.mkdir(parents=True, exist_ok=True)
    # Copy (or symlink) once so we have it under our project tree
    for item in src.iterdir():
        dest = target / item.name
        if dest.exists(): continue
        try:
            os.symlink(item, dest)
        except (OSError, NotImplementedError):
            shutil.copytree(item, dest) if item.is_dir() else shutil.copy(item, dest)
    marker.touch()
    return target


def normalize_class_name(name: str) -> Optional[str]:
    \"\"\"Map an external class name onto our 36-way taxonomy.\"\"\"
    n = name.strip().upper()
    # Strip common decorations
    for ch in ['_', '-', ' ']:
        n = n.replace(ch, '')
    if n in {c.upper() for c in CLASSES}:
        return next(c for c in CLASSES if c.upper() == n)
    # Single-character match (e.g. 'A0001.jpg' folder name 'A')
    if len(n) >= 1 and n[0] in {c.upper() for c in CLASSES}:
        return next(c for c in CLASSES if c.upper() == n[0])
    return None


def build_external_index(root: Path) -> pd.DataFrame:
    \"\"\"Walk *root* and build a (path, class) dataframe for any folder that
    matches one of our 36 classes. Subject is filled in opportunistically
    from path components.\"\"\"
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        cls_guess = normalize_class_name(Path(dirpath).name)
        if cls_guess is None:
            continue
        # parent of dirpath might encode subject or split — keep it for tracing
        subject = Path(dirpath).parent.name
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                rows.append({
                    'path': str(Path(dirpath) / f),
                    'class': cls_guess,
                    'subject': f'EXT_{subject}',
                    'split': 'external',
                })
    if not rows:
        print('WARNING: no images mapped — inspect the dataset layout manually.')
    return pd.DataFrame(rows)


def load_external() -> pd.DataFrame:
    root = download_kaggle_external()
    df = build_external_index(root)
    if len(df):
        print(f'External dataset: {len(df):,} images across '
              f'{df[\"class\"].nunique()} classes from {df[\"subject\"].nunique()} subjects')
        print(df.groupby('class').size().describe())
    return df

# Run only if you want to download now (it can be skipped during code review).
# external_df = load_external()
""")

# =====================================================================
# 3. MODELS
# =====================================================================
md("""## 3. Models

We compare four architectures on the same subject-independent split:

| Tag | Backbone | Pretraining | Why include |
|---|---|---|---|
| `resnet18`  | ResNet-18  | ImageNet-1k (supervised) | MS3 baseline, kept for reference |
| `resnet50`  | ResNet-50  | ImageNet-1k (supervised) | larger CNN, well-tuned recipe |
| `vit_b16`   | ViT-B/16   | ImageNet-21k → 1k (supervised) | strong supervised transformer |
| `dinov2`    | ViT-S/14   | DINOv2 (self-supervised, 142M imgs) | self-supervised features test |

DINOv2 is the most interesting comparison: its features were learned without
any class labels, so its performance probes whether self-supervised
representations transfer better across signers than supervised ones.""")

code("""@dataclass
class TrainConfig:
    name: str                    # unique run id, used as folder name under runs/
    arch: str                    # 'resnet18' | 'resnet50' | 'vit_b16' | 'dinov2'
    image_size: int = 224
    batch_size: int = 64
    epochs:     int = 20
    lr:         float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    label_smoothing: float = 0.05
    dropout:        float = 0.0
    augment:        str   = 'strong'   # 'strong' | 'mild' | 'none'
    use_amp:        bool  = True
    optimizer:      str   = 'adamw'    # 'adamw' | 'sgd'
    seed:           int   = 42
    # DINOv2 / ViT specific
    finetune_blocks: Optional[int] = None  # if set, freeze all but last K blocks


def build_model(cfg: TrainConfig, n_classes: int = N_CLASSES) -> nn.Module:
    \"\"\"Instantiate the backbone with a fresh head sized to n_classes.\"\"\"
    if cfg.arch == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n_classes) if cfg.dropout == 0 else \\
               nn.Sequential(nn.Dropout(cfg.dropout),
                             nn.Linear(m.fc.in_features, n_classes))
    elif cfg.arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_classes) if cfg.dropout == 0 else \\
               nn.Sequential(nn.Dropout(cfg.dropout),
                             nn.Linear(m.fc.in_features, n_classes))
    elif cfg.arch == 'vit_b16':
        m = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k',
                              pretrained=True, num_classes=n_classes,
                              drop_rate=cfg.dropout)
    elif cfg.arch == 'dinov2':
        m = timm.create_model('vit_small_patch14_dinov2.lvd142m',
                              pretrained=True, num_classes=n_classes,
                              img_size=cfg.image_size)
        if cfg.finetune_blocks is not None:
            # Freeze all but the last K blocks + head
            for p in m.parameters(): p.requires_grad = False
            for blk in m.blocks[-cfg.finetune_blocks:]:
                for p in blk.parameters(): p.requires_grad = True
            for p in m.head.parameters(): p.requires_grad = True
            for p in m.norm.parameters(): p.requires_grad = True
    else:
        raise ValueError(f'Unknown arch: {cfg.arch}')
    return m
""")

# =====================================================================
# 4. TRAINING
# =====================================================================
md("""## 4. Training

A single config-driven training loop is reused for every model. It supports
mixed-precision (AMP), linear-warmup → cosine-annealing learning-rate
schedule, label smoothing, and the strong-augmentation pipeline defined
above. Per-epoch metrics are saved to `runs/<name>/history.csv` and the
best-validation checkpoint to `runs/<name>/ckpt_best.pt`.""")

code("""def cosine_lr(step: int, total: int, warmup: int) -> float:
    \"\"\"Linear warmup → cosine decay multiplier in [0, 1].\"\"\"
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one(cfg: TrainConfig,
              num_workers: int = 2,
              save_every: bool = False) -> Path:
    \"\"\"Train a single config end-to-end. Returns the run directory.\"\"\"
    out_dir = RUNS_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train_loader, val_loader, _ = make_loaders(cfg.image_size, cfg.batch_size,
                                               cfg.augment, num_workers)
    model = build_model(cfg).to(DEVICE)

    if cfg.optimizer == 'adamw':
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay,
                              momentum=0.9, nesterov=True)
    else:
        raise ValueError(cfg.optimizer)

    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    use_amp = cfg.use_amp and DEVICE.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_lr(s, total_steps, warmup_steps))

    history, best_val = [], -1.0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        run_loss = run_corr = run_seen = 0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
            sched.step()
            run_loss += loss.item() * x.size(0)
            run_corr += (logits.argmax(1) == y).sum().item()
            run_seen += x.size(0)
        tr_loss, tr_acc = run_loss / run_seen, run_corr / run_seen

        # validation
        model.eval()
        v_corr = v_seen = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                v_corr += (logits.argmax(1) == y).sum().item()
                v_seen += x.size(0)
        val_acc = v_corr / v_seen
        dt = time.time() - t0

        history.append({'epoch': ep, 'train_loss': tr_loss, 'train_acc': tr_acc,
                        'val_acc': val_acc, 'lr': sched.get_last_lr()[0], 'time_s': dt})
        print(f'[{cfg.name}] ep {ep:2d}/{cfg.epochs}  loss {tr_loss:.3f}  '
              f'tr_acc {tr_acc:.3f}  val_acc {val_acc:.3f}  '
              f'lr {sched.get_last_lr()[0]:.2e}  ({dt:.1f}s)')

        ckpt = {'model': model.state_dict(), 'epoch': ep, 'val_acc': val_acc,
                'cfg': asdict(cfg)}
        if val_acc > best_val:
            best_val = val_acc
            torch.save(ckpt, out_dir / 'ckpt_best.pt')
        if save_every:
            torch.save(ckpt, out_dir / f'ckpt_ep{ep:02d}.pt')
        torch.save(ckpt, out_dir / 'ckpt_last.pt')
        pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)

    return out_dir
""")

md("""### 4.1 Run configurations

The configs below define the supervised + transformer + self-supervised
sweep. ResNet-50 is run with two augmentation strengths (strong vs mild)
to measure the augmentation contribution. ViT-B/16 uses a lower LR to
match the standard ViT recipe. DINOv2 is fine-tuned only on its last 4
blocks to limit overfitting on our 21.6k-image training set.

**Resource budget on a Colab L4 / A100 GPU:** roughly 5-6 minutes per
ResNet-18 epoch, 8-10 minutes per ResNet-50 epoch, 12-15 minutes per
ViT-B/16 epoch. Reduce `epochs` for a smoke-test run.""")

code("""CONFIGS: List[TrainConfig] = [
    # MS3 reference baseline at 96x96 — kept for direct comparison
    TrainConfig(name='resnet18_ms3ref', arch='resnet18', image_size=96,
                batch_size=128, epochs=8, lr=1e-3, augment='mild',
                warmup_epochs=1, label_smoothing=0.0),

    # ResNet-18 at 224 with strong augmentation (resolution-scaling control)
    TrainConfig(name='resnet18_224', arch='resnet18', image_size=224,
                batch_size=128, epochs=20, lr=1e-3, augment='strong'),

    # ResNet-50 — main supervised CNN
    TrainConfig(name='resnet50_strong', arch='resnet50', image_size=224,
                batch_size=64, epochs=25, lr=1e-3, weight_decay=5e-4,
                dropout=0.1, augment='strong'),
    TrainConfig(name='resnet50_mild',   arch='resnet50', image_size=224,
                batch_size=64, epochs=25, lr=1e-3, weight_decay=5e-4,
                augment='mild'),

    # ViT-B/16 — supervised transformer
    TrainConfig(name='vit_b16', arch='vit_b16', image_size=224,
                batch_size=48, epochs=20, lr=1e-4, weight_decay=5e-2,
                warmup_epochs=2, dropout=0.1, augment='strong'),

    # DINOv2 ViT-S/14 — self-supervised transformer, partial fine-tune
    TrainConfig(name='dinov2_partial', arch='dinov2', image_size=224,
                batch_size=48, epochs=15, lr=3e-4, weight_decay=5e-2,
                warmup_epochs=2, augment='strong', finetune_blocks=4),
]

# Convert configs to a printable summary
pd.DataFrame([asdict(c) for c in CONFIGS])
""")

md("""### 4.2 Train all configs

The cell below trains each config sequentially and skips any run that
already has a `ckpt_best.pt` on disk (so you can resume after a Colab
disconnect). Set `force_retrain=True` to overwrite.""")

code("""def run_all_configs(configs=CONFIGS, force_retrain=False, limit=None):
    completed = []
    for cfg in (configs[:limit] if limit else configs):
        ckpt = RUNS_DIR / cfg.name / 'ckpt_best.pt'
        if ckpt.exists() and not force_retrain:
            print(f'[SKIP] {cfg.name} already trained (ckpt at {ckpt})')
            completed.append(RUNS_DIR / cfg.name)
            continue
        print(f'[TRAIN] {cfg.name}')
        completed.append(train_one(cfg))
    return completed

# UNCOMMENT to actually train:
# run_dirs = run_all_configs()
""")

# =====================================================================
# 5. EVALUATION
# =====================================================================
md("""## 5. Evaluation

For every trained run we record three numbers on **held-out signers
P9 + P10**: top-1 accuracy, macro-F1, top-5 accuracy. We also compute a
per-signer breakdown (P9 vs P10 separately), evaluate on the external
Kaggle dataset, and run a synthetic-corruption robustness sweep.""")

code("""@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader,
                   n_classes: int = N_CLASSES) -> Dict:
    \"\"\"Run *model* on *loader* and return preds/probs + metrics.\"\"\"
    model.eval()
    all_preds, all_probs, all_true = [], [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x).cpu()
        probs  = F.softmax(logits, dim=1).numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(1))
        all_true.append(y.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    probs  = np.concatenate(all_probs)
    return {
        'y_true': y_true, 'y_pred': y_pred, 'probs': probs,
        'acc':      float((y_pred == y_true).mean()),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'top5':     float(top_k_accuracy_score(y_true, probs, k=5,
                                               labels=list(range(n_classes)))),
    }


def load_run(run_dir: Path) -> Tuple[nn.Module, TrainConfig]:
    cfg_dict = json.loads((run_dir / 'config.json').read_text())
    cfg = TrainConfig(**cfg_dict)
    ckpt = torch.load(run_dir / 'ckpt_best.pt', map_location=DEVICE)
    model = build_model(cfg).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, cfg


def evaluate_run(run_dir: Path, save_predictions: bool = True) -> Dict:
    \"\"\"Evaluate a trained run on the held-out test set.\"\"\"
    model, cfg = load_run(run_dir)
    _, _, test_loader = make_loaders(cfg.image_size, cfg.batch_size,
                                     strength='none', num_workers=2)
    res = evaluate_model(model, test_loader)
    if save_predictions:
        pd.DataFrame({'y_true': res['y_true'], 'y_pred': res['y_pred']}) \\
          .to_csv(run_dir / 'predictions_test.csv', index=False)
        np.save(run_dir / 'probs_test.npy', res['probs'])
    return res
""")

md("""### 5.1 Master metrics table""")

code("""def collect_metrics(force_eval: bool = False) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        if not (run_dir / 'ckpt_best.pt').exists():
            continue
        cache = run_dir / 'metrics.json'
        if cache.exists() and not force_eval:
            m = json.loads(cache.read_text())
        else:
            res = evaluate_run(run_dir)
            m = {'name': run_dir.name, 'acc': res['acc'],
                 'macro_f1': res['macro_f1'], 'top5': res['top5']}
            cache.write_text(json.dumps(m, indent=2))
        rows.append(m)
    return pd.DataFrame(rows).sort_values('acc', ascending=False).reset_index(drop=True)

# metrics_df = collect_metrics(); metrics_df
""")

md("""### 5.2 Per-signer breakdown

The MS3 case study showed that some test signs collapse asymmetrically
across P9 vs P10 (W → 4 for P9, W → 6 for P10). We extend that analysis
to every model.""")

code("""def per_signer_breakdown(run_dir: Path) -> pd.DataFrame:
    \"\"\"Return a (signer × class) recall table for a trained run.\"\"\"
    test_df = splits_df[splits_df['split'] == 'test'].reset_index(drop=True)
    preds = pd.read_csv(run_dir / 'predictions_test.csv')
    df = test_df.assign(y_true=preds['y_true'], y_pred=preds['y_pred'])
    out = (df.assign(correct=df.y_pred == df.y_true)
             .groupby(['subject', 'class'])['correct'].mean()
             .unstack('class'))
    return out


def per_signer_overall(run_dir: Path) -> pd.Series:
    test_df = splits_df[splits_df['split'] == 'test'].reset_index(drop=True)
    preds = pd.read_csv(run_dir / 'predictions_test.csv')
    df = test_df.assign(y_true=preds['y_true'], y_pred=preds['y_pred'])
    return df.assign(correct=df.y_pred == df.y_true).groupby('subject')['correct'].mean()
""")

md("""### 5.3 External-dataset transfer (Kaggle)

Evaluate every trained model on the external Kaggle dataset. We map its
class labels onto our 36-way taxonomy, take only the overlapping classes,
and report accuracy + macro-F1. This is the strictest generalization
test: a different team, different signers, different lighting/background.""")

code("""def evaluate_external(run_dir: Path, external_df: pd.DataFrame,
                      batch_size: int = 64) -> Dict:
    if len(external_df) == 0:
        return {'acc': float('nan'), 'macro_f1': float('nan'), 'n': 0}
    model, cfg = load_run(run_dir)
    eval_tf = make_transforms(cfg.image_size, training=False)
    ds = AslDataset(external_df, class_to_idx, eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    res = evaluate_model(model, loader)
    res['n'] = len(external_df)
    return {k: v for k, v in res.items()
            if k in ('acc', 'macro_f1', 'top5', 'n')}


def evaluate_external_all() -> pd.DataFrame:
    external_df = load_external() if not (DATA_DIR / 'external_index.csv').exists() else \\
                  pd.read_csv(DATA_DIR / 'external_index.csv')
    if not (DATA_DIR / 'external_index.csv').exists():
        external_df.to_csv(DATA_DIR / 'external_index.csv', index=False)
    rows = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        if not (run_dir / 'ckpt_best.pt').exists(): continue
        m = evaluate_external(run_dir, external_df)
        m['name'] = run_dir.name
        rows.append(m)
    return pd.DataFrame(rows)
""")

md("""### 5.4 Robustness sweep

How fragile is each model under brightness drift, blur, and JPEG
compression? We synthesize three corruption families and report
top-1 accuracy as a function of corruption severity.""")

code("""class CorruptedDataset(Dataset):
    \"\"\"Wraps an image dataframe and applies a deterministic corruption to each image.\"\"\"
    def __init__(self, df, class_to_idx, image_size, corruption: str, severity: float):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.corruption = corruption
        self.severity   = severity
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self): return len(self.df)

    def _corrupt(self, img: Image.Image) -> Image.Image:
        if self.corruption == 'brightness':
            from PIL import ImageEnhance
            return ImageEnhance.Brightness(img).enhance(1.0 + self.severity)
        if self.corruption == 'blur':
            from PIL import ImageFilter
            return img.filter(ImageFilter.GaussianBlur(radius=self.severity))
        if self.corruption == 'jpeg':
            import io
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=int(self.severity))
            buf.seek(0)
            return Image.open(buf).convert('RGB')
        return img

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row['path']).convert('RGB')
        img = self._corrupt(img)
        img = img.resize((self.image_size, self.image_size))
        x = transforms.functional.to_tensor(img)
        x = self.norm(x)
        return x, self.class_to_idx[row['class']]


def robustness_sweep(run_dir: Path, severities: Dict[str, List[float]] = None,
                    batch: int = 64) -> pd.DataFrame:
    severities = severities or {
        'brightness': [-0.4, -0.2, 0.0, 0.2, 0.4],
        'blur':       [0.0, 1.0, 2.0, 3.0, 4.0],
        'jpeg':       [95, 60, 30, 15, 5],
    }
    model, cfg = load_run(run_dir)
    test_df = splits_df[splits_df['split'] == 'test']
    rows = []
    for kind, levels in severities.items():
        for s in levels:
            ds = CorruptedDataset(test_df, class_to_idx, cfg.image_size, kind, s)
            loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
            res = evaluate_model(model, loader)
            rows.append({'corruption': kind, 'severity': s, 'acc': res['acc']})
    return pd.DataFrame(rows)
""")

md("""### 5.5 Test-time augmentation (TTA)

Five-crop TTA on the best model — averages softmax probabilities over the
four corners and the center crop of a slightly larger resize. Often
worth ~1-2 accuracy points on top of a single forward pass.""")

code("""@torch.no_grad()
def evaluate_tta(run_dir: Path, batch: int = 32) -> Dict:
    model, cfg = load_run(run_dir)
    base = transforms.Resize((int(cfg.image_size * 1.15), int(cfg.image_size * 1.15)))
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    df = splits_df[splits_df['split'] == 'test'].reset_index(drop=True)

    all_probs = np.zeros((len(df), N_CLASSES), dtype=np.float32)
    y_true = np.zeros(len(df), dtype=np.int64)
    for i, row in enumerate(df.itertuples()):
        img = Image.open(row.path).convert('RGB')
        img = base(img)
        crops = transforms.functional.five_crop(img, cfg.image_size)
        x = torch.stack([norm(transforms.functional.to_tensor(c)) for c in crops])
        x = x.to(DEVICE)
        logits = model(x)
        probs = F.softmax(logits, dim=1).mean(0).cpu().numpy()
        all_probs[i] = probs
        y_true[i] = class_to_idx[getattr(row, 'class')]
    y_pred = all_probs.argmax(1)
    return {
        'acc':      float((y_pred == y_true).mean()),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'top5':     float(top_k_accuracy_score(y_true, all_probs, k=5,
                                                labels=list(range(N_CLASSES)))),
    }
""")

# =====================================================================
# 6. INTERPRETABILITY (AC209B)
# =====================================================================
md("""## 6. Interpretability — *AC209B method-not-covered-in-class component*

The MS3 case study (W → 0% recall, decomposed cleanly per signer) raised a
question the accuracy table cannot answer: **what does the model actually
look at, and where in its depth does signer identity get factored out
relative to sign identity?**

This section answers that with four techniques, none of which were
covered in 109b lecture material:

1. **Layer-by-layer linear probing** — quantify how much of a
   representation is class-relevant vs signer-relevant at each depth.
2. **TCAV (Concept Activation Vectors, Kim et al. 2018)** — quantify
   how strongly each model's prediction for class *c* is sensitive to a
   specific hand-shape concept direction.
3. **Saliency / Grad-CAM / Integrated Gradients** — visualize which
   pixels drive each prediction.
4. **Counterfactual perturbation** — find the minimal pixel change that
   flips a wrong prediction back to the correct class.

The unifying claim: these methods let us decide whether the W-collapse is
a *resolution* failure (the relevant pixels are too small), a *coverage*
failure (the model never saw P9/P10's W posture), or a *representation*
failure (the model's internal features genuinely entangle W with 4/6).""")

md("""### 6.1 Layer-by-layer linear probing

Hook every residual block of the CNN (or every transformer block of the
ViT). Extract pooled features on a held-out subset. For each layer, train
two linear classifiers:

- a **class probe** (36-way): how well can a linear model recover the sign?
- a **signer probe** (10-way): how well can a linear model recover P1-P10?

Plot probe accuracy vs depth. The depth where class probe rises and
signer probe falls is where the model *disentangles* sign from signer.
That's the canonical "subject-invariance" measurement, adapted from the
NLP probing literature (Belinkov & Glass 2019; Tenney et al. 2019).""")

code("""def get_resnet_hooks(model) -> Dict[str, nn.Module]:
    \"\"\"Return name -> module for all major ResNet stages.\"\"\"
    return {
        'stem':    model.relu,
        'layer1':  model.layer1,
        'layer2':  model.layer2,
        'layer3':  model.layer3,
        'layer4':  model.layer4,
        'avgpool': model.avgpool,
    }

def get_vit_hooks(model) -> Dict[str, nn.Module]:
    \"\"\"Return name -> module for ~6 evenly-spaced ViT blocks.\"\"\"
    n = len(model.blocks)
    idxs = list(range(0, n, max(1, n // 6)))[:6]
    if (n - 1) not in idxs: idxs.append(n - 1)
    return {f'block_{i:02d}': model.blocks[i] for i in idxs}


def extract_features(model: nn.Module, loader: DataLoader,
                      hook_modules: Dict[str, nn.Module]
                      ) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    \"\"\"Forward pass with hooks; returns:
        feats:    {layer_name: (N, D) np.array}
        labels:   (N,) class indices
        subjects: list of (N,) subject IDs (for the signer probe)
    \"\"\"
    feats: Dict[str, list] = {n: [] for n in hook_modules}
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            if isinstance(out, tuple): out = out[0]
            if out.dim() == 4:                           # CNN: B, C, H, W
                pooled = out.mean(dim=[2, 3])
            elif out.dim() == 3:                          # ViT: B, T, D
                pooled = out[:, 0, :]                     # CLS token
            else:
                pooled = out.flatten(1)
            feats[name].append(pooled.detach().cpu().float().numpy())
        return hook

    for n, m in hook_modules.items():
        handles.append(m.register_forward_hook(make_hook(n)))

    labels: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            _ = model(x)
            labels.append(y.numpy())
    for h in handles: h.remove()

    return ({n: np.concatenate(arr) for n, arr in feats.items()},
            np.concatenate(labels), [])


def make_probe_loader(image_size: int, split: str = 'val',
                      batch: int = 64, max_n: int = 4000) -> Tuple[DataLoader, pd.DataFrame]:
    df = splits_df[splits_df['split'] == split].sample(
        n=min(max_n, len(splits_df[splits_df['split'] == split])),
        random_state=SEED).reset_index(drop=True)
    eval_tf = make_transforms(image_size, training=False)
    loader = DataLoader(AslDataset(df, class_to_idx, eval_tf),
                        batch_size=batch, shuffle=False, num_workers=2)
    return loader, df


def train_probe(X_tr: np.ndarray, y_tr: np.ndarray,
                X_te: np.ndarray, y_te: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=500, n_jobs=-1, C=1.0)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def run_layer_probes(run_dir: Path, max_n: int = 3000) -> pd.DataFrame:
    \"\"\"Run class + signer probes at every hook layer.

    For the signer probe we use train signers only (to keep it well-posed):
    we extract train-set features, split 70/30 by image, and report test
    accuracy. The class probe uses the same scheme but on val features.
    \"\"\"
    model, cfg = load_run(run_dir)
    if 'resnet' in cfg.arch:
        hooks = get_resnet_hooks(model)
    else:
        hooks = get_vit_hooks(model)

    # --- class probe: train on val, test on test ---
    val_loader, val_df = make_probe_loader(cfg.image_size, 'val',  max_n=max_n)
    te_loader, te_df   = make_probe_loader(cfg.image_size, 'test', max_n=max_n)
    val_feats, val_y, _ = extract_features(model, val_loader, hooks)
    te_feats,  te_y,  _ = extract_features(model, te_loader,  hooks)

    # --- signer probe: train on train, test on train holdout ---
    tr_loader, tr_df = make_probe_loader(cfg.image_size, 'train', max_n=max_n)
    tr_feats, _, _   = extract_features(model, tr_loader, hooks)
    subj_to_idx = {s: i for i, s in enumerate(sorted(tr_df['subject'].unique()))}
    tr_subj = tr_df['subject'].map(subj_to_idx).values
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr_subj))
    cut  = int(0.7 * len(perm))
    tr_idx, ho_idx = perm[:cut], perm[cut:]

    rows = []
    for layer in hooks.keys():
        cls_acc = train_probe(val_feats[layer], val_y, te_feats[layer], te_y)
        sig_acc = train_probe(tr_feats[layer][tr_idx], tr_subj[tr_idx],
                               tr_feats[layer][ho_idx], tr_subj[ho_idx])
        rows.append({'layer': layer, 'class_probe_acc': cls_acc,
                     'signer_probe_acc': sig_acc,
                     'n_train_signers': len(subj_to_idx)})
    return pd.DataFrame(rows)


def plot_probe_curves(probe_df: pd.DataFrame, model_name: str = ''):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(probe_df))
    ax.plot(x, probe_df['class_probe_acc'],  marker='o', label='Class probe (36-way)')
    ax.plot(x, probe_df['signer_probe_acc'], marker='s', label='Signer probe')
    ax.axhline(1 / N_CLASSES, color='gray', linestyle='--', linewidth=1, label='Class chance')
    chance_signer = 1 / probe_df['n_train_signers'].iloc[0]
    ax.axhline(chance_signer, color='gray', linestyle=':', linewidth=1, label='Signer chance')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_df['layer'], rotation=30, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Linear-probe accuracy')
    ax.set_title(f'Probe accuracy vs depth — {model_name}')
    ax.legend()
    plt.tight_layout()
    return fig
""")

md("""### 6.2 TCAV — Concept Activation Vectors

For each hand-shape *concept* (e.g. "open hand", "fist", "three fingers
up"), we curate a small set of exemplar images and compute the Concept
Activation Vector (CAV) at a chosen layer:

1. Collect activations on concept images and on random images.
2. Train a linear SVM/logistic separating the two; the normal vector to
   that hyperplane is the CAV — the direction in feature space that
   represents the concept.
3. For each target class *c*, compute the directional derivative of the
   class logit *with respect to* the CAV direction, on samples from
   class *c*.
4. **TCAV score** = fraction of those directional derivatives that are
   positive.

Interpretation: TCAV(class=`W`, concept=`three_fingers_up`) > 0.9 means
that increasing the "three-fingers-up" direction in the model's internal
representation reliably increases the W-logit — strong evidence that the
model has learned this concept and uses it for W.

Reference: Kim et al., *"Interpretability Beyond Feature Attribution:
Quantitative Testing with Concept Activation Vectors (TCAV)"* (ICML 2018).""")

code("""# Concept curation: each concept is a list of class names whose images
# are used as positive exemplars. Random images are drawn from all other
# classes. This is a coarse but standard approximation.
CONCEPTS = {
    'open_hand':         ['5', 'B'],
    'fist':              ['A', 'S', 'T'],
    'one_finger_up':     ['1', 'D'],
    'three_fingers_up':  ['W', '6'],
    'pinky_extended':    ['I', 'J', 'Y'],
    'thumb_extended':    ['L', 'Y'],
    'two_fingers_up':    ['V', '2'],
}

def build_concept_loaders(concept_classes: List[str], image_size: int,
                          n_per_class: int = 30, n_random: int = 200,
                          source: str = 'train') -> Tuple[DataLoader, DataLoader]:
    \"\"\"Build (concept_loader, random_loader) for TCAV.\"\"\"
    src = splits_df[splits_df['split'] == source]
    pos_rows = []
    for c in concept_classes:
        pool = src[src['class'] == c]
        if len(pool) == 0: continue
        pos_rows.append(pool.sample(min(n_per_class, len(pool)), random_state=SEED))
    pos_df = pd.concat(pos_rows, ignore_index=True)
    neg_df = (src[~src['class'].isin(concept_classes)]
              .sample(n_random, random_state=SEED).reset_index(drop=True))

    eval_tf = make_transforms(image_size, training=False)
    pos_loader = DataLoader(AslDataset(pos_df, class_to_idx, eval_tf),
                            batch_size=32, shuffle=False, num_workers=2)
    neg_loader = DataLoader(AslDataset(neg_df, class_to_idx, eval_tf),
                            batch_size=32, shuffle=False, num_workers=2)
    return pos_loader, neg_loader


def compute_cav(model: nn.Module, hook_module: nn.Module,
                pos_loader: DataLoader, neg_loader: DataLoader) -> np.ndarray:
    \"\"\"Compute the unit-norm Concept Activation Vector at *hook_module*.\"\"\"
    pos_f, _, _ = extract_features(model, pos_loader, {'L': hook_module})
    neg_f, _, _ = extract_features(model, neg_loader, {'L': hook_module})
    X = np.concatenate([pos_f['L'], neg_f['L']], axis=0)
    y = np.concatenate([np.ones(len(pos_f['L'])), np.zeros(len(neg_f['L']))])
    clf = LogisticRegression(max_iter=500, C=1.0).fit(X, y)
    cav = clf.coef_[0]
    cav /= (np.linalg.norm(cav) + 1e-8)
    return cav


def tcav_score(model: nn.Module, hook_module: nn.Module, target_class: int,
               cav: np.ndarray, target_loader: DataLoader) -> float:
    \"\"\"Fraction of target-class samples whose class-logit gradient at
    *hook_module* points in the +cav direction.\"\"\"
    cav_t = torch.tensor(cav, dtype=torch.float32, device=DEVICE)
    sign_count = total = 0

    activation = {}
    def hook(_m, _inp, out):
        if isinstance(out, tuple): out = out[0]
        activation['x'] = out
    h = hook_module.register_forward_hook(hook)

    for x, y in target_loader:
        x = x.to(DEVICE, non_blocking=True)
        x.requires_grad_(False)
        # forward
        logits = model(x)
        a = activation['x']
        if a.dim() == 4: a = a.mean(dim=[2, 3])
        elif a.dim() == 3: a = a[:, 0, :]
        # gradient of class logit w.r.t. activation
        out = logits[:, target_class].sum()
        grad = torch.autograd.grad(out, a, retain_graph=False, create_graph=False)[0]
        # directional derivative
        dirderiv = (grad * cav_t.unsqueeze(0)).sum(dim=1)
        sign_count += (dirderiv > 0).sum().item()
        total      += dirderiv.numel()

    h.remove()
    return sign_count / max(1, total)


def get_penultimate(model: nn.Module, arch: str) -> nn.Module:
    \"\"\"Return the module whose output we use for CAVs.\"\"\"
    if 'resnet' in arch:
        return model.layer4
    return model.blocks[-1]


def run_tcav_grid(run_dir: Path, target_classes: Optional[List[str]] = None,
                  n_concept: int = 30, n_random: int = 200) -> pd.DataFrame:
    \"\"\"Compute TCAV(class, concept) for all concepts and a chosen target
    class list (default = the 8 worst classes from MS3 results).\"\"\"
    model, cfg = load_run(run_dir)
    target_classes = target_classes or ['W', 'S', 'T', '1', 'D', 'X', 'I', 'U']
    hook = get_penultimate(model, cfg.arch)

    rows = []
    for concept_name, concept_classes in CONCEPTS.items():
        # Need to enable gradient flow through the model
        for p in model.parameters(): p.requires_grad_(True)
        pos_loader, neg_loader = build_concept_loaders(
            concept_classes, cfg.image_size, n_concept, n_random, source='train')
        cav = compute_cav(model, hook, pos_loader, neg_loader)

        # Build target loaders (one per class)
        for tcls in target_classes:
            t_df = splits_df[(splits_df['split'] == 'test') &
                              (splits_df['class'] == tcls)].reset_index(drop=True)
            if len(t_df) == 0: continue
            eval_tf = make_transforms(cfg.image_size, training=False)
            t_loader = DataLoader(AslDataset(t_df, class_to_idx, eval_tf),
                                   batch_size=32, shuffle=False, num_workers=0)
            score = tcav_score(model, hook, class_to_idx[tcls], cav, t_loader)
            rows.append({'concept': concept_name, 'target_class': tcls, 'tcav': score})
    return pd.DataFrame(rows)


def plot_tcav_heatmap(tcav_df: pd.DataFrame, model_name: str = ''):
    pivot = tcav_df.pivot(index='concept', columns='target_class', values='tcav')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, vmin=0, vmax=1, cmap='RdBu_r', annot=True, fmt='.2f',
                center=0.5, ax=ax, cbar_kws={'label': 'TCAV score'})
    ax.set_title(f'TCAV(class, concept) — {model_name}')
    plt.tight_layout()
    return fig
""")

md("""### 6.3 Saliency: Grad-CAM + Integrated Gradients

For each hard test case (top confusions from MS3 §11.3), visualize:

1. **Grad-CAM** — last-conv class activation map, useful for CNNs.
2. **Integrated Gradients** — pixel-level attribution, baseline-independent.

We render side-by-side: original image | Grad-CAM | IG, for both correctly
and incorrectly classified images of the worst classes.""")

code("""def gradcam_resnet(model: nn.Module, x: torch.Tensor, target_class: int,
                   target_layer: nn.Module) -> np.ndarray:
    \"\"\"Vanilla Grad-CAM for a ResNet-style model. Returns (H, W) heatmap in [0,1].\"\"\"
    activations, gradients = {}, {}
    def fwd(_m, _i, out): activations['x'] = out
    def bwd(_m, _gi, go): gradients['x'] = go[0]
    h1 = target_layer.register_forward_hook(fwd)
    h2 = target_layer.register_full_backward_hook(bwd)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[:, target_class].sum()
    score.backward()

    a = activations['x']     # (1, C, H, W)
    g = gradients['x']       # (1, C, H, W)
    weights = g.mean(dim=[2, 3], keepdim=True)
    cam = (weights * a).sum(dim=1).relu()
    cam = cam[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h1.remove(); h2.remove()
    return cam


def integrated_gradients(model: nn.Module, x: torch.Tensor, target_class: int,
                         steps: int = 32) -> np.ndarray:
    \"\"\"Integrated Gradients (Sundararajan et al. 2017) with zero baseline.\"\"\"
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps, device=x.device)
    grads = torch.zeros_like(x)
    for a in alphas:
        x_int = baseline + a * (x - baseline)
        x_int.requires_grad_(True)
        logits = model(x_int)
        score = logits[:, target_class].sum()
        g = torch.autograd.grad(score, x_int)[0]
        grads += g
    avg_grad = grads / steps
    ig = (x - baseline) * avg_grad
    return ig[0].abs().sum(0).detach().cpu().numpy()  # (H, W)


def visualize_attribution(run_dir: Path, cases: List[Tuple[str, str]] = None,
                           n_per_case: int = 2):
    \"\"\"Render a grid: rows = (true→pred) cases, cols = original | GradCAM | IG.\"\"\"
    cases = cases or [('W', '6'), ('W', '4'), ('T', 'N'), ('D', '1'), ('X', 'Q')]
    model, cfg = load_run(run_dir)
    if 'resnet' not in cfg.arch:
        print(f'Grad-CAM helper here is ResNet-only; skip for {cfg.arch}.')
        return None

    eval_tf = make_transforms(cfg.image_size, training=False)
    test_df = splits_df[splits_df['split'] == 'test'].reset_index(drop=True)
    preds = pd.read_csv(run_dir / 'predictions_test.csv')
    test_df = test_df.assign(y_true=preds['y_true'], y_pred=preds['y_pred'])

    fig, axes = plt.subplots(len(cases) * n_per_case, 3,
                             figsize=(9, 3 * len(cases) * n_per_case))
    target_layer = model.layer4
    row = 0
    for true_c, pred_c in cases:
        sub = test_df[(test_df['class'] == true_c) &
                      (test_df['y_pred'] == class_to_idx[pred_c])]
        sub = sub.head(n_per_case)
        for _, r in sub.iterrows():
            img = Image.open(r['path']).convert('RGB')
            x = eval_tf(img).unsqueeze(0).to(DEVICE)
            x.requires_grad_(True)
            cam = gradcam_resnet(model, x, class_to_idx[pred_c], target_layer)
            ig  = integrated_gradients(model, x, class_to_idx[pred_c])

            axes[row, 0].imshow(img.resize((cfg.image_size, cfg.image_size)))
            axes[row, 0].set_title(f'{true_c} ({r[\"subject\"]}) → pred {pred_c}', fontsize=9)
            axes[row, 0].axis('off')

            axes[row, 1].imshow(img.resize((cfg.image_size, cfg.image_size)))
            axes[row, 1].imshow(cam, alpha=0.5, cmap='jet',
                                extent=(0, cfg.image_size, cfg.image_size, 0))
            axes[row, 1].set_title('Grad-CAM ↑pred', fontsize=9); axes[row, 1].axis('off')

            axes[row, 2].imshow(ig, cmap='hot')
            axes[row, 2].set_title('Integrated Gradients', fontsize=9)
            axes[row, 2].axis('off')
            row += 1
    plt.tight_layout()
    return fig
""")

md("""### 6.4 Counterfactual perturbation

For each W-test image misclassified as 4 or 6, find the *minimal* L∞
perturbation that flips the prediction back to W. Implementation: a
targeted PGD attack against the W class, with a tight ε budget. The
perturbation magnitude tells us *how close to a correct decision* the
model was; the perturbation pattern tells us *which pixel region* the
model was relying on.

Hypothesis: if the W → 4 collapse for P9 is a finger-position issue, the
counterfactual perturbation should concentrate on the pinky region.""")

code("""def counterfactual_pgd(model: nn.Module, x: torch.Tensor, true_class: int,
                       eps: float = 0.05, alpha: float = 0.005,
                       steps: int = 60) -> Tuple[torch.Tensor, bool]:
    \"\"\"Targeted PGD that pushes prediction toward *true_class*.

    Returns (delta, success). delta is the perturbation; if success the
    model now predicts true_class on x + delta.\"\"\"
    delta = torch.zeros_like(x, requires_grad=True)
    target = torch.tensor([true_class], device=x.device)
    for _ in range(steps):
        logits = model(x + delta)
        loss = F.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)
        delta.requires_grad_(True)
        with torch.no_grad():
            pred = (x + delta).clamp(-3, 3)
            new_logits = model(pred)
            if new_logits.argmax(1).item() == true_class:
                return delta.detach(), True
    return delta.detach(), False


def visualize_counterfactuals(run_dir: Path, true_class: str = 'W',
                              n_examples: int = 4, eps: float = 0.05):
    model, cfg = load_run(run_dir)
    eval_tf = make_transforms(cfg.image_size, training=False)
    test_df = splits_df[splits_df['split'] == 'test'].reset_index(drop=True)
    preds = pd.read_csv(run_dir / 'predictions_test.csv')
    test_df = test_df.assign(y_pred=preds['y_pred'])

    sub = test_df[(test_df['class'] == true_class) &
                  (test_df['y_pred'] != class_to_idx[true_class])].head(n_examples)
    fig, axes = plt.subplots(len(sub), 4, figsize=(12, 3 * len(sub)))
    if len(sub) == 1: axes = axes[None, :]

    for i, (_, r) in enumerate(sub.iterrows()):
        img = Image.open(r['path']).convert('RGB')
        x = eval_tf(img).unsqueeze(0).to(DEVICE)
        delta, success = counterfactual_pgd(model, x, class_to_idx[true_class], eps=eps)

        axes[i, 0].imshow(img.resize((cfg.image_size, cfg.image_size)))
        axes[i, 0].set_title(f'Original ({r[\"subject\"]})\\npred {idx_to_class[r[\"y_pred\"]]}',
                             fontsize=9); axes[i, 0].axis('off')

        x_adv = (x + delta).clamp(-3, 3)
        adv_img = x_adv[0].cpu()
        # de-normalize for display
        mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        std  = torch.tensor(IMAGENET_STD )[:, None, None]
        disp = (adv_img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, 1].imshow(disp)
        axes[i, 1].set_title(f'Counterfactual\\n(success={success})', fontsize=9)
        axes[i, 1].axis('off')

        delta_mag = delta[0].abs().sum(0).cpu().numpy()
        axes[i, 2].imshow(delta_mag, cmap='hot')
        axes[i, 2].set_title(f'|Δ| (L∞={eps})', fontsize=9); axes[i, 2].axis('off')

        axes[i, 3].imshow(img.resize((cfg.image_size, cfg.image_size)))
        axes[i, 3].imshow(delta_mag, alpha=0.55, cmap='hot',
                          extent=(0, cfg.image_size, cfg.image_size, 0))
        axes[i, 3].set_title('Δ overlay', fontsize=9); axes[i, 3].axis('off')

    plt.suptitle(f'Counterfactual perturbations: misclassified {true_class} → push back to {true_class}',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    return fig
""")

md("""### 6.5 ViT attention head specialization

For a ViT, each attention head produces an N×N matrix relating image
patches. We compute, for the [CLS] token, the average attention weight
each head pays to *foreground* (hand) vs *background* patches. Heads
with high foreground concentration are interpreted as "hand-focused";
those with diffuse attention are "global-context" heads.

Foreground is approximated by a lightweight rule: the bottom-half of each
image's central crop tends to contain the hand in this dataset (per the
§7.4 MS3 EDA). For a more rigorous bbox we'd run MediaPipe; this
heuristic suffices for the head-comparison narrative.""")

code("""@torch.no_grad()
def vit_head_attention_to_fg(run_dir: Path, n_imgs: int = 200,
                              fg_mask_fn: Optional[Callable] = None) -> pd.DataFrame:
    \"\"\"Per-(layer, head) average attention weight from CLS to FG patches.\"\"\"
    model, cfg = load_run(run_dir)
    if 'vit' not in cfg.arch and cfg.arch != 'dinov2':
        raise ValueError('ViT-only analysis')
    eval_tf = make_transforms(cfg.image_size, training=False)
    df = splits_df[splits_df['split'] == 'test'].sample(n=n_imgs, random_state=SEED)
    loader = DataLoader(AslDataset(df, class_to_idx, eval_tf),
                        batch_size=16, shuffle=False, num_workers=2)

    # Patch grid
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple): patch_size = patch_size[0]
    n_patches_side = cfg.image_size // patch_size
    if fg_mask_fn is None:
        # Default: bottom half of the image is FG
        mask = np.zeros((n_patches_side, n_patches_side), dtype=bool)
        mask[n_patches_side // 4 : 3 * n_patches_side // 4,
             n_patches_side // 4 : 3 * n_patches_side // 4] = True
        mask_flat = mask.flatten()
    else:
        mask_flat = fg_mask_fn(n_patches_side)

    # Hook attention modules — timm ViT blocks have a .attn submodule
    attn_outputs = {i: [] for i in range(len(model.blocks))}

    def make_hook(idx):
        def hook(module, inp, out):
            # out: (B, T, D); we'll need attention weights — use module internals
            pass
        return hook

    # Easier: monkey-patch attn forward to record weights
    captured = {}
    saved_forwards = []
    for i, blk in enumerate(model.blocks):
        attn = blk.attn
        orig_forward = attn.forward
        def patched(self, x, _idx=i, _orig=orig_forward):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim if hasattr(self,'head_dim') else C//self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_w = (q @ k.transpose(-2, -1)) * (self.scale if hasattr(self,'scale') else 1.0)
            attn_w = attn_w.softmax(dim=-1)
            captured.setdefault(_idx, []).append(attn_w.detach().cpu())
            x = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x); x = self.proj_drop(x)
            return x
        attn.forward = patched.__get__(attn, type(attn))
        saved_forwards.append((attn, orig_forward))

    for x, _ in loader:
        x = x.to(DEVICE)
        _ = model(x)

    # Restore
    for attn, orig in saved_forwards:
        attn.forward = orig

    rows = []
    for layer_idx, ws in captured.items():
        W = torch.cat(ws, dim=0)  # (B, H, T, T)
        # CLS row → patch columns; T = 1 + n_patches (or 1 + n_patches + reg_tokens for dinov2)
        cls_to_patch = W[:, :, 0, 1:]  # (B, H, P)
        # Trim register tokens if any (DINOv2 has 4 register tokens)
        P = cls_to_patch.shape[-1]
        n_real = n_patches_side ** 2
        if P > n_real:
            cls_to_patch = cls_to_patch[..., -n_real:]
        for head in range(cls_to_patch.shape[1]):
            attn_to_fg  = cls_to_patch[:, head][:, mask_flat].sum(-1)
            attn_to_all = cls_to_patch[:, head].sum(-1)
            ratio = (attn_to_fg / attn_to_all.clamp(min=1e-6)).mean().item()
            rows.append({'layer': layer_idx, 'head': head, 'fg_attention': ratio})
    return pd.DataFrame(rows)


def plot_head_specialization(df: pd.DataFrame, model_name: str = ''):
    pivot = df.pivot(index='head', columns='layer', values='fg_attention')
    fig, ax = plt.subplots(figsize=(min(14, 0.6 * pivot.shape[1] + 2), 4))
    sns.heatmap(pivot, vmin=0, vmax=1, cmap='RdBu_r', center=0.5,
                cbar_kws={'label': 'fraction CLS attention to FG patches'}, ax=ax)
    ax.set_title(f'ViT attention-head FG specialization — {model_name}')
    ax.set_xlabel('Block index'); ax.set_ylabel('Head')
    plt.tight_layout()
    return fig
""")

# =====================================================================
# 7. FINAL SUMMARY + REFERENCES
# =====================================================================
md("""## 7. Final Summary

The cell below assembles the master comparison table consumed by the MS4
report. It joins (a) held-out P9/P10 metrics, (b) per-signer breakdown,
(c) external Kaggle metrics, (d) TTA delta — into a single dataframe per
model.""")

code("""def assemble_summary() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        if not (run_dir / 'ckpt_best.pt').exists(): continue
        if not (run_dir / 'predictions_test.csv').exists():
            evaluate_run(run_dir)
        m_path = run_dir / 'metrics.json'
        m = json.loads(m_path.read_text()) if m_path.exists() else None
        if m is None:
            res = evaluate_run(run_dir)
            m = {'name': run_dir.name, 'acc': res['acc'],
                 'macro_f1': res['macro_f1'], 'top5': res['top5']}
            m_path.write_text(json.dumps(m, indent=2))
        # Per-signer split
        try:
            ps = per_signer_overall(run_dir)
            m['acc_P9']  = float(ps.get('P9',  np.nan))
            m['acc_P10'] = float(ps.get('P10', np.nan))
        except Exception as e:
            m['acc_P9'] = m['acc_P10'] = float('nan')
        rows.append(m)
    df = pd.DataFrame(rows).sort_values('acc', ascending=False).reset_index(drop=True)
    return df

# summary_df = assemble_summary(); summary_df
""")

md("""## 8. References & AI-assistance disclosure

**Datasets**

- *American Sign Language Dataset* (Ayush Rai, 2020) — Kaggle, CC0.
  https://www.kaggle.com/datasets/ayuraj/asl-dataset
- *ASL Hand Landmark and Gesture Dataset* (avinashkr090502) — Kaggle.
  https://www.kaggle.com/datasets/iamavinashkr090502/asl-hand-landmark-and-gesture-dataset

**Models / pretraining**

- He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image
  Recognition.* CVPR 2016. (ResNet-18, ResNet-50.)
- Dosovitskiy, A. et al. *An Image is Worth 16x16 Words: Transformers for
  Image Recognition at Scale.* ICLR 2021. (ViT-B/16.)
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without
  Supervision.* TMLR 2024.
- timm (PyTorch Image Models): Wightman, R. https://github.com/huggingface/pytorch-image-models

**Training / regularization**

- Cubuk, E. et al. *RandAugment.* CVPRW 2020.
- Zhang, H. et al. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018.
- Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization
  (AdamW).* ICLR 2019.

**Interpretability — AC209B novel-method core**

- Kim, B. et al. *Interpretability Beyond Feature Attribution: Quantitative
  Testing with Concept Activation Vectors (TCAV).* ICML 2018.
- Selvaraju, R. et al. *Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization.* ICCV 2017.
- Sundararajan, M., Taly, A., Yan, Q. *Axiomatic Attribution for Deep
  Networks (Integrated Gradients).* ICML 2017.
- Belinkov, Y., Glass, J. *Analysis Methods in Neural Language Processing:
  A Survey.* TACL 2019. (Linear-probing methodology.)
- Tenney, I. et al. *BERT Rediscovers the Classical NLP Pipeline.*
  ACL 2019. (Layer-by-layer probing.)
- Madry, A. et al. *Towards Deep Learning Models Resistant to Adversarial
  Attacks (PGD).* ICLR 2018. (Counterfactual perturbations.)

**AI assistance disclosure**

This notebook's scaffolding (training loop, evaluation helpers, probing
implementations) was developed in collaboration with Claude (Anthropic).
All design decisions, dataset curation, model choices, hyperparameter
selection, and final analyses were authored by the project team. AI was
used as a coding accelerant, not a methodological co-author.
""")

# =====================================================================
# Build the notebook
# =====================================================================
nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}
with open(OUT, 'w') as f:
    nbf.write(nb, f)
print(f'Wrote {OUT} with {len(cells)} cells')
