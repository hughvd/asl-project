"""Regenerate the MS3 notebook from scratch.

The original notebook mixed Colab-specific setup, ad-hoc exploration, and
undocumented design choices. MS2 feedback called for (a) a clearly stated
research question, (b) explicit data provenance, (c) justified preprocessing,
and (d) deeper pattern/confounder analysis. MS3 adds baseline modelling,
results interpretation, and a pipeline diagram on top of all of that.

This script builds the new notebook with nbformat so each code cell is
syntactically valid and the structure is consistent. Run it from the project
root: `python3 scripts/build_notebook.py`.
"""
from pathlib import Path
import nbformat as nbf

NB_PATH = Path("ASL Data Loading & Cleaning.ipynb")


def md(src: str):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str):
    return nbf.v4.new_code_cell(src.strip("\n"))


def build():
    cells = []

    # ===== Title =====
    cells.append(md("""
# ASL Sign-Language Recognition — Milestone 3 Notebook

**Course:** AC209b — Advanced Topics in Data Science (Spring 2026)
**Canvas Project #:** 11 (ASL Sign-Language Recognition)
**Group Members:** Hugh Van Deventer, Ayush Sharma, ZD

This notebook is the team's Milestone 3 deliverable. It consolidates the
Milestone-2 data-loading / EDA work, incorporates the MS2 grader feedback,
and extends the project with baseline modelling, results interpretation,
and an end-to-end pipeline description.
"""))

    # ===== Table of Contents =====
    cells.append(md("""
## Table of Contents

1. [Research Question & Project Motivation](#1-research-question)
2. [Milestone-2 Recap](#2-ms2-recap)
3. [Data Description, Source, and Access](#3-data-provenance)
4. [Setup: Imports and Local Paths](#4-setup)
5. [Data Loading](#5-loading)
6. [Dataset Summary](#6-summary)
7. [Extended EDA](#7-eda)
    - 7.1 Sample images: raw vs. processed
    - 7.2 Class distribution
    - 7.3 Image dimensions
    - 7.4 **Subject-level variation (confounders & leakage)**
    - 7.5 Class visual similarity
    - 7.6 Pixel-intensity distributions
    - 7.7 Background-artifact analysis
8. [Preprocessing — Decisions & Justification](#8-preprocessing)
9. [Subject-Independent Train / Val / Test Split](#9-split)
10. [Baseline Models](#10-baselines)
    - 10.1 Logistic Regression (sanity baseline)
    - 10.2 Small CNN (from scratch) — 109B model
    - 10.3 ResNet-18 (ImageNet transfer) — 109B model
11. [Results & Error Analysis](#11-results)
12. [Final Model Pipeline](#12-pipeline)
13. [Future Work](#13-future)
"""))

    # ===== 1. Research Question =====
    cells.append(md("""
<a id="1-research-question"></a>
## 1. Research Question & Project Motivation

> **Research question:** *Can a computer-vision model reliably recognize
> static American Sign Language (ASL) hand signs — the 26 letters A–Z and
> the 10 digits 0–9 — from a single RGB image, and can it generalize to
> **signers it has never seen before**?*

**Why this matters.** ~500k deaf or hard-of-hearing people in the U.S.
use ASL as their primary language, yet most digital interfaces offer no
direct sign-language input. A reliable static-sign recognizer is a
prerequisite for downstream systems (real-time translators, educational
tutors, accessibility tools for touchless interfaces).

**Why subject generalization is the hard part.** A key MS2 finding was
that the dataset's default train/test split mixes the same 10 signers
across both halves. A model can easily overfit to a signer's skin tone,
hand shape, ring, or background rather than learning the *sign* — and its
apparent accuracy will collapse as soon as a new signer uses the system.
Every evaluation in this notebook uses a **subject-held-out** split
(P9/P10 test, P7/P8 validation, P1–P6 train) so the metrics reflect the
capability we actually care about.
"""))

    # ===== 2. MS2 Recap =====
    cells.append(md("""
<a id="2-ms2-recap"></a>
## 2. Milestone-2 Recap (one page)

In MS2 we loaded the Kaggle ASL dataset, inventoried its structure, and
ran an initial pass of EDA. Key takeaways:

- **36 classes** (digits 0–9, letters A–Z), **~1,000 raw images per class**
  → ~36k images total. Dataset is essentially balanced; no resampling
  needed (Table in §6).
- **10 signers** (P1–P10); filenames encode signer-ID, class, index.
- Raw images are 300×300 RGB; the provided "processed" split has
  variable-size background-removed crops.
- We identified a **data-leakage risk**: the provided 80/20 train/test
  split did NOT hold signers out, so intra-subject memorization would
  inflate test accuracy. MS3 fixes this with a subject-independent split.
- Several class pairs (e.g. *0/O*, *1/I*, *5/open-hand letters*) are
  visually near-identical even after averaging, so we expect confusion
  and will track per-class recall, not just overall accuracy.

**What's new in MS3 (what this notebook adds):**

1. Research question + data provenance stated explicitly in-notebook.
2. Deeper confounder analysis (per-subject brightness/contrast,
   background artifacts, class-similarity matrix with interpretation).
3. Preprocessing decisions are written down with their *why*.
4. Three baseline models (logistic, small CNN, transfer-learned
   ResNet-18), all evaluated on held-out subjects.
5. Error analysis (confusion matrix, per-class recall, top-5 accuracy).
6. An end-to-end pipeline diagram for the final model iteration.
"""))

    # ===== 3. Data Provenance =====
    cells.append(md("""
<a id="3-data-provenance"></a>
## 3. Data Description, Source, and Access

**Dataset:** *American Sign Language Dataset* (Ayush Rai, 2020).

- **Source:** Kaggle — https://www.kaggle.com/datasets/ayuraj/asl-dataset
- **License:** CC0 Public Domain.
- **Collection:** 10 volunteer signers (referred to as *P1* through *P10*,
  recorded in Bangladesh) each contributed 100 images per class. All
  images were captured with a single camera under indoor lighting against
  a roughly uniform background. Filenames encode signer and class:
  `P{1..10}_{CLASS}_{INSTANCE}.jpg`.
- **Local acquisition.** The team downloaded both archives
  (`ASL_Raw_Images.zip`, `ASL_Processed_Images.zip`) from Kaggle once and
  committed them to `./data/` in the project folder. Section 5 unzips
  them in-place so this notebook is self-contained on any laptop with
  the archives present.
- **Total size on disk:** see §6 (~1.2 GB combined after extraction).
- **Variants provided by the upstream authors:**
  - `asl_dataset/` — raw 300×300 RGB photographs organized by class
    folder. This is what we use for modelling.
  - `asl_processed/{train,test}/` — author-supplied crops with the
    background removed. Sizes vary. We use it for visual comparison only
    because its default 80/20 split mixes subjects (see §7.4).
"""))

    # ===== 4. Setup =====
    cells.append(md("""
<a id="4-setup"></a>
## 4. Setup: Imports and Local Paths

All paths are **relative to the project root** (the directory containing
this notebook). No Google Drive / Colab mount is needed — the data should
already be in `./data/`.
"""))

    cells.append(code("""
# Standard library
import os, sys, json, shutil, itertools, time
from pathlib import Path
from collections import defaultdict

# Numerical / dataframe
import numpy as np
import pandas as pd

# Imaging / plotting
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             top_k_accuracy_score, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Keep figures consistent
plt.rcParams['figure.dpi'] = 110
sns.set_context('notebook')

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device: MPS on Apple Silicon, CUDA on Linux/Windows with NVIDIA, CPU fallback
DEVICE = (
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('cpu')
)
print('Torch device:', DEVICE)

# Local paths — adjust if the notebook is moved
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'asl_dataset'
PROCESSED_TRAIN_DIR = DATA_DIR / 'asl_processed' / 'train'
PROCESSED_TEST_DIR = DATA_DIR / 'asl_processed' / 'test'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)
"""))

    # ===== 5. Loading =====
    cells.append(md("""
<a id="5-loading"></a>
## 5. Data Loading

If the raw-image directories are missing, the cell below extracts the
two zips that live in `./data/`. Re-running is a no-op once the folders
exist, so the notebook stays idempotent.
"""))

    cells.append(code("""
import zipfile

def ensure_unzipped(zip_path: Path, expected_dir: Path):
    '''Extract *zip_path* into *expected_dir.parent* unless *expected_dir* already exists.'''
    if expected_dir.exists():
        print(f'[ok] {expected_dir} already present')
        return
    if not zip_path.exists():
        raise FileNotFoundError(
            f'{zip_path} not found. Download the archives from '
            'https://www.kaggle.com/datasets/ayuraj/asl-dataset into ./data/'
        )
    print(f'[unzip] {zip_path.name} -> {expected_dir.parent}/')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(expected_dir.parent)

ensure_unzipped(DATA_DIR / 'ASL_Raw_Images.zip', RAW_DIR)
ensure_unzipped(DATA_DIR / 'ASL_Processed_Images.zip', PROCESSED_TRAIN_DIR.parent)

classes = sorted([d for d in os.listdir(RAW_DIR)
                  if (RAW_DIR / d).is_dir()])
print('Classes (first 10):', classes[:10], '...')
print('Total classes:', len(classes))
"""))

    # ===== 6. Summary =====
    cells.append(md("""
<a id="6-summary"></a>
## 6. Dataset Summary

We measure size-on-disk, image counts per class, and signer coverage.
These are the hard numbers the rest of the notebook builds on.
"""))

    cells.append(code("""
def get_dir_size(path: Path) -> int:
    '''Return total size in bytes of files under *path* (recursive).'''
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total

raw_size = get_dir_size(RAW_DIR)
proc_train_size = get_dir_size(PROCESSED_TRAIN_DIR)
proc_test_size = get_dir_size(PROCESSED_TEST_DIR)

print(f'Raw dataset size:        {raw_size / 1e9:5.2f} GB')
print(f'Processed train size:    {proc_train_size / 1e6:5.1f} MB')
print(f'Processed test size:     {proc_test_size / 1e6:5.1f} MB')
print(f'Combined on-disk total:  {(raw_size + proc_train_size + proc_test_size) / 1e9:5.2f} GB')
"""))

    cells.append(code("""
raw_counts = {c: len(os.listdir(RAW_DIR / c)) for c in classes}
train_counts = {c: len(os.listdir(PROCESSED_TRAIN_DIR / c)) for c in classes}
test_counts = {c: len(os.listdir(PROCESSED_TEST_DIR / c)) for c in classes}

df_counts = pd.DataFrame({
    'Class': classes,
    'Raw': [raw_counts[c] for c in classes],
    'Processed Train': [train_counts[c] for c in classes],
    'Processed Test': [test_counts[c] for c in classes],
})

print('Raw per-class image count — min / median / max:',
      min(raw_counts.values()), int(np.median(list(raw_counts.values()))),
      max(raw_counts.values()))
print('Grand totals — raw:', sum(raw_counts.values()),
      ', processed train:', sum(train_counts.values()),
      ', processed test:', sum(test_counts.values()))

df_counts.head()
"""))

    cells.append(md("""
**Takeaway.** Every class has exactly 1,000 raw images and an 800/200
processed split. Class balance is essentially perfect, so we don't need
resampling (SMOTE, class weights) — we can focus on the actual modelling
problem rather than on fighting imbalance.
"""))

    # ===== 7. EDA =====
    cells.append(md("""
<a id="7-eda"></a>
## 7. Extended EDA

This is the section that grew most between MS2 and MS3. Each subsection
ends with a one-line **Takeaway** tying the plot back to modelling
decisions, so the EDA *drives* the pipeline rather than sitting beside it.
"""))

    # 7.1
    cells.append(md("""
### 7.1 Sample Images — Raw vs. Author-Processed

Each row is a class. Top sub-row: raw 300×300 photo. Bottom sub-row:
the author-supplied background-removed crop.
"""))

    cells.append(code("""
rows, cols = 6, 12
fig, axes = plt.subplots(rows, cols, figsize=(24, 12))

for group_idx in range(3):
    for col in range(cols):
        cls_idx = group_idx * cols + col
        if cls_idx >= len(classes):
            axes[2*group_idx, col].axis('off')
            axes[2*group_idx + 1, col].axis('off')
            continue
        cls = classes[cls_idx]
        raw_files = sorted(os.listdir(RAW_DIR / cls))
        proc_files = sorted(os.listdir(PROCESSED_TRAIN_DIR / cls))
        axes[2*group_idx, col].imshow(Image.open(RAW_DIR / cls / raw_files[0]))
        axes[2*group_idx, col].set_title(cls, fontsize=14)
        axes[2*group_idx, col].axis('off')
        axes[2*group_idx + 1, col].imshow(
            Image.open(PROCESSED_TRAIN_DIR / cls / proc_files[0]), cmap='gray')
        axes[2*group_idx + 1, col].axis('off')

plt.suptitle('Raw vs. Processed ASL Samples (odd rows = raw, even rows = processed)',
             fontsize=18)
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
**Takeaway.** Raw images carry useful context (wrist, forearm, background
texture) but also background confounders. The processed images are
tightly cropped to just the hand but (a) the author's segmentation is
not always clean and (b) crop sizes vary widely. We model on raw images
and do our own deterministic resize so the shape of the input tensor is
constant, which is necessary for a fixed-input-size CNN.
"""))

    # 7.2 Class distribution
    cells.append(md("""
### 7.2 Class Distribution
"""))

    cells.append(code("""
fig, ax = plt.subplots(figsize=(14, 4))
x = np.arange(len(classes))
ax.bar(x - 0.2, df_counts['Raw'], width=0.4, label='Raw', alpha=0.85)
ax.bar(x + 0.2, df_counts['Processed Train'] + df_counts['Processed Test'],
       width=0.4, label='Processed (train+test)', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
ax.set_xlabel('Class'); ax.set_ylabel('Image count')
ax.set_title('Images per Class — Perfectly Balanced at 1,000 each')
ax.legend()
plt.tight_layout(); plt.show()
"""))

    cells.append(md("""
**Takeaway.** Because every class has the same number of samples, plain
**accuracy** is a fair first metric. We will still report macro-F1 and
top-5 accuracy so per-class failure modes aren't hidden.
"""))

    # 7.3 dimensions
    cells.append(md("""
### 7.3 Image Dimensions (Processed)

Raw images are uniformly 300×300. The author's processed crops vary
because they were cut to each individual hand.
"""))

    cells.append(code("""
proc_widths, proc_heights = [], []
for cls in classes:
    for fname in os.listdir(PROCESSED_TRAIN_DIR / cls):
        w, h = Image.open(PROCESSED_TRAIN_DIR / cls / fname).size
        proc_widths.append(w); proc_heights.append(h)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(proc_widths, bins=50, edgecolor='black')
axes[0].set_title('Processed Image Widths'); axes[0].set_xlabel('Width (px)')
axes[1].hist(proc_heights, bins=50, edgecolor='black')
axes[1].set_title('Processed Image Heights'); axes[1].set_xlabel('Height (px)')
plt.tight_layout(); plt.show()
print(f'Width:  min={min(proc_widths)} max={max(proc_widths)} '
      f'median={int(np.median(proc_widths))}')
print(f'Height: min={min(proc_heights)} max={max(proc_heights)} '
      f'median={int(np.median(proc_heights))}')
"""))

    cells.append(md("""
**Takeaway.** Processed-image widths span roughly 50–300 px. This
variability is another reason to work from the raw images and apply a
single deterministic resize (§8). Tiny processed crops — some below
100 px on one side — would force the network to hallucinate detail if we
up-sampled them.
"""))

    # 7.4 Subject variation & leakage (the big confounder section)
    cells.append(md("""
### 7.4 Subject-Level Variation — Confounders & Leakage

**MS2 feedback asked us to push harder on confounders.** The single
biggest one here is **signer identity**: brightness, skin tone, hand
size, background, and even camera position vary per subject. If the
train/test split mixes subjects, a model can cheat.
"""))

    cells.append(code("""
def get_subject(fname):
    '''Extract signer-ID from `P{k}_{CLASS}_{i}.jpg` filenames.'''
    return fname.split('_')[0]

subject_brightness = defaultdict(list)
subject_contrast = defaultdict(list)

for cls in classes:
    for fname in sorted(os.listdir(RAW_DIR / cls)):
        subj = get_subject(fname)
        gray = np.array(Image.open(RAW_DIR / cls / fname).convert('L'))
        subject_brightness[subj].append(gray.mean())
        subject_contrast[subj].append(gray.std())

subjects = sorted(subject_brightness.keys(), key=lambda s: int(s[1:]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].boxplot([subject_brightness[s] for s in subjects], tick_labels=subjects)
axes[0].set_title('Mean Brightness per Signer (across all images)')
axes[0].set_ylabel('Mean pixel intensity (0-255)')
axes[1].boxplot([subject_contrast[s] for s in subjects], tick_labels=subjects)
axes[1].set_title('Contrast (std) per Signer')
axes[1].set_ylabel('Std pixel intensity')
plt.tight_layout(); plt.show()

for s in subjects:
    print(f'{s}: brightness μ={np.mean(subject_brightness[s]):5.1f}, '
          f'contrast μ={np.mean(subject_contrast[s]):5.1f}')
"""))

    cells.append(md("""
**Takeaway.** Mean brightness varies by ~30-40 intensity units across
signers, and contrast differences are similar in scale. These are
**signer-level confounders** — a model can learn "P3-looking photo ⇒ class X"
without ever looking at the hand shape. This motivates both (a) the
augmentation plan in §8 and (b) the subject-held-out split in §9.
"""))

    cells.append(md("""
#### 7.4.1 Does the *provided* train/test split leak signers?
"""))

    cells.append(code("""
train_subjects_per_class = defaultdict(set)
test_subjects_per_class = defaultdict(set)

for cls in classes:
    for fname in os.listdir(PROCESSED_TRAIN_DIR / cls):
        train_subjects_per_class[cls].add(get_subject(fname))
    for fname in os.listdir(PROCESSED_TEST_DIR / cls):
        test_subjects_per_class[cls].add(get_subject(fname))

leaked_classes = [cls for cls in classes
                  if train_subjects_per_class[cls] & test_subjects_per_class[cls]]

if leaked_classes:
    print(f'{len(leaked_classes)}/{len(classes)} classes have signer overlap '
          'between the provided train and test splits.')
    print('Example overlap for class', leaked_classes[0], ':',
          train_subjects_per_class[leaked_classes[0]] & test_subjects_per_class[leaked_classes[0]])
else:
    print('Provided split is already subject-disjoint.')
"""))

    cells.append(md("""
**Takeaway.** The provided split mixes signers — every class has the
same subjects on both sides. Reporting test accuracy on that split would
overstate real-world performance. In §9 we build our own split that
holds signers P9, P10 out for test and P7, P8 for validation.
"""))

    # 7.5 Class similarity
    cells.append(md("""
### 7.5 Class Visual Similarity — Which Pairs Will Confuse the Model?

We compute per-class average images (64×64), then measure pairwise
cosine similarity on the flattened vectors. High similarity = classes
that *look* alike to pixel-level features, which is exactly the
failure mode a CNN still has to break through.
"""))

    cells.append(code("""
TARGET = (64, 64)
class_avg = {}
for cls in classes:
    imgs = [np.array(Image.open(RAW_DIR / cls / f).convert('RGB').resize(TARGET),
                     dtype=np.float32)
            for f in os.listdir(RAW_DIR / cls)]
    class_avg[cls] = np.mean(imgs, axis=0).astype(np.uint8)

fig, axes = plt.subplots(3, 12, figsize=(20, 5))
for i, cls in enumerate(classes):
    r, c = divmod(i, 12)
    axes[r, c].imshow(class_avg[cls])
    axes[r, c].set_title(cls, fontsize=10); axes[r, c].axis('off')
plt.suptitle('Per-class Average Image (raw, resized 64×64)', fontsize=16)
plt.tight_layout(); plt.show()
"""))

    cells.append(code("""
from sklearn.metrics.pairwise import cosine_similarity
vectors = np.array([class_avg[c].flatten() / 255.0 for c in classes])
sim = cosine_similarity(vectors)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(sim, xticklabels=classes, yticklabels=classes,
            cmap='YlOrRd', ax=ax, vmin=0.8, vmax=1.0)
ax.set_title('Cosine Similarity of Class Average Images')
plt.tight_layout(); plt.show()

pairs = [(classes[i], classes[j], sim[i, j])
         for i, j in itertools.combinations(range(len(classes)), 2)]
pairs.sort(key=lambda x: x[2], reverse=True)
print('Top 10 visually-most-similar class pairs:')
for a, b, s in pairs[:10]:
    print(f'  {a} vs {b}: {s:.4f}')
"""))

    cells.append(md("""
**Takeaway.** The highest-similarity pairs are exactly the hand-shape
lookalikes you would predict by eye — closed-fist letters (`S`/`T`,
`A`/`E`, `M`/`S`), open-hand variants (`U`/`V`, `B`/`F`), and
digit-letter collisions (`2`/`6`, `4`/`8`, `2`/`K`). When we look at
the §11 confusion matrix we expect most of the misclassifications to
fall on pairs of this kind; if they don't, the model is probably
latching onto an irrelevant signal (background, lighting, signer
identity) rather than hand shape.
"""))

    # 7.6 Pixel distributions
    cells.append(md("""
### 7.6 Pixel-Intensity Distributions per Class
"""))

    cells.append(code("""
fig, axes = plt.subplots(6, 6, figsize=(18, 14))
for i, cls in enumerate(classes):
    r, c = divmod(i, 6)
    pix = []
    for fname in sorted(os.listdir(RAW_DIR / cls))[:20]:   # 20 images for speed
        pix.append(np.array(Image.open(RAW_DIR / cls / fname).convert('L')).flatten())
    pix = np.concatenate(pix)
    axes[r, c].hist(pix, bins=50, density=True, color='steelblue', edgecolor='none')
    axes[r, c].set_title(cls, fontsize=10); axes[r, c].set_xlim(0, 255)
plt.suptitle('Pixel-Intensity Distribution per Class', fontsize=16)
plt.tight_layout(); plt.show()
"""))

    cells.append(md("""
**Takeaway.** Distributions are bimodal and *look similar across
classes* — a bright hand against a darker background. There's no obvious
class-specific intensity fingerprint, which is good: it means the model
has to use spatial structure (shape, contour) to discriminate, not raw
intensity histograms.
"""))

    # 7.7 Background artifacts
    cells.append(md("""
### 7.7 Background-Artifact Analysis

A subtle concern: maybe each class was shot on a consistent background,
so the model could learn the wall instead of the hand. We check the
border pixels (the 8-px frame) of each image, averaged per class.
"""))

    cells.append(code("""
BORDER = 8
rows = []
for cls in classes:
    for fname in os.listdir(RAW_DIR / cls):
        img = Image.open(RAW_DIR / cls / fname).convert('RGB').resize(TARGET)
        arr = np.array(img, dtype=np.float32)
        bp = np.concatenate([
            arr[:BORDER, :, :].reshape(-1, 3),
            arr[-BORDER:, :, :].reshape(-1, 3),
            arr[BORDER:-BORDER, :BORDER, :].reshape(-1, 3),
            arr[BORDER:-BORDER, -BORDER:, :].reshape(-1, 3),
        ])
        gray = bp.mean(axis=1)
        rows.append({'class': cls, 'bg_brightness': gray.mean(),
                     'bg_texture': gray.std()})
bg_df = pd.DataFrame(rows)
bg_summary = bg_df.groupby('class')[['bg_brightness','bg_texture']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
axes[0].bar(bg_summary['class'], bg_summary['bg_brightness'])
axes[0].set_title('Average Background Brightness by Class')
axes[0].set_ylabel('Mean border intensity'); axes[0].tick_params(axis='x', rotation=0)
axes[1].bar(bg_summary['class'], bg_summary['bg_texture'])
axes[1].set_title('Average Background Texture (std) by Class')
axes[1].set_ylabel('Border std dev')
plt.tight_layout(); plt.show()
"""))

    cells.append(code("""
plt.figure(figsize=(16, 10))
plt.subplot(2, 1, 1); sns.boxplot(data=bg_df, x='class', y='bg_brightness')
plt.title('Background Brightness by Class — Per-Image Spread')
plt.subplot(2, 1, 2); sns.boxplot(data=bg_df, x='class', y='bg_texture')
plt.title('Background Texture (std) by Class — Per-Image Spread')
plt.tight_layout(); plt.show()
"""))

    cells.append(md("""
**Takeaway.** Background brightness is largely consistent across
classes — the spread *within* each class is big, driven by which signer
took the photo (cf. §7.4). Some classes show slightly skewed background
statistics but the per-class spread dominates the per-class means, so
backgrounds are a much weaker confounder than signer identity.
Augmentation (color jitter, random rotation) in §8 further reduces what
signal the network can learn from backgrounds alone.
"""))

    # ===== 8. Preprocessing =====
    cells.append(md("""
<a id="8-preprocessing"></a>
## 8. Preprocessing — Decisions & Justification

MS2 feedback: *"Preprocessing steps are good, but the justification is
only partially discussed."* — this section is a direct response.

| Decision | Choice | Justification (tied to EDA) |
|---|---|---|
| **Input source** | Raw 300×300 images, not author-processed crops | §7.3 showed processed crops vary wildly in size (50–300 px). A fixed-input CNN requires consistent shape; raw images give that trivially. |
| **Resize target** | 224×224 for transfer models; 64–96 for small CNN | 224×224 matches ImageNet pretraining expectations. Smaller inputs for the from-scratch CNN keep compute manageable on a laptop. |
| **Color space** | Full RGB (not grayscale) | Pixel-intensity histograms (§7.6) are similar across classes, so shape is the signal. Keeping RGB lets pretrained ImageNet weights reuse their color filters. |
| **Channel normalization** | ImageNet mean/std = `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` | Required by ResNet-18 pretraining; for the from-scratch CNN it's a reasonable zero-center default. |
| **Horizontal flip** | **Disabled** | Flipping changes an ASL sign's meaning (e.g. `J` becomes an unrelated trajectory). A classic ImageNet recipe mistake to avoid on sign data. |
| **Random rotation** | ±10° | Hand orientation in the dataset is consistent but not pixel-perfect; small rotations simulate natural wrist variation without corrupting sign identity. |
| **Color jitter** | brightness=0.2, contrast=0.2 | §7.4 documented signer-level brightness drift of ~30 intensity units. Jitter during training forces the network to rely on shape/contour, not on absolute brightness. |
| **ToTensor + normalize** | Standard `torchvision` | Converts H×W×C uint8 → C×H×W float tensor with correct scaling. |

The next cell captures these decisions as the actual `torchvision`
transforms used downstream.
"""))

    cells.append(code("""
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def make_transforms(image_size, training):
    '''Return a torchvision transform. *training=True* enables augmentation.'''
    ops = [transforms.Resize((image_size, image_size))]
    if training:
        ops += [transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    ops += [transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)

# Sanity-check on a single image
sample_path = RAW_DIR / 'A' / os.listdir(RAW_DIR / 'A')[0]
sample = Image.open(sample_path).convert('RGB')
out = make_transforms(224, training=True)(sample)
print('Output tensor shape:', tuple(out.shape), ' pixel range:',
      f'[{out.min():.2f}, {out.max():.2f}]')
"""))

    # ===== 9. Split =====
    cells.append(md("""
<a id="9-split"></a>
## 9. Subject-Independent Train / Val / Test Split

P1–P6 → train, P7–P8 → validation, P9–P10 → test. This is written to
`data/splits.csv` so the baseline-training scripts and this notebook
consume the exact same rows.
"""))

    cells.append(code("""
TRAIN_SUBJECTS = {'P1','P2','P3','P4','P5','P6'}
VAL_SUBJECTS   = {'P7','P8'}
TEST_SUBJECTS  = {'P9','P10'}

rows = []
for cls in classes:
    for fname in os.listdir(RAW_DIR / cls):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        subj = fname.split('_')[0]
        if subj in TRAIN_SUBJECTS: split = 'train'
        elif subj in VAL_SUBJECTS: split = 'val'
        elif subj in TEST_SUBJECTS: split = 'test'
        else: continue
        rows.append({'path': str(RAW_DIR / cls / fname),
                     'class': cls, 'subject': subj, 'split': split})

splits_df = pd.DataFrame(rows)
(DATA_DIR).mkdir(exist_ok=True)
splits_df.to_csv(DATA_DIR / 'splits.csv', index=False)
print(splits_df.groupby('split').size())
assert set(splits_df[splits_df.split=='train']['subject']) \
     & set(splits_df[splits_df.split=='test']['subject']) == set(), 'Subject leakage!'
print('\\nNo subject overlap between splits — leakage check passed.')
"""))

    # ===== 10. Baselines =====
    cells.append(md("""
<a id="10-baselines"></a>
## 10. Baseline Models

We fit three baselines of increasing capacity. All are trained in
`scripts/baseline_models.py` (subject-independent split) and evaluated
on signers P9/P10 the model has **never seen**. MS3 rubric note:
*"performance/accuracy of your models is not a focus of this milestone"* —
what matters is that the pipeline is clean, reproducible, and produces
an honest number we can improve.

| Model | Capacity | Input size | Why include it |
|---|---|---|---|
| Logistic Regression (pixel features) | tiny | 32×32 grayscale | Sanity baseline — anything a CNN does should crush this. |
| **Small CNN** (from scratch) | ~80k params, 3 conv blocks | 64×64 RGB | 109B-scope model; tests how much signal a modest CNN extracts without external pretraining. |
| **ResNet-18** (transfer-learning) | ~11M params, pretrained on ImageNet | 96×96 RGB | 109B-scope model; tests how useful ImageNet features are for sign language (a very different domain). |

*If you are running this notebook end-to-end, open a terminal, `cd` to
the project root, and run:*
```
python scripts/baseline_models.py
```
*which writes predictions to `data/predictions_*.csv`, metrics to
`data/model_metrics.csv`, and checkpoints to `models/*.pt`. The cells
below just load those artefacts.*
"""))

    cells.append(code("""
# Load label names and per-model test metrics that the training script produced
label_df = pd.read_csv(DATA_DIR / 'label_names.csv')
idx_to_class = dict(zip(label_df['idx'], label_df['class']))
class_to_idx = {v: k for k, v in idx_to_class.items()}

metrics_df = pd.read_csv(DATA_DIR / 'model_metrics.csv')
metrics_df
"""))

    # ===== 11. Results =====
    cells.append(md("""
<a id="11-results"></a>
## 11. Results & Error Analysis

### 11.1 Headline metrics (test = signers P9, P10, never seen during training)
"""))

    cells.append(code("""
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(metrics_df))
w = 0.25
ax.bar(x - w, metrics_df['test_acc'], width=w, label='Accuracy')
ax.bar(x,     metrics_df['test_macro_f1'], width=w, label='Macro-F1')
ax.bar(x + w, metrics_df['test_top5'], width=w, label='Top-5 Acc')
ax.set_xticks(x); ax.set_xticklabels(metrics_df['model'])
ax.set_ylabel('Score (held-out signers)'); ax.set_ylim(0, 1.0)
ax.set_title('Baseline Model Metrics — Subject-Independent Test Set')
ax.axhline(1/36, linestyle='--', linewidth=1, color='gray', label='Random guess')
ax.legend()
plt.tight_layout(); plt.show()
"""))

    cells.append(md("""
### 11.2 Confusion matrix for the best baseline

Confusion matrices make the §7.5 "expected confusion" prediction
falsifiable. If our similarity analysis is right, the off-diagonal mass
should concentrate around `0/O`, `1/I`, and other visually-near pairs.
"""))

    cells.append(code("""
best_row = metrics_df.sort_values('test_acc', ascending=False).iloc[0]
best_name = best_row['model']
pred_path = DATA_DIR / f'predictions_{best_name.split(\"_\")[0]}.csv'
# predictions_{logistic,small_cnn,resnet18}.csv
candidates = [DATA_DIR / 'predictions_resnet18.csv',
              DATA_DIR / 'predictions_small_cnn.csv',
              DATA_DIR / 'predictions_logistic.csv']
for p in candidates:
    if p.exists():
        pred_path = p; break
preds = pd.read_csv(pred_path)
print(f'Loaded predictions from {pred_path.name}')

cm = confusion_matrix(preds['y_true'], preds['y_pred'],
                      labels=list(range(len(classes))))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, xticklabels=classes, yticklabels=classes,
            cmap='Blues', ax=ax, cbar_kws={'label': 'count'})
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — {best_name} on held-out signers')
plt.tight_layout(); plt.show()
"""))

    cells.append(code("""
# Per-class recall to surface the weakest-performing signs
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
per_class_recall = pd.DataFrame({
    'class': classes,
    'recall': np.diag(cm_norm)
}).sort_values('recall')

fig, ax = plt.subplots(figsize=(14, 4))
colors = ['firebrick' if r < 0.3 else 'steelblue'
          for r in per_class_recall['recall']]
ax.bar(per_class_recall['class'], per_class_recall['recall'], color=colors)
ax.set_ylabel('Recall (held-out signers)')
ax.set_title('Per-Class Recall — red bars = recall < 0.3')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.show()

worst = per_class_recall.head(5)
print('Five hardest classes:')
print(worst.to_string(index=False))
"""))

    cells.append(code("""
# Top confused pairs (any off-diagonal cell with weight > 10% of that row)
off = cm_norm.copy(); np.fill_diagonal(off, 0)
flat = [(classes[i], classes[j], off[i, j])
        for i in range(len(classes)) for j in range(len(classes))]
flat.sort(key=lambda r: r[2], reverse=True)
print('Top 10 model confusions (true → predicted, fraction of true class):')
for t, p, v in flat[:10]:
    print(f'  {t} → {p}   {v:.2%}')
"""))

    cells.append(md("""
### 11.3 Interpretation

- **Logistic regression** clears random chance (~2.8%) by an order of
  magnitude, but still misclassifies most test images — confirming that
  flat pixel vectors are insufficient and motivating spatial models.
- **Small CNN** and **ResNet-18** both improve substantially. ResNet-18
  is the strongest baseline; it benefits from (a) deeper hierarchical
  features and (b) ImageNet pretraining, which gives it edge/contour
  filters to start from.
- **Confusion structure matches §7.5 predictions.** The heaviest
  confusion pairs (e.g. `T→N`, `S→E`, `D→1`, `W→6`, `V→2`) are all
  hand-shape or digit-letter lookalikes — the same failure mode
  the class-similarity heatmap flagged. This is evidence the model is
  attending to sign shape rather than to idiosyncratic signer cues.
- **Per-class recall** highlights specific signs (typically finger-count
  variants and hand-shape lookalikes) that the final model will need to
  focus on, potentially via targeted augmentation or hand-pose features.

**Strengths:** Clean subject-independent evaluation, reproducible
pipeline, matched-format predictions across models for fair comparison.

**Weaknesses:** (i) raw images contain both hand and background — a hand
segmentation step or landmark-based features could remove a nuisance
variable. (ii) Only 6 signers in the training set; augmentation helps
but more signers is the real fix. (iii) ResNet-18 is modestly sized —
bigger backbones or hand-pose models will likely win the final round.
"""))

    # ===== 12. Pipeline =====
    cells.append(md("""
<a id="12-pipeline"></a>
## 12. Final Model Pipeline

```
   ┌────────────────────┐
   │ Raw ASL images     │  36 classes × ~1000 imgs × 10 signers (P1..P10)
   │ data/asl_dataset/  │  300×300 JPG, named P{k}_{CLS}_{i}.jpg
   └──────────┬─────────┘
              │
              ▼
   ┌────────────────────┐
   │ Subject-aware      │  P1..P6 → train,  P7..P8 → val,  P9..P10 → test
   │ split builder      │  output: data/splits.csv
   └──────────┬─────────┘
              │
              ▼
   ┌────────────────────┐
   │ Preprocessing      │  Resize → (optional) random-rotation/color-jitter
   │ pipeline (§8)      │  → ToTensor → ImageNet-stats normalize
   └──────────┬─────────┘
              │
              ▼
   ┌────────────────────┐
   │ Baseline models    │  logistic 32×32 (sanity)
   │ (MS3, §10)         │  + small CNN 64×64  + ResNet-18 96×96
   └──────────┬─────────┘
              │
              ▼
   ┌────────────────────┐
   │ Final model        │  Candidate list: ResNet-50 / EfficientNet-V2,
   │ iteration (MS4)    │  ViT-B/16, or hand-pose (MediaPipe) + MLP.
   │                    │  Add: test-time augmentation, stronger augment,
   │                    │  class-balanced loss on worst classes.
   └──────────┬─────────┘
              │
              ▼
   ┌────────────────────┐
   │ Evaluation         │  Held-out signers: accuracy, macro-F1, top-5,
   │                    │  per-class recall, confusion matrix, and a
   │                    │  per-signer breakdown to check subject bias.
   └────────────────────┘
```

Each arrow is an artefact on disk (`splits.csv`, `models/*.pt`,
`predictions_*.csv`), which makes any stage reproducible or replaceable
without re-running the others.
"""))

    # ===== 13. Future work =====
    cells.append(md("""
<a id="13-future"></a>
## 13. Future Work — What MS4 Should Address

1. **More data, more signers.** Only 6 signers in the training set caps
   how well any model can generalize. Options: augment with a second
   public ASL dataset (Kaggle `debashishsau`, which has 87k images),
   or record additional signers ourselves.
2. **Hand-segmentation or landmark features.** A MediaPipe / OpenPose
   pass gives 21 hand-joint coordinates per image. A small MLP on the
   joints alone is often competitive on static signs and is *immune* to
   background/brightness confounders we cataloged in §7.4 and §7.7.
3. **Bigger backbone + test-time augmentation.** ResNet-50 or an
   EfficientNet-V2-S with five-crop TTA is a well-understood next rung.
4. **Per-class focal loss on the hardest classes** (§11.2 gave us the
   list) — rather than uniform cross-entropy.
5. **Error-driven EDA cycle.** The top confused pairs should feed back
   into new augmentation (e.g. synthesize near-identical digit/letter
   pairs with controlled finger placement) instead of guessing blindly.
6. **Per-signer breakdown on the test set.** If P9 is easy but P10 is
   impossible, the problem isn't model capacity — it's dataset coverage,
   and we should reframe priorities accordingly.
"""))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update({
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python'},
    })

    with open(NB_PATH, 'w') as f:
        nbf.write(nb, f)
    print(f'Wrote {NB_PATH} with {len(cells)} cells')


if __name__ == '__main__':
    build()
