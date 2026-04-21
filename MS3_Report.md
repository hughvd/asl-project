# MS3 — Work Report

**Project:** ASL Sign-Language Recognition (Canvas #11)
**Group:** Hugh Van Deventer, Ayush Sharma, ZD
**Date:** 2026-04-19

This report summarizes what was done to address the MS2 TF feedback and to
complete the MS3 rubric in the single notebook
`ASL Data Loading & Cleaning.ipynb`. All code is reproducible: run
`python scripts/build_splits.py`, then `python scripts/baseline_models.py`,
then open the notebook and Run-All. It executes end-to-end with zero errors
on Apple-silicon MPS, a CUDA GPU, or CPU.

---

## 1. How the MS2 feedback was addressed

| MS2 item | Original grade | What we changed |
|---|---|---|
| **Problem definition — research question missing from notebook** | 0.75 | Notebook now opens with a bolded research question (§1) covering both the classification task *and* the subject-generalization constraint. The TOC labels it explicitly. |
| **Data access & provenance — source/link not specified** | 0.5 | New §3 "Data Description, Source, and Access" gives the Kaggle dataset name, URL, license, signer metadata, filename convention, and on-disk size. §5 has an `ensure_unzipped()` helper that makes the notebook self-contained starting from the two archives in `./data/`. |
| **Preprocessing justification — only partial** | 0.5 | §8 is a table with one row per decision and a *why* column tied to specific EDA findings (e.g. "horizontal flip disabled because flipping changes sign identity"; "color jitter chosen because §7.4 documented 30-unit per-signer brightness drift"). The transforms are then implemented in a single `make_transforms()` helper. |
| **Analysis — patterns/relationships/confounders limited** | 0.3 | §7 is the biggest expansion. It now contains seven sub-sections: samples, class distribution, dimensions, **signer-level brightness/contrast with the leakage check (§7.4)**, class cosine-similarity heatmap with ranked pairs (§7.5), per-class pixel histograms (§7.6), and a new **background-artifact analysis** (§7.7). Each subsection ends with a "Takeaway" line that ties the finding to a modelling decision. |
| **Visualization quality — no interpretation** | 0.75 | Every EDA subsection now has a bold-marked Takeaway. Every model-results plot in §11 is paired with an interpretation paragraph, and the confusion matrix is paired with a ranked per-class recall bar-chart and a top-10-confusions table that make the failure modes human-readable. |
| **Insights → decisions linkage not explicit** | 0.75 | The §8 preprocessing table explicitly cites the EDA section that justifies each choice. §9 is framed as the direct fix for the §7.4 leakage finding. §13 "Future Work" is written as a list of decisions motivated by *specific* results (e.g. hand-landmark features motivated by §7.4/§7.7 confounder findings; per-signer test breakdown motivated by the per-class recall pattern). |
| Problem-statement rescoping — already 1.0 | 1.0 | Kept but made the new subject-generalization constraint the load-bearing framing. |

## 2. How the MS3 rubric items were addressed

1. **Problem Statement Refinement & Introduction.** §1 gives the
   refined research question; §2 is the one-page MS2 recap plus a
   bullet list of what MS3 adds.
2. **Comprehensive EDA Review.** §7 extends MS2 with confounder
   analysis (§7.4), class-similarity heatmap (§7.5), pixel-intensity
   histograms (§7.6), and a new background-artifact analysis (§7.7).
3. **Baseline Model Selection & Justification.** §10 introduces three
   models in increasing capacity and explains each choice. Two of the
   three (small CNN, ResNet-18) are 109B-scope deep-learning models.
   Training uses the subject-independent split; code lives in
   `scripts/baseline_models.py` for reproducibility.
4. **Results Interpretation & Analysis.** §11 has (a) headline metrics
   bar chart, (b) confusion matrix on the best baseline, (c) ranked
   per-class recall chart with red bars for classes <30% recall,
   (d) top-10 confused pairs table, and (e) a prose interpretation
   including strengths and weaknesses.
5. **Final Model Pipeline.** §12 is an ASCII diagram tracing raw
   images → split builder → preprocessing → baselines → final-model
   iteration → evaluation, with each arrow representing a persisted
   on-disk artefact so stages are swappable.
6. **Notebook hygiene (table of contents, code comments, docstrings,
   reproducibility).** The notebook has a top-of-file TOC with anchor
   links; each function has a docstring; all paths are relative to the
   project root; running the notebook end-to-end after the training
   script writes a fully-rendered deliverable.

## 3. What was built

### New files in the project

```
scripts/
├── build_splits.py        # writes data/splits.csv (subject-independent)
├── baseline_models.py     # trains logistic / small CNN / ResNet-18
├── build_notebook.py      # regenerates the notebook programmatically
└── quick_bench.py         # data-loader sanity bench
```

Auxiliary artefacts written to `data/` by the training script:

```
data/
├── splits.csv                    # 36,000 rows; train=21.6k, val=7.2k, test=7.2k
├── label_names.csv               # class-index ↔ class-label mapping
├── model_metrics.csv             # accuracy / macro-F1 / top-5 per model
├── predictions_logistic.csv      # y_true, y_pred on test set
├── predictions_small_cnn.csv
├── predictions_resnet18.csv
├── history_small_cnn.csv         # per-epoch train-loss/val-acc
└── history_resnet18.csv
```

Model checkpoints in `models/small_cnn.pt` and `models/resnet18.pt`.

### Refactored notebook

The old notebook was Colab-specific (drive mount, absolute paths under
`/content/drive/My Drive/…`). The new notebook is **local-path only**
and runs on any laptop with the two zip archives in `./data/`. It has
55 cells (34 markdown, 21 code) organized under 13 TOC sections.

## 4. Results

All three baselines are evaluated on **signers P9 and P10, held out of
training entirely**. Metrics are on the 7,200-image held-out test set.

| Model | Test Accuracy | Macro-F1 | Top-5 Accuracy |
|---|---|---|---|
| Random-guess reference | 0.028 (1/36) | — | 0.139 (5/36) |
| Logistic regression, 32×32 grey | **0.161** | 0.157 | 0.432 |
| Small CNN (3 conv blocks, 64×64) | **0.326** | 0.314 | 0.828 |
| **ResNet-18 ImageNet transfer (96×96)** | **0.733** | **0.721** | **0.967** |

Key observations (all discussed in notebook §11):

- The progression across the three models is monotonic and large,
  which is the sanity check we wanted: a flat-pixel linear model barely
  works, a modest CNN helps, and an ImageNet-pretrained backbone
  transfers well even at a small 96×96 input.
- Top-5 accuracy of 96.7% for ResNet-18 means the correct sign is
  almost always in the model's top guesses — the remaining error is
  concentrated in a small number of very-similar pairs, not spread
  uniformly.
- The top confusions the model makes (`T→N`, `D→1`, `X→Q`, `6→4`,
  `I→S`, …) are all **hand-shape lookalikes or digit/letter
  collisions**, matching the failure modes the §7.5 cosine-similarity
  heatmap flagged. This is evidence that the model is attending to
  sign shape rather than to a signer-specific confound.
- The weakest classes are `W` (0% recall), `S` (7.5%), and `T` (9.5%)
  — all hand configurations that differ from another class only in
  finger placement. §13 proposes targeted fixes for these (class-focal
  loss, more augmentation, or switching to landmark features).

## 5. How to run it

```bash
# From the project root
python3 scripts/build_splits.py         # ~1 s, writes data/splits.csv
python3 scripts/baseline_models.py      # ~10 min on MPS / CUDA, ~30 min CPU
jupyter notebook "ASL Data Loading & Cleaning.ipynb"   # Cell → Run All
```

The training script is idempotent — delete `data/model_metrics.csv` to
force re-training. The notebook does not retrain; it only loads the
CSV and checkpoint artefacts, so it renders in under a minute.

## 6. Known caveats and what we'd do next

1. **Only 6 signers in the training set** caps generalization — the
   top-priority MS4 fix is adding more signers (augment from the
   Kaggle `debashishsau` dataset or record our own).
2. **ResNet-18 is a modest backbone** — moving to ResNet-50 or
   EfficientNet-V2-S with test-time augmentation is the straight-line
   improvement.
3. **Background and lighting are still weak confounders** — a
   MediaPipe hand-landmark pass would remove them entirely and often
   matches or beats CNNs on static signs.
4. **Per-signer test breakdown is missing** — if P10 is the entire
   source of error it reframes the problem as dataset coverage, not
   model capacity. This is a one-page addition for MS4.
5. **Augmentation is conservative** — class-conditional augmentation
   targeted at the hardest confusion pairs (e.g. synthesize near-
   identical `T`/`S` examples with controlled thumb position) should
   help more than uniform jitter.

---

### TL;DR

- MS2 feedback → every item has a concrete diff in the notebook.
- MS3 rubric → every item has a dedicated labeled section.
- Three baselines (one non-neural sanity check + two 109B-scope
  models) trained on a subject-independent split with 16% → 33% →
  73% accuracy on held-out signers.
- Everything is reproducible from the project root with two python
  commands and a Run-All in the notebook.
