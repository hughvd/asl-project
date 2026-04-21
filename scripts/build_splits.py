"""
Build a subject-independent train / val / test split from the raw ASL images,
and save a CSV that downstream training scripts consume.

Subjects P1..P10. We hold P9 and P10 out for the test set, P7 and P8 for
validation, and train on P1..P6. This prevents subject-level leakage — the
MS2 EDA showed the processed-split train/test both contained all 10 subjects,
so a model could memorize a subject's hand/background instead of the sign.
"""
import os
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/asl_dataset")
OUT_CSV = Path("data/splits.csv")

TRAIN_SUBJECTS = {"P1", "P2", "P3", "P4", "P5", "P6"}
VAL_SUBJECTS = {"P7", "P8"}
TEST_SUBJECTS = {"P9", "P10"}


def build():
    classes = sorted(os.listdir(RAW_DIR))
    rows = []
    for cls in classes:
        cls_dir = RAW_DIR / cls
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            subj = fname.split("_")[0]
            if subj in TRAIN_SUBJECTS:
                split = "train"
            elif subj in VAL_SUBJECTS:
                split = "val"
            elif subj in TEST_SUBJECTS:
                split = "test"
            else:
                continue
            rows.append({"path": str(cls_dir / fname), "class": cls,
                         "subject": subj, "split": split})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Wrote {OUT_CSV} with {len(df)} rows")
    print(df.groupby("split").size())
    print("Classes:", len(classes))
    return df


if __name__ == "__main__":
    build()
