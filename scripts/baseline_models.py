"""
Train three baseline models on the subject-independent split and save
metrics + predictions for later analysis.

Models:
  1. Logistic regression on raw-pixel features (sanity baseline).
  2. Small CNN trained from scratch.
  3. ResNet-18 fine-tuned from ImageNet (transfer learning).

All three are evaluated on the held-out subjects (P9, P10). Predictions are
saved to data/predictions_<model>.csv, class labels to data/label_names.csv,
and a summary row per model to data/model_metrics.csv.

Dataset is small enough that we train at 64x64 (logistic / small CNN) and
96x96 (ResNet-18) to keep runtime manageable on CPU/MPS. MS3 rubric
explicitly says performance is not the focus at this stage; we want a clean,
reproducible baseline that later iterations can beat.
"""
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

DATA_CSV = Path("data/splits.csv")
OUT_DIR = Path("data")
CKPT_DIR = Path("models")
CKPT_DIR.mkdir(exist_ok=True)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----- data -----
class AslDataset(Dataset):
    """Loads ASL images from a dataframe row and applies torchvision transforms."""

    def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[row["class"]]


def load_splits():
    df = pd.read_csv(DATA_CSV)
    classes = sorted(df["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    splits = {s: df[df["split"] == s].reset_index(drop=True)
              for s in ["train", "val", "test"]}
    return splits, classes, class_to_idx


# ----- logistic regression baseline -----
def to_flat_array(df, size=32):
    """Loads every image in df, resizes to size x size grayscale, returns (N, size*size)."""
    X = np.zeros((len(df), size * size), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)
    for i, row in enumerate(df.itertuples()):
        img = Image.open(row.path).convert("L").resize((size, size))
        X[i] = np.asarray(img, dtype=np.float32).flatten() / 255.0
        y[i] = row.class_idx
    return X, y


def run_logistic(splits, class_to_idx, size=32):
    """Fit multinomial logistic regression on flat 32x32 grayscale pixels."""
    for s in splits:
        splits[s] = splits[s].assign(class_idx=splits[s]["class"].map(class_to_idx))

    t0 = time.time()
    X_tr, y_tr = to_flat_array(splits["train"], size)
    X_te, y_te = to_flat_array(splits["test"], size)
    print(f"Logistic features loaded in {time.time() - t0:.1f}s — "
          f"X_tr {X_tr.shape}, X_te {X_te.shape}")

    clf = LogisticRegression(max_iter=500, n_jobs=-1, solver="lbfgs",
                             multi_class="multinomial")
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    print(f"Logistic fit in {time.time() - t0:.1f}s")

    probs = clf.predict_proba(X_te)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="macro")
    top5 = top_k_accuracy_score(y_te, probs, k=5,
                                labels=list(range(len(class_to_idx))))
    print(f"Logistic — acc {acc:.3f}, macro-F1 {f1:.3f}, top5 {top5:.3f}")

    pd.DataFrame({"y_true": y_te, "y_pred": preds}).to_csv(
        OUT_DIR / "predictions_logistic.csv", index=False)
    return {"model": "logistic_32x32", "test_acc": acc, "test_macro_f1": f1,
            "test_top5": top5}


# ----- CNN utilities -----
def make_loaders(splits, class_to_idx, size, batch=128, aug=True):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]) if aug else transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tr = DataLoader(AslDataset(splits["train"], class_to_idx, train_tf),
                    batch_size=batch, shuffle=True, num_workers=0)
    va = DataLoader(AslDataset(splits["val"], class_to_idx, eval_tf),
                    batch_size=batch, shuffle=False, num_workers=0)
    te = DataLoader(AslDataset(splits["test"], class_to_idx, eval_tf),
                    batch_size=batch, shuffle=False, num_workers=0)
    return tr, va, te


class SmallCNN(nn.Module):
    """Three-block conv net — MS3-appropriate baseline, trained from scratch."""

    def __init__(self, n_classes=36):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def train_model(model, train_loader, val_loader, epochs, lr, tag):
    """Standard train loop with Adam, cross-entropy, validation after every epoch."""
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    history = []
    best_val = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)
        tr_loss, tr_acc = running / seen, correct / seen

        model.eval()
        correct = seen = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                seen += x.size(0)
        val_acc = correct / seen
        dt = time.time() - t0
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_acc": val_acc, "time_s": dt})
        print(f"[{tag}] ep {ep:2d} loss {tr_loss:.3f} "
              f"tr_acc {tr_acc:.3f} val_acc {val_acc:.3f} ({dt:.1f}s)")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CKPT_DIR / f"{tag}.pt")
    return history, best_val


@torch.no_grad()
def eval_on_test(model, loader, class_to_idx):
    model.eval()
    all_pred = []
    all_true = []
    all_top5 = []
    n_classes = len(class_to_idx)
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x).cpu()
        probs = logits.softmax(dim=1).numpy()
        all_pred.append(probs.argmax(1))
        all_true.append(y.numpy())
        top5 = probs.argsort(axis=1)[:, -5:]
        all_top5.append(top5)
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    top5 = np.concatenate(all_top5)
    acc = (y_pred == y_true).mean()
    f1 = f1_score(y_true, y_pred, average="macro")
    t5 = np.mean([t in top for t, top in zip(y_true, top5)])
    return y_true, y_pred, {"test_acc": float(acc), "test_macro_f1": float(f1),
                            "test_top5": float(t5)}


# ----- runners -----
def run_small_cnn(splits, class_to_idx, classes, epochs=8):
    tr, va, te = make_loaders(splits, class_to_idx, size=64, batch=128)
    model = SmallCNN(n_classes=len(classes))
    history, _ = train_model(model, tr, va, epochs=epochs, lr=1e-3, tag="small_cnn")
    model.load_state_dict(torch.load(CKPT_DIR / "small_cnn.pt"))
    y_true, y_pred, m = eval_on_test(model, te, class_to_idx)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        OUT_DIR / "predictions_small_cnn.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "history_small_cnn.csv", index=False)
    m["model"] = "small_cnn_64"
    print(f"Small CNN — {m}")
    return m


def run_resnet18(splits, class_to_idx, classes, epochs=5):
    tr, va, te = make_loaders(splits, class_to_idx, size=96, batch=64)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    history, _ = train_model(model, tr, va, epochs=epochs, lr=1e-3, tag="resnet18")
    model.load_state_dict(torch.load(CKPT_DIR / "resnet18.pt"))
    y_true, y_pred, m = eval_on_test(model, te, class_to_idx)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        OUT_DIR / "predictions_resnet18.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "history_resnet18.csv", index=False)
    m["model"] = "resnet18_96"
    print(f"ResNet-18 — {m}")
    return m


def main():
    splits, classes, class_to_idx = load_splits()
    pd.DataFrame({"class": classes, "idx": range(len(classes))}).to_csv(
        OUT_DIR / "label_names.csv", index=False)

    metrics = []
    metrics.append(run_logistic(splits, class_to_idx))
    metrics.append(run_small_cnn(splits, class_to_idx, classes))
    metrics.append(run_resnet18(splits, class_to_idx, classes))
    pd.DataFrame(metrics).to_csv(OUT_DIR / "model_metrics.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
