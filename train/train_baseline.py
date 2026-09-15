# -*- coding: utf-8 -*-
"""
train/train_baseline.py
-----------------------
Phase 2 — Baseline FP32 training.

Model   : MobileNetV3-Small (ImageNet pre-trained), head replaced for 3-class multi-label.
Loss    : BCEWithLogitsLoss
Optim   : Adam, lr=1e-4
Schedule: ReduceLROnPlateau (val AUC, patience=3)
Stopping: Early stopping on mean val AUC (patience=7)
Epochs  : up to 30
Metrics : torchmetrics.AUROC (multilabel), F1, Precision, Recall

Outputs
-------
train/checkpoints/baseline_fp32.pt  – best model state dict
train/metrics.json                  – per-class + macro AUC/F1/Precision/Recall
"""

import sys
import json
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader
import torchmetrics

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import (
    get_patient_splits,
    ChestXRayDataset,
    get_transforms,
    TARGET_CLASSES,
)

# ── paths ────────────────────────────────────────────────────────────────────
CSV_PATH  = ROOT / "dataset" / "Data_Entry_2017.csv"
IMAGE_DIR = ROOT / "dataset" / "images-224" / "images-224"
CKPT_DIR  = ROOT / "train" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT   = CKPT_DIR / "baseline_fp32.pt"
METRICS_OUT = ROOT / "train" / "metrics.json"

# ── hyper-parameters ─────────────────────────────────────────────────────────
NUM_CLASSES      = len(TARGET_CLASSES)   # 3
BATCH_SIZE       = 64
NUM_WORKERS      = 0     # 0 avoids Windows multiprocessing spawn overhead
LR               = 1e-4
MAX_EPOCHS       = 30
PATIENCE         = 7     # early stopping patience (epochs without val AUC improvement)
LR_PATIENCE      = 3     # ReduceLROnPlateau patience
SEED             = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ── augmented train transform ────────────────────────────────────────────────
import torchvision.transforms as T

def get_train_transforms():
    """
    Light augmentation on top of the standard ImageNet normalisation:
    random horizontal flip + random ±10° rotation to reduce overfitting.
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ── model ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Replace the final linear layer (classifier[-1])
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ── metric helpers ───────────────────────────────────────────────────────────
def build_metric_collection(num_labels: int, device: torch.device) -> torchmetrics.MetricCollection:
    return torchmetrics.MetricCollection({
        "auroc":     torchmetrics.AUROC(task="multilabel",     num_labels=num_labels),
        "f1":        torchmetrics.F1Score(task="multilabel",   num_labels=num_labels, threshold=0.5),
        "precision": torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=0.5),
        "recall":    torchmetrics.Recall(task="multilabel",    num_labels=num_labels, threshold=0.5),
    }).to(device)


def per_class_auroc(num_labels: int, device: torch.device) -> torchmetrics.AUROC:
    """Returns an AUROC metric that computes per-class scores (average=None)."""
    return torchmetrics.AUROC(
        task="multilabel", num_labels=num_labels, average=None
    ).to(device)


# ── training loop ────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, metrics, device, train: bool):
    model.train() if train else model.eval()
    metrics.reset()

    running_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)           # float32, shape [B, C]

            logits = model(images)               # [B, C]
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            metrics.update(probs, labels.int())

            running_loss += loss.item()
            n_batches    += 1

    epoch_metrics = metrics.compute()
    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss, epoch_metrics


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Target classes: {TARGET_CLASSES}")

    # ── data ─────────────────────────────────────────────────────────────────
    print("[INFO] Building patient-level splits …")
    train_df, val_df, test_df = get_patient_splits(CSV_PATH)
    print(f"       Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

    train_ds = ChestXRayDataset(train_df, IMAGE_DIR, transform=get_train_transforms())
    val_ds   = ChestXRayDataset(val_df,   IMAGE_DIR, transform=get_transforms())
    test_ds  = ChestXRayDataset(test_df,  IMAGE_DIR, transform=get_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    # ── model / loss / optimiser ─────────────────────────────────────────────
    model     = build_model(NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE
    )

    # ── metrics ──────────────────────────────────────────────────────────────
    train_metrics = build_metric_collection(NUM_CLASSES, device)
    val_metrics   = build_metric_collection(NUM_CLASSES, device)

    # ── training ─────────────────────────────────────────────────────────────
    best_val_auc  = -1.0
    best_state    = None
    no_improve    = 0
    history       = []

    print(f"{'Epoch':>5}  {'TrLoss':>8}  {'TrAUC':>7}  {'VaLoss':>8}  {'VaAUC':>7}  {'LR':>9}  {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer,
                                   train_metrics, device, train=True)
        va_loss, va_m = run_epoch(model, val_loader,   criterion, None,
                                   val_metrics,   device, train=False)

        tr_auc = tr_m["auroc"].item()
        va_auc = va_m["auroc"].item()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_auc:>7.4f}  {va_loss:>8.4f}  {va_auc:>7.4f}  {lr_now:>9.2e}  {elapsed:>5.1f}s")

        history.append({
            "epoch": epoch, "tr_loss": tr_loss, "tr_auc": tr_auc,
            "va_loss": va_loss, "va_auc": va_auc, "lr": lr_now,
        })

        scheduler.step(va_auc)

        if va_auc > best_val_auc + 1e-5:
            best_val_auc = va_auc
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, BEST_CKPT)
            print(f"         [OK] New best val AUC={best_val_auc:.4f} -> saved checkpoint")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[INFO] Early stopping triggered (no improvement for {PATIENCE} epochs).")
                break

    # ── load best weights for evaluation ─────────────────────────────────────
    print("\n[INFO] Loading best checkpoint for final evaluation …")
    model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
    model.eval()

    # ── final evaluation on val & test ───────────────────────────────────────
    def evaluate_split(loader, split_name: str) -> dict:
        macro_m = build_metric_collection(NUM_CLASSES, device)
        pc_auroc = per_class_auroc(NUM_CLASSES, device)
        macro_m.reset()
        pc_auroc.reset()

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                probs  = torch.sigmoid(model(images))
                macro_m.update(probs, labels.int())
                pc_auroc.update(probs, labels.int())

        macro  = macro_m.compute()
        per_cl = pc_auroc.compute()          # shape [NUM_CLASSES]

        result = {
            "macro_auroc":     macro["auroc"].item(),
            "macro_f1":        macro["f1"].item(),
            "macro_precision": macro["precision"].item(),
            "macro_recall":    macro["recall"].item(),
            "per_class_auroc": {
                cls: per_cl[i].item() for i, cls in enumerate(TARGET_CLASSES)
            },
        }

        print(f"\n[{split_name}]")
        print(f"  Macro AUC       : {result['macro_auroc']:.4f}")
        print(f"  Macro F1        : {result['macro_f1']:.4f}")
        print(f"  Macro Precision : {result['macro_precision']:.4f}")
        print(f"  Macro Recall    : {result['macro_recall']:.4f}")
        print("  Per-class AUC:")
        for cls, auc in result["per_class_auroc"].items():
            print(f"    {cls:<15}: {auc:.4f}")

        return result

    val_results  = evaluate_split(val_loader,  "Validation")
    test_results = evaluate_split(test_loader, "Test")

    # ── save metrics.json ─────────────────────────────────────────────────────
    metrics_payload = {
        "model":          "mobilenet_v3_small",
        "precision":      "fp32",
        "target_classes": TARGET_CLASSES,
        "best_val_auroc": best_val_auc,
        "val":            val_results,
        "test":           test_results,
        "training_history": history,
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n[INFO] metrics.json saved → {METRICS_OUT}")
    print(f"[INFO] Best checkpoint  → {BEST_CKPT}")
    print("\n[DONE] Phase 2 complete.")


if __name__ == "__main__":
    main()
