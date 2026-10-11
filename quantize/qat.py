# -*- coding: utf-8 -*-
"""
quantize/qat.py
---------------
Quantization-Aware Training (INT8 QAT) for MobileNetV3-Small.
Fuses layers with is_qat=True, prepares QAT fake-quantization nodes,
fine-tunes for a few epochs at lr=1e-5 on GPU, converts to INT8,
saves the quantized checkpoint, and evaluates test accuracy.
"""

import sys
import os
import time
import copy
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torchvision.models.quantization.mobilenetv3 as mq
import torch.ao.quantization as tq
from torch.utils.data import DataLoader
import torchmetrics

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import (
    get_patient_splits,
    ChestXRayDataset,
    get_transforms,
    TARGET_CLASSES,
)
from train.train_baseline import get_train_transforms
from quantize.eval_quantized import evaluate_model, print_metrics
from quantize.ptq import get_backend, build_quantizable_model

CSV_PATH  = ROOT / "dataset" / "Data_Entry_2017.csv"
IMAGE_DIR = ROOT / "dataset" / "images-224" / "images-224"
BASELINE_CKPT = ROOT / "train" / "checkpoints" / "baseline_fp32.pt"
OUTPUT_DIR = ROOT / "quantize" / "checkpoints"
OUTPUT_CKPT = OUTPUT_DIR / "int8_qat.pt"


def run_qat(
    epochs: int = 7,
    lr: float = 1e-5,
    batch_size: int = 64,
    save_ckpt: bool = True,
    evaluate: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = get_backend()
    print(f"[INFO] Using quantized backend: {backend}")
    if backend in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = backend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device for QAT fine-tuning: {device}")

    # 1. Build and load baseline
    model = build_quantizable_model(BASELINE_CKPT)
    model.train()

    # 2. Fuse modules for QAT
    print("[INFO] Fusing layers for QAT...")
    model.fuse_model(is_qat=True)

    # 3. Assign QAT qconfig and prepare
    model.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(model, inplace=True)
    model.to(device)

    # 4. Data loaders
    print("[INFO] Preparing data loaders for QAT fine-tuning...")
    train_df, val_df, _ = get_patient_splits(CSV_PATH)
    train_ds = ChestXRayDataset(train_df, IMAGE_DIR, transform=get_train_transforms())
    val_ds   = ChestXRayDataset(val_df,   IMAGE_DIR, transform=get_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # 5. Optimizer & Criterion
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_metric = torchmetrics.AUROC(task="multilabel", num_labels=len(TARGET_CLASSES)).to(device)

    best_val_auc = -1.0
    best_weights = None

    print(f"\n[INFO] Starting QAT fine-tuning for {epochs} epochs (lr={lr:.1e})...")
    print(f"{'Epoch':>5}  {'Train Loss':>11}  {'Val AUC':>9}  {'Time':>7}")
    print("-" * 40)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)

        # Validation step
        model.eval()
        val_metric.reset()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                probs = torch.sigmoid(model(images))
                val_metric.update(probs, labels.int())

        val_auc = val_metric.compute().item()
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {avg_loss:>11.4f}  {val_auc:>9.4f}  {elapsed:>6.1f}s")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())

    # 6. Load best fine-tuned state and convert to quantized INT8 model
    print(f"\n[INFO] Best QAT Validation AUC: {best_val_auc:.4f}")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    print("[INFO] Converting QAT model to INT8 on CPU...")
    model.to("cpu")
    model.eval()
    tq.convert(model, inplace=True)

    # 7. Save checkpoint
    if save_ckpt:
        torch.save(model.state_dict(), OUTPUT_CKPT)
        size_mb = os.path.getsize(OUTPUT_CKPT) / (1024 * 1024)
        print(f"[INFO] Saved INT8-QAT checkpoint -> {OUTPUT_CKPT} ({size_mb:.2f} MB)")
    else:
        size_mb = 0.0

    # 8. Evaluate on held-out test split
    metrics = {}
    if evaluate:
        print("[INFO] Evaluating INT8-QAT model on test set...")
        metrics = evaluate_model(model, device=torch.device("cpu"))
        print_metrics("INT8-QAT", metrics, size_mb=size_mb)

    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run INT8 QAT")
    parser.add_argument("--epochs", type=int, default=7, help="Number of QAT fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    run_qat(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
