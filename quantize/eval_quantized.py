# -*- coding: utf-8 -*-
"""
quantize/eval_quantized.py
--------------------------
Reusable evaluation helper for Phase 3 models (FP32, INT8-PTQ, INT8-QAT, INT4).
Computes macro & per-class AUROC, F1, Precision, and Recall on the held-out test split.
"""

import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
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

CSV_PATH  = ROOT / "dataset" / "Data_Entry_2017.csv"
IMAGE_DIR = ROOT / "dataset" / "images-224" / "images-224"


def get_test_loader(batch_size: int = 64, num_workers: int = 0) -> DataLoader:
    _, _, test_df = get_patient_splits(CSV_PATH)
    test_ds = ChestXRayDataset(test_df, IMAGE_DIR, transform=get_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return test_loader


def evaluate_model(model: nn.Module, device: torch.device, loader: DataLoader = None) -> Dict[str, Any]:
    """
    Evaluates model on the test dataset.
    Args:
        model: PyTorch model (can be FP32, INT8 quantised, or INT4 fake-quant).
        device: Device to run on (e.g., 'cpu' or 'cuda').
        loader: Optional test DataLoader. If None, created automatically.
    Returns:
        dict with macro_auroc, macro_f1, macro_precision, macro_recall, and per_class_auroc.
    """
    if loader is None:
        loader = get_test_loader()

    model.eval()
    num_classes = len(TARGET_CLASSES)

    auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes).to(device)
    auroc_per_class = torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average=None).to(device)
    f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, threshold=0.5).to(device)
    precision = torchmetrics.Precision(task="multilabel", num_labels=num_classes, threshold=0.5).to(device)
    recall = torchmetrics.Recall(task="multilabel", num_labels=num_classes, threshold=0.5).to(device)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            auroc.update(probs, labels.int())
            auroc_per_class.update(probs, labels.int())
            f1.update(probs, labels.int())
            precision.update(probs, labels.int())
            recall.update(probs, labels.int())

    macro_auc_val = auroc.compute().item()
    per_class_val = auroc_per_class.compute()
    macro_f1_val = f1.compute().item()
    macro_prec_val = precision.compute().item()
    macro_rec_val = recall.compute().item()

    results = {
        "macro_auroc": macro_auc_val,
        "macro_f1": macro_f1_val,
        "macro_precision": macro_prec_val,
        "macro_recall": macro_rec_val,
        "per_class_auroc": {
            cls_name: per_class_val[i].item()
            for i, cls_name in enumerate(TARGET_CLASSES)
        }
    }

    return results


def print_metrics(variant_name: str, metrics: Dict[str, Any], size_mb: float = None):
    print("\n" + "=" * 50)
    size_str = f" ({size_mb:.2f} MB)" if size_mb is not None else ""
    print(f"       RESULTS: {variant_name}{size_str}")
    print("=" * 50)
    print(f"  Macro ROC-AUC  : {metrics['macro_auroc']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
    print("-" * 50)
    print("  Per-Class ROC-AUC:")
    for cls_name, val in metrics["per_class_auroc"].items():
        print(f"    - {cls_name:<15}: {val:.4f}")
    print("=" * 50 + "\n")
