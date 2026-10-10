# -*- coding: utf-8 -*-
"""
quantize/ptq.py
---------------
Post-Training Quantization (INT8 PTQ) for MobileNetV3-Small.
Fuses Conv-BN-ReLU, attaches qconfig, runs calibration batches, converts to INT8,
saves the quantized checkpoint, and evaluates test accuracy.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torchvision.models.quantization.mobilenetv3 as mq
import torch.ao.quantization as tq
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import (
    get_patient_splits,
    ChestXRayDataset,
    get_transforms,
    TARGET_CLASSES,
)
from quantize.eval_quantized import evaluate_model, print_metrics

CSV_PATH  = ROOT / "dataset" / "Data_Entry_2017.csv"
IMAGE_DIR = ROOT / "dataset" / "images-224" / "images-224"
BASELINE_CKPT = ROOT / "train" / "checkpoints" / "baseline_fp32.pt"
OUTPUT_DIR = ROOT / "quantize" / "checkpoints"
OUTPUT_CKPT = OUTPUT_DIR / "int8_ptq.pt"


def get_backend() -> str:
    return "qnnpack"


def build_quantizable_model(ckpt_path: Path = BASELINE_CKPT) -> mq.QuantizableMobileNetV3:
    conf, last_ch = mq._mobilenet_v3_conf("mobilenet_v3_small")
    model = mq.QuantizableMobileNetV3(
        conf, last_ch,
        block=mq.QuantizableInvertedResidual,
        num_classes=len(TARGET_CLASSES),
    )
    mq._replace_relu(model)
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print(f"[INFO] Loaded FP32 baseline weights from: {ckpt_path}")
    else:
        raise FileNotFoundError(f"Baseline checkpoint not found at: {ckpt_path}")
    return model


def run_calibration(model: nn.Module, calib_loader: DataLoader, num_batches: int = 100):
    print(f"[INFO] Running PTQ calibration over {num_batches} batches...")
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= num_batches:
                break
            model(images)
            if (i + 1) % 20 == 0 or (i + 1) == num_batches:
                print(f"  Calibration step: {i+1}/{num_batches}")
    elapsed = time.time() - t0
    print(f"[INFO] Calibration completed in {elapsed:.1f}s")


def run_ptq(
    num_calib_batches: int = 100,
    batch_size: int = 32,
    save_ckpt: bool = True,
    evaluate: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = get_backend()
    print(f"[INFO] Using quantized backend: {backend}")
    if backend in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = backend

    # 1. Load model & weights
    model = build_quantizable_model(BASELINE_CKPT)
    model.eval()

    # 2. Fuse modules (Conv + BN + ReLU / Hardswish / SqueezeExcitation)
    print("[INFO] Fusing layers (Conv+BN+Act)...")
    model.fuse_model(is_qat=False)

    # 3. Assign qconfig & prepare
    model.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(model, inplace=True)

    # 4. Run calibration on validation data
    _, val_df, _ = get_patient_splits(CSV_PATH)
    val_ds = ChestXRayDataset(val_df, IMAGE_DIR, transform=get_transforms())
    calib_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    run_calibration(model, calib_loader, num_batches=num_calib_batches)

    # 5. Convert to quantized model
    print("[INFO] Converting model to INT8...")
    tq.convert(model, inplace=True)

    # 6. Save checkpoint
    if save_ckpt:
        torch.save(model.state_dict(), OUTPUT_CKPT)
        size_mb = os.path.getsize(OUTPUT_CKPT) / (1024 * 1024)
        print(f"[INFO] Saved INT8-PTQ checkpoint -> {OUTPUT_CKPT} ({size_mb:.2f} MB)")
    else:
        size_mb = 0.0

    # 7. Evaluate on held-out test split
    metrics = {}
    if evaluate:
        print("[INFO] Evaluating INT8-PTQ model on test set...")
        metrics = evaluate_model(model, device=torch.device("cpu"))
        print_metrics("INT8-PTQ", metrics, size_mb=size_mb)

    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run INT8 PTQ")
    parser.add_argument("--calib-batches", type=int, default=100, help="Number of calibration batches")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for calibration")
    args = parser.parse_args()

    run_ptq(num_calib_batches=args.calib_batches, batch_size=args.batch_size)
