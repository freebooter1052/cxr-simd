# -*- coding: utf-8 -*-
"""
quantize/export_onnx.py
-----------------------
Exports all 4 model variants (FP32, INT8-PTQ, INT8-QAT, INT4) to ONNX format
using opset_version=13 as specified in the Phase 3 requirements.
Verifies each exported model with onnx.checker.
"""

import sys
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import onnx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.train_baseline import build_model
from data.dataset import TARGET_CLASSES
from quantize.ptq import build_quantizable_model, get_backend
from quantize.int4 import load_int4_model
import torchvision.models.quantization.mobilenetv3 as mq
import torch.ao.quantization as tq

ONNX_DIR = ROOT / "quantize" / "onnx"
CKPT_DIR = ROOT / "quantize" / "checkpoints"
BASELINE_CKPT = ROOT / "train" / "checkpoints" / "baseline_fp32.pt"


def export_fp32_onnx(output_path: Optional[Path] = None, dummy_input: Optional[torch.Tensor] = None) -> Path:
    if output_path is None:
        output_path = ONNX_DIR / "fp32.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dummy_input is None:
        dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[INFO] Exporting FP32 model to ONNX: {output_path}")
    model = build_model(num_classes=len(TARGET_CLASSES))
    model.load_state_dict(torch.load(BASELINE_CKPT, map_location="cpu"))
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=13,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Validated FP32 ONNX model ({size_mb:.2f} MB)")
    return output_path


def export_int8_onnx(
    model: nn.Module,
    variant_name: str,
    output_path: Optional[Path] = None,
    dummy_input: Optional[torch.Tensor] = None,
) -> Path:
    if output_path is None:
        output_path = ONNX_DIR / f"{variant_name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dummy_input is None:
        dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[INFO] Exporting {variant_name} to ONNX: {output_path}")
    model.eval()
    model.to("cpu")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=13,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Validated {variant_name} ONNX model ({size_mb:.2f} MB)")
    return output_path


def export_int4_onnx(output_path: Optional[Path] = None, dummy_input: Optional[torch.Tensor] = None) -> Path:
    if output_path is None:
        output_path = ONNX_DIR / "int4.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dummy_input is None:
        dummy_input = torch.randn(1, 3, 224, 224)

    int4_ckpt = CKPT_DIR / "int4.pt"
    if not int4_ckpt.exists():
        raise FileNotFoundError(f"INT4 checkpoint not found at: {int4_ckpt}")

    print(f"[INFO] Exporting INT4 model to ONNX: {output_path}")
    model = load_int4_model(int4_ckpt, device="cpu")
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=13,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Validated INT4 ONNX model ({size_mb:.2f} MB)")
    return output_path


def export_all_onnx(
    ptq_model: Optional[nn.Module] = None,
    qat_model: Optional[nn.Module] = None,
):
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)

    print("\n" + "=" * 50)
    print("       EXPORTING ALL VARIANTS TO ONNX")
    print("=" * 50)

    # 1. FP32
    export_fp32_onnx(dummy_input=dummy)

    # 2. INT8-PTQ
    if ptq_model is not None:
        export_int8_onnx(ptq_model, "int8_ptq", dummy_input=dummy)
    else:
        print("[SKIP] PTQ model instance not provided for ONNX export.")

    # 3. INT8-QAT
    if qat_model is not None:
        export_int8_onnx(qat_model, "int8_qat", dummy_input=dummy)
    else:
        print("[SKIP] QAT model instance not provided for ONNX export.")

    # 4. INT4
    int4_ckpt = CKPT_DIR / "int4.pt"
    if int4_ckpt.exists():
        export_int4_onnx(dummy_input=dummy)
    else:
        print(f"[SKIP] INT4 checkpoint {int4_ckpt} not found.")


if __name__ == "__main__":
    export_all_onnx()
