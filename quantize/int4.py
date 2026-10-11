# -*- coding: utf-8 -*-
"""
quantize/int4.py
----------------
INT4 Quantization for MobileNetV3-Small using Straight-Through Estimator (STE)
fake-quantization during fine-tuning, and real INT4 weight bit-packing at export.

Features:
- Symmetric per-channel 4-bit quantization (range [-8, 7]).
- STE fake-quantization during training.
- 4-bit packing (2 values per uint8 byte) for true INT4 disk & memory footprint.
- Standalone runner and evaluation.
"""

import sys
import os
import time
import copy
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from train.train_baseline import build_model, get_train_transforms
from quantize.eval_quantized import evaluate_model, print_metrics

CSV_PATH  = ROOT / "dataset" / "Data_Entry_2017.csv"
IMAGE_DIR = ROOT / "dataset" / "images-224" / "images-224"
BASELINE_CKPT = ROOT / "train" / "checkpoints" / "baseline_fp32.pt"
OUTPUT_DIR = ROOT / "quantize" / "checkpoints"
OUTPUT_CKPT = OUTPUT_DIR / "int4.pt"


def pack_int4_tensor(w_int4: torch.Tensor) -> torch.Tensor:
    """
    Packs a 1D tensor of signed int8 in [-8, 7] into uint8 bytes (2 values per byte).
    """
    flat = w_int4.flatten()
    if flat.numel() % 2 != 0:
        flat = F.pad(flat, (0, 1), value=0)
    u4 = (flat + 8).to(torch.uint8)
    low = u4[0::2]
    high = u4[1::2]
    packed = low | (high << 4)
    return packed


def unpack_int4_tensor(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    """
    Unpacks uint8 bytes into signed int8 values in [-8, 7].
    """
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.int8, device=packed.device)
    unpacked[0::2] = low
    unpacked[1::2] = high
    return unpacked[:original_numel]


class FakeQuantLinearINT4(nn.Module):
    """
    Linear layer with symmetric per-channel 4-bit weight fake-quantization.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_float(cls, float_layer: nn.Linear) -> "FakeQuantLinearINT4":
        layer = cls(float_layer.in_features, float_layer.out_features, bias=float_layer.bias is not None)
        with torch.no_grad():
            layer.weight.copy_(float_layer.weight)
            if float_layer.bias is not None:
                layer.bias.copy_(float_layer.bias)
        return layer

    def quantize_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-channel scale across input dimensions (dim 1)
        max_val = torch.amax(torch.abs(self.weight), dim=1, keepdim=True)
        scale = torch.clamp(max_val / 7.0, min=1e-8)
        # Quantize to [-8, 7]
        q_weight = torch.clamp(torch.round(self.weight / scale), -8.0, 7.0)
        dequant = q_weight * scale
        return dequant, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequant_w, _ = self.quantize_weights()
        # STE: straight-through estimator for gradients
        w_ste = self.weight + (dequant_w - self.weight).detach()
        return F.linear(x, w_ste, self.bias)

    def pack_weights(self) -> Dict[str, Any]:
        """
        Extracts packed 4-bit weights and scale factor.
        """
        with torch.no_grad():
            max_val = torch.amax(torch.abs(self.weight), dim=1, keepdim=True)
            scale = torch.clamp(max_val / 7.0, min=1e-8)
            q_weight = torch.clamp(torch.round(self.weight / scale), -8, 7).to(torch.int8)
            packed = pack_int4_tensor(q_weight)
            return {
                "packed_weight": packed,
                "scale": scale,
                "bias": self.bias.data.clone() if self.bias is not None else None,
                "shape": tuple(self.weight.shape),
            }

    def load_packed(self, packed_dict: Dict[str, Any]):
        shape = packed_dict["shape"]
        unpacked_q = unpack_int4_tensor(packed_dict["packed_weight"], shape[0] * shape[1])
        unpacked_q = unpacked_q.view(shape).to(self.weight.device, dtype=torch.float32)
        scale = packed_dict["scale"].to(self.weight.device)
        dequant = unpacked_q * scale
        with torch.no_grad():
            self.weight.copy_(dequant)
            if packed_dict["bias"] is not None and self.bias is not None:
                self.bias.copy_(packed_dict["bias"].to(self.weight.device))


def apply_int4_to_model(model: nn.Module) -> nn.Module:
    """
    Replaces classifier linear layers with FakeQuantLinearINT4.
    """
    for name, module in model.classifier.named_children():
        if isinstance(module, nn.Linear):
            int4_layer = FakeQuantLinearINT4.from_float(module)
            setattr(model.classifier, name, int4_layer)
    return model


def save_packed_int4_model(model: nn.Module, save_path: Path):
    """
    Saves state dictionary with packed INT4 weights for fake-quant layers.
    """
    save_dict = {}
    packed_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantLinearINT4):
            packed_layers[name] = module.pack_weights()

    # Save non-quantized weights as standard state dict
    standard_sd = {}
    for k, v in model.state_dict().items():
        # Exclude original floating weights for packed layers to save disk space
        layer_prefix = k.rsplit(".", 1)[0]
        if layer_prefix in packed_layers and k.endswith("weight"):
            continue
        standard_sd[k] = v

    save_dict["standard_weights"] = standard_sd
    save_dict["int4_packed_layers"] = packed_layers
    torch.save(save_dict, save_path)


def load_int4_model(save_path: Path = OUTPUT_CKPT, device: str = "cpu") -> nn.Module:
    model = build_model(num_classes=len(TARGET_CLASSES))
    model = apply_int4_to_model(model)

    ckpt = torch.load(save_path, map_location=device)
    if "int4_packed_layers" in ckpt:
        model.load_state_dict(ckpt["standard_weights"], strict=False)
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantLinearINT4) and name in ckpt["int4_packed_layers"]:
                module.load_packed(ckpt["int4_packed_layers"][name])
    else:
        # Fallback if standard state dict
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()
    return model


def run_int4(
    epochs: int = 5,
    lr: float = 5e-6,
    batch_size: int = 64,
    save_ckpt: bool = True,
    evaluate: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] INT4 fine-tuning device: {device}")

    # 1. Build model and load FP32 weights
    model = build_model(num_classes=len(TARGET_CLASSES))
    if BASELINE_CKPT.exists():
        model.load_state_dict(torch.load(BASELINE_CKPT, map_location="cpu"))
        print(f"[INFO] Loaded FP32 baseline weights from {BASELINE_CKPT}")
    else:
        raise FileNotFoundError(f"Baseline checkpoint not found at: {BASELINE_CKPT}")

    # 2. Wrap classifier with INT4 fake quantization
    model = apply_int4_to_model(model)
    model.to(device)

    # 3. Data loaders
    print("[INFO] Preparing data loaders for INT4 fine-tuning...")
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

    # 4. Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_metric = torchmetrics.AUROC(task="multilabel", num_labels=len(TARGET_CLASSES)).to(device)

    best_val_auc = -1.0
    best_weights = None

    print(f"\n[INFO] Starting INT4 fine-tuning for {epochs} epochs (lr={lr:.1e})...")
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

        # Validation
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

    print(f"\n[INFO] Best INT4 Validation AUC: {best_val_auc:.4f}")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 5. Pack and save checkpoint
    model.eval()
    if save_ckpt:
        save_packed_int4_model(model, OUTPUT_CKPT)
        size_mb = os.path.getsize(OUTPUT_CKPT) / (1024 * 1024)
        print(f"[INFO] Saved packed INT4 checkpoint -> {OUTPUT_CKPT} ({size_mb:.2f} MB)")
    else:
        size_mb = 0.0

    # 6. Evaluate on test set
    metrics = {}
    if evaluate:
        print("[INFO] Evaluating INT4 model on test set...")
        metrics = evaluate_model(model, device=device)
        print_metrics("INT4", metrics, size_mb=size_mb)

    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run INT4 Quantization")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    run_int4(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
