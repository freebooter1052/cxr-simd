"""
evaluate.py
-----------
Standalone evaluation and inference script for the trained baseline model.

Usage:
  1. Full test split evaluation:
     python evaluate.py

  2. Predict on a single image:
     python evaluate.py --image path/to/image.png
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.models as tv_models
from PIL import Image
from torch.utils.data import DataLoader
import torchmetrics

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import get_patient_splits, ChestXRayDataset, get_transforms, TARGET_CLASSES

CSV_PATH = ROOT / "dataset" / "Data_Entry_2017.csv"
IMAGE_DIR = ROOT / "dataset" / "images-224" / "images-224"
CKPT_PATH = ROOT / "train" / "checkpoints" / "baseline_fp32.pt"


def load_model(ckpt_path, num_classes=3, device="cpu"):
    model = tv_models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_test_set(device):
    print("[INFO] Loading dataset test split...")
    _, _, test_df = get_patient_splits(CSV_PATH)
    print(f"Test samples: {len(test_df)}")

    test_ds = ChestXRayDataset(test_df, IMAGE_DIR, transform=get_transforms())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = load_model(CKPT_PATH, num_classes=len(TARGET_CLASSES), device=device)

    auroc = torchmetrics.AUROC(task="multilabel", num_labels=len(TARGET_CLASSES)).to(device)
    auroc_per_class = torchmetrics.AUROC(task="multilabel", num_labels=len(TARGET_CLASSES), average=None).to(device)
    f1 = torchmetrics.F1Score(task="multilabel", num_labels=len(TARGET_CLASSES), threshold=0.5).to(device)
    precision = torchmetrics.Precision(task="multilabel", num_labels=len(TARGET_CLASSES), threshold=0.5).to(device)
    recall = torchmetrics.Recall(task="multilabel", num_labels=len(TARGET_CLASSES), threshold=0.5).to(device)

    print("[INFO] Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            auroc.update(probs, labels.int())
            auroc_per_class.update(probs, labels.int())
            f1.update(probs, labels.int())
            precision.update(probs, labels.int())
            recall.update(probs, labels.int())

    macro_auc = auroc.compute().item()
    per_class = auroc_per_class.compute()
    macro_f1 = f1.compute().item()
    macro_prec = precision.compute().item()
    macro_rec = recall.compute().item()

    print("\n" + "=" * 45)
    print("             TEST SET RESULTS")
    print("=" * 45)
    print(f"  Macro ROC-AUC  : {macro_auc:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Macro Precision: {macro_prec:.4f}")
    print(f"  Macro Recall   : {macro_rec:.4f}")
    print("-" * 45)
    print("  Per-Class ROC-AUC:")
    for idx, name in enumerate(TARGET_CLASSES):
        print(f"    - {name:<15}: {per_class[idx].item():.4f}")
    print("=" * 45)


def predict_single_image(image_path, device):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    transform = get_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = load_model(CKPT_PATH, num_classes=len(TARGET_CLASSES), device=device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    print("\n" + "=" * 45)
    print(f"Prediction for: {image_path.name}")
    print("=" * 45)
    for idx, name in enumerate(TARGET_CLASSES):
        prob = probs[idx].item()
        bar = "#" * int(prob * 20)
        status = "[POSITIVE]" if prob >= 0.5 else "[NEGATIVE]"
        print(f"  {name:<15}: {prob*100:5.1f}% {bar:<20} {status}")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline FP32 model")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image for prediction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if args.image:
        predict_single_image(args.image, device)
    else:
        evaluate_test_set(device)
