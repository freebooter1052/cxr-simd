"""
diagnose.py
-----------
End-to-End Disease Classification & Model Comparison Tool.

Takes a chest X-ray image (224x224 PNG/JPG) and predicts the presence of:
  - Pneumonia
  - Cardiomegaly
  - Effusion

Compares diagnosis probabilities, confidence, and inference latency across
all 4 precision variants:
  1. FP32 (Full Precision Baseline)
  2. INT8-PTQ (Post-Training Quantization)
  3. INT8-QAT (Quantization-Aware Training)
  4. INT4 (Aggressive 4-bit edge compression)

Usage:
  # Predict on a specific image:
  python diagnose.py --image dataset/images-224/images-224/00000001_000.png

  # Pick an image automatically from the test dataset:
  python diagnose.py
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

ROOT = Path(__file__).resolve().parent
TARGET_CLASSES = ["Pneumonia", "Cardiomegaly", "Effusion"]

# ImageNet normalization parameters used during training
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess_image(image_path: Path) -> np.ndarray:
    """Preprocess chest X-ray image to [1, 3, 224, 224] float32 array."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # [224, 224, 3]
    arr = np.transpose(arr, (2, 0, 1))            # [3, 224, 224]
    arr = np.expand_dims(arr, axis=0)             # [1, 3, 224, 224]
    arr = (arr - MEAN) / STD
    return arr.astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_onnx_inference(model_path: Path, input_data: np.ndarray, warmup=3, runs=20):
    """Run inference using ONNXRuntime and return (probabilities, latency_ms)."""
    if not model_path.exists():
        return None, None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        session.run(None, {inp_name: input_data})

    # Benchmark
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = session.run(None, {inp_name: input_data})
        times.append((time.perf_counter() - t0) * 1000.0)

    logits = out[0].squeeze(0)
    probs = sigmoid(logits)
    mean_latency = float(np.mean(times))
    return probs, mean_latency


def main():
    parser = argparse.ArgumentParser(description="Predict chest diseases from an X-ray image")
    parser.add_argument("--image", type=str, default=None, help="Path to input X-ray image")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs for timing")
    args = parser.parse_args()

    # Find sample image if none provided
    if args.image is None:
        candidate_dir = ROOT / "dataset" / "images-224" / "images-224"
        sample_images = list(candidate_dir.glob("*.png")) if candidate_dir.exists() else []
        if not sample_images:
            print("[ERROR] No image provided and no images found in dataset/images-224/images-224")
            sys.exit(1)
        image_path = sample_images[0]
        print(f"[INFO] No image specified. Defaulting to sample: {image_path.name}")
    else:
        image_path = Path(args.image)

    if not image_path.exists():
        print(f"[ERROR] Image file does not exist: {image_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("           CXR-SIMD: CHEST X-RAY MULTI-DISEASE DIAGNOSIS             ")
    print("=" * 70)
    print(f" Target Image : {image_path.name}")
    print(f" File Path    : {image_path.resolve()}")

    input_data = preprocess_image(image_path)

    models = [
        ("FP32 Baseline",      ROOT / "quantize" / "onnx" / "fp32.onnx"),
        ("INT8-PTQ (3.8x)",    ROOT / "quantize" / "onnx" / "int8_ptq.onnx"),
        ("INT8-QAT (3.8x)",    ROOT / "quantize" / "onnx" / "int8_qat.onnx"),
        ("INT4 Fake-Quant",    ROOT / "quantize" / "onnx" / "int4.onnx"),
    ]

    results = {}
    print("\n[INFO] Running model inference across precision levels...")
    for label, path in models:
        probs, lat = run_onnx_inference(path, input_data, runs=args.runs)
        if probs is not None:
            results[label] = (probs, lat)

    # Print Probability Table
    print("\n" + "-" * 70)
    print(f"{'Condition':<16} | {'FP32 Base':<10} | {'INT8 PTQ':<10} | {'INT8 QAT':<10} | {'INT4':<10} | {'Final Call'}")
    print("-" * 70)

    for idx, disease in enumerate(TARGET_CLASSES):
        vals = []
        for label, _ in models:
            if label in results:
                p = results[label][0][idx] * 100.0
                vals.append(f"{p:5.1f}%")
            else:
                vals.append("N/A")
        
        # Determine consensus status based on FP32 / QAT
        p_ref = results["INT8-QAT (3.8x)"][0][idx] if "INT8-QAT (3.8x)" in results else results["FP32 Baseline"][0][idx]
        call = "[POSITIVE]" if p_ref >= 0.50 else "[NEGATIVE]"
        print(f"{disease:<16} | {vals[0]:<10} | {vals[1]:<10} | {vals[2]:<10} | {vals[3]:<10} | {call}")

    print("-" * 70)

    # Print Latency & Performance Table
    print("\n" + "-" * 70)
    print(" INFERENCE LATENCY & THROUGHPUT (CPU)")
    print("-" * 70)
    base_lat = results.get("FP32 Baseline", (None, None))[1]
    for label, _ in models:
        if label in results:
            _, lat = results[label]
            speedup = f"{base_lat / lat:.2f}x" if (base_lat and lat) else "1.00x"
            fps = 1000.0 / lat if lat > 0 else 0
            print(f"  * {label:<20}: {lat:6.2f} ms ({fps:5.1f} FPS)  [Speedup: {speedup}]")

    print("=" * 70)

    # Clinical findings summary
    print("\nCLINICAL SUMMARY:")
    detected = []
    for idx, disease in enumerate(TARGET_CLASSES):
        p = results.get("INT8-QAT (3.8x)", results.get("FP32 Baseline"))[0][idx]
        if p >= 0.50:
            detected.append(f"{disease} (confidence: {p*100:.1f}%)")

    if detected:
        print(f"  >> DETECTED PATHOLOGIES: {', '.join(detected)}")
    else:
        print("  >> NO TARGET ABNORMALITIES DETECTED (All probabilities < 50%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
