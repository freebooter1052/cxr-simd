# Phase 2 — Baseline FP32 Model Deliverables & Results

## 1. Overview & Setup
- **Architecture**: `MobileNetV3-Small` (ImageNet pre-trained), final layer adapted to 3 outputs: `Pneumonia`, `Cardiomegaly`, `Effusion`
- **Loss**: `BCEWithLogitsLoss` (multi-label)
- **Optimizer**: Adam, initial $\text{lr} = 1\times 10^{-4}$, `ReduceLROnPlateau(factor=0.5, patience=3)`
- **Stopping**: Early stopping on mean validation AUROC (patience = 7 epochs)
- **Hardware Acceleration**: NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.4)
- **Data Splits**: Patient-level split — 11,186 Train | 2,546 Val | 2,368 Test

---

## 2. Reference Results Table (Baseline FP32)

### Test Set Metrics (Held-out Patients)
| Pathology / Metric | ROC-AUC | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Cardiomegaly** | **0.8937** | — | — | — |
| **Effusion** | **0.8588** | — | — | — |
| **Pneumonia** | **0.7385** | — | — | — |
| **Macro Average (Overall)** | **0.8303** | **0.8394** | **0.8775** | **0.8044** |

### Validation Set Metrics
| Pathology / Metric | ROC-AUC | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Cardiomegaly** | **0.9055** | — | — | — |
| **Effusion** | **0.8495** | — | — | — |
| **Pneumonia** | **0.7327** | — | — | — |
| **Macro Average (Overall)** | **0.8292** | **0.8316** | **0.8728** | **0.7941** |

---

## 3. Training History Summary
- Best validation AUROC reached at **Epoch 6** ($\text{AUC} = 0.8292$).
- Checkpoint was saved to [`train/checkpoints/baseline_fp32.pt`](file:///d:/coa/cxr-simd/train/checkpoints/baseline_fp32.pt).
- Early stopping triggered at Epoch 13 after 7 epochs without higher val AUROC.

---

## 4. Deliverables Checklist
- [x] Training script: [`train/train_baseline.py`](file:///d:/coa/cxr-simd/train/train_baseline.py)
- [x] Best model weights: [`train/checkpoints/baseline_fp32.pt`](file:///d:/coa/cxr-simd/train/checkpoints/baseline_fp32.pt)
- [x] Full metrics log: [`train/metrics.json`](file:///d:/coa/cxr-simd/train/metrics.json)
- [x] Standalone test & inference tool: [`evaluate.py`](file:///d:/coa/cxr-simd/evaluate.py)
