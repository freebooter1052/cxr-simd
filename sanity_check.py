import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from data.dataset import get_patient_splits, ChestXRayDataset, get_transforms, TARGET_CLASSES

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Un-normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=8)
    plt.axis('off')

def main():
    base_dir = Path(r"d:\coa\cxr-simd")
    csv_path = base_dir / "dataset" / "Data_Entry_2017.csv"
    image_dir = base_dir / "dataset" / "images-224" / "images-224"
    
    print("Generating patient-level splits (70/15/15)...")
    train_df, val_df, test_df = get_patient_splits(csv_path)
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")
    
    # Initialize datasets
    transforms = get_transforms()
    
    train_dataset = ChestXRayDataset(train_df, image_dir, transform=transforms)
    
    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    
    print("Fetching a batch of 8 images...")
    # Get a batch of training data
    inputs, classes = next(iter(train_loader))
    
    print(f"Batch inputs shape: {inputs.shape}")
    print(f"Batch labels shape: {classes.shape}")
    
    # Plotting
    plt.figure(figsize=(16, 8))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        # Convert binary tensor back to class names
        active_labels = [TARGET_CLASSES[j] for j, val in enumerate(classes[i]) if val == 1.0]
        title = " | ".join(active_labels) if active_labels else "No Target Finding"
        imshow(inputs[i], title=title)
        
    out_path = "sanity_check_output.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved sanity check visualization to {out_path}")

if __name__ == '__main__':
    main()
