import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from pathlib import Path

TARGET_CLASSES = ['Pneumonia', 'Cardiomegaly', 'Effusion']

def get_patient_splits(csv_path, val_size=0.15, test_size=0.15, random_state=42):
    """
    Reads the metadata, filters for target classes, and performs a patient-level split.
    Returns: train_df, val_df, test_df
    """
    df = pd.read_csv(csv_path)
    
    # Filter for our 3 target pathologies
    mask = df['Finding Labels'].apply(lambda x: any(p in x for p in TARGET_CLASSES))
    df = df[mask].copy()
    
    # Get unique patient IDs
    unique_patients = df['Patient ID'].unique()
    
    # Calculate split sizes
    # If val is 0.15 and test is 0.15, we first split test out
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state
    )
    
    # Now split the remaining into train and val
    # val_size is relative to the total, so we need to adjust
    relative_val_size = val_size / (1.0 - test_size)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=relative_val_size, random_state=random_state
    )
    
    # Create the dataframes for each split
    train_df = df[df['Patient ID'].isin(train_patients)].copy()
    val_df = df[df['Patient ID'].isin(val_patients)].copy()
    test_df = df[df['Patient ID'].isin(test_patients)].copy()
    
    # Create binary labels
    for p in TARGET_CLASSES:
        train_df[p] = train_df['Finding Labels'].apply(lambda x: 1 if p in x else 0)
        val_df[p] = val_df['Finding Labels'].apply(lambda x: 1 if p in x else 0)
        test_df[p] = test_df['Finding Labels'].apply(lambda x: 1 if p in x else 0)
        
    return train_df, val_df, test_df

class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe containing the image names and labels.
            image_dir (str or Path): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Pre-extract labels into a tensor for faster access
        self.labels = self.dataframe[TARGET_CLASSES].values.astype('float32')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'Image Index']
        img_path = self.image_dir / img_name
        
        # Some images might be grayscale, convert to RGB for standard models
        image = Image.open(img_path).convert('RGB')
        
        labels = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels)

def get_transforms():
    """
    Returns standard ImageNet transforms.
    Resize to 224x224, convert to Tensor, and normalize.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
