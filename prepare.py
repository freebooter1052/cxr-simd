import pandas as pd
import os
import shutil
from pathlib import Path
from PIL import Image

def main():
    base_dir = Path(r"d:\coa\cxr-simd")
    dataset_dir = base_dir / "dataset"
    images_dir = dataset_dir / "images-224" / "images-224"
    csv_path = dataset_dir / "Data_Entry_2017.csv"
    
    out_dir = dataset_dir / "processed"
    out_dir.mkdir(exist_ok=True)
    
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(exist_ok=True)

    print("Loading metadata...")
    df = pd.read_csv(csv_path)

    # Focus pathologies
    target_pathologies = ['Pneumonia', 'Cardiomegaly', 'Effusion']
    
    print(f"Filtering for pathologies: {target_pathologies}")
    
    # We can either look for exact matches or rows that contain these pathologies
    # We will create a boolean mask for rows containing any of the target pathologies
    mask = df['Finding Labels'].apply(lambda x: any(p in x for p in target_pathologies))
    filtered_df = df[mask].copy()
    
    print(f"Found {len(filtered_df)} matching records out of {len(df)}")
    
    # Optional: we can add one-hot encoded columns for our target pathologies
    for p in target_pathologies:
        filtered_df[p] = filtered_df['Finding Labels'].apply(lambda x: 1 if p in x else 0)

    # Now let's preprocess the images (e.g., ensure they are copied and resized if necessary)
    # The current images-224 seem to be 224x224 already, but we will verify and copy them
    processed_records = []
    
    print("Preprocessing and copying images...")
    count = 0
    for idx, row in filtered_df.iterrows():
        img_name = row['Image Index']
        src_path = images_dir / img_name
        dst_path = out_images_dir / img_name
        
        if src_path.exists():
            # Copy file
            shutil.copy2(src_path, dst_path)
            processed_records.append(row)
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} images...")
        else:
            print(f"Warning: Image {img_name} not found at {src_path}")
            
    final_df = pd.DataFrame(processed_records)
    out_csv = out_dir / "filtered_metadata.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Done! Saved filtered metadata to {out_csv}")

if __name__ == "__main__":
    main()
