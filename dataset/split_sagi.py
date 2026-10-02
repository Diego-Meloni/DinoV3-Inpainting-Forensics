import shutil
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Paths configuration
BASE_DIR = Path("dataset/SAGI-D")
ORIGINAL_CSV = BASE_DIR / "sagi_dataset.csv"
IMAGES_DIR = BASE_DIR / "images_inpaint"
MASKS_DIR = BASE_DIR / "masks"

def split_dataset():
    print(">>> Loading original CSV...")
    df = pd.read_csv(ORIGINAL_CSV)
    
    # Initialize containers for valid data and mismatch tracking
    valid_records = []
    mismatch_counter = {model: 0 for model in df['inpainting_model'].unique()}
    missing_files = 0
    
    print(">>> Verifying files and extracting metadata...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking dimensions"):
        # The CSV has long kaggle paths (e.g., sagid\train\coco\original\0000.jpg)
        # We extract just the base filename to locate it in our local flat folders
        img_filename = Path(row['img_path']).name
        mask_filename = Path(row['mask_path']).name
        
        img_local_path = IMAGES_DIR / img_filename
        mask_local_path = MASKS_DIR / mask_filename
        
        # Check if files physically exist to avoid crash
        if not img_local_path.exists() or not mask_local_path.exists():
            missing_files += 1
            continue
            
        # Verify dimensions by just reading the file header (fast)
        with Image.open(img_local_path) as img, Image.open(mask_local_path) as mask:
            img_shape = img.size  # (width, height)
            mask_shape = mask.size
            
        # Drop the pair if sizes do not align to prevent PyTorch crashes
        if img_shape != mask_shape:
            mismatch_counter[row['inpainting_model']] += 1
            continue 
            
        # Keep valid records with new requested fields
        valid_records.append({
            'filename': img_filename,
            'mask_filename': mask_filename,
            'original_img_path': str(img_local_path),
            'original_mask_path': str(mask_local_path),
            'inpainting_model': row['inpainting_model'],
            'prompt': row.get('prompt', ''),
            'img_shape': f"{img_shape[0]}x{img_shape[1]}",
            'mask_shape': f"{mask_shape[0]}x{mask_shape[1]}"
        })
        
    print("\n>>> DATASET VERIFICATION RESULTS")
    print(f"Missing files skipped: {missing_files}")
    print("Mismatched sizes (safely excluded) per model:")
    for model, count in mismatch_counter.items():
        if count > 0:
            print(f"  - {model}: {count} images")
            
    valid_df = pd.DataFrame(valid_records)
    print(f"\nTotal valid image-mask pairs ready for split: {len(valid_df)}")
    
    # Stratified Split: 80% Train, 10% Val, 10% Test
    print("\n>>> Performing stratified split...")
    train_df, temp_df = train_test_split(
        valid_df, test_size=0.20, 
        stratify=valid_df['inpainting_model'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, 
        stratify=temp_df['inpainting_model'], random_state=42
    )
    
    # Dictionary mapping splits to their dataframes
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name, split_data in splits.items():
        print(f"\n>>> Creating {split_name} split ({len(split_data)} images)...")
        split_dir = BASE_DIR / f"{split_name}"
        target_img_dir = split_dir / "images"
        target_mask_dir = split_dir / "masks"
        
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)
        
        final_csv_records = []
        
        for _, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Moving {split_name}"):
            src_img = Path(row['original_img_path'])
            src_mask = Path(row['original_mask_path'])
            
            dst_img = target_img_dir / row['filename']
            dst_mask = target_mask_dir / row['mask_filename']
            
            # Physically move files to their new split folders
            if src_img.exists():
                shutil.move(str(src_img), str(dst_img))
            if src_mask.exists():
                shutil.move(str(src_mask), str(dst_mask))
                
            final_csv_records.append({
                'filename': row['filename'],
                'imgage_path': f"{split_name}/images/{row['filename']}",
                'mask_path': f"{split_name}/masks/{row['mask_filename']}",
                'inpainting_model': row['inpainting_model'],
                'image_shape': row['img_shape'],
                'mask_shape': row['mask_shape'],
                'prompt': row['prompt']
            })
            
        # Save CSV for the current split directly inside the split's folder
        csv_path = split_dir / f"{split_name}.csv"
        pd.DataFrame(final_csv_records).to_csv(csv_path, index=False)
        print(f"- Saved split CSV to {csv_path.name}")

    # Optional: Clean up original empty folders if you want
    # shutil.rmtree(IMAGES_DIR, ignore_errors=True)
    # shutil.rmtree(MASKS_DIR, ignore_errors=True)

    print("\n>>> Split completed successfully!")

if __name__ == "__main__":
    split_dataset()