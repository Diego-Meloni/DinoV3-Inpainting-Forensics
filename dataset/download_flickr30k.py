import shutil
import random
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent

def download_dataset():
    # Download files to local cache avoiding redundant network requests
    print(">>> Starting dataset download from HuggingFace...")
    snapshot_download(
        repo_id="lesc-unifi/beyond-the-brush",
        repo_type="dataset",
        allow_patterns="flickr30k/**",
        local_dir=str(SCRIPT_DIR) 
    )
    print("\n>>> Download completed or folder already synchronized.\n")
    
def split_and_standardize(seed=42):
    base_path = SCRIPT_DIR / "flickr30k"
    
    # Abort if the target training folder already exists to prevent duplication
    if (base_path / "train").exists():
        print(f">>> WARNING: The folder {base_path / 'train'} already exists.")
        print(">>> Operation cancelled.")
        return

    print(">>> Scanning files and extracting metadata...")
    images_dir = base_path / "images"
    all_images = list(images_dir.glob("*_inpainted-*.png"))
    
    # List containing a dictionary of metadata for each valid image-mask pair
    # Example: [{'img_path': Path(...), 'id': '953941506', 'mask_size': 'medium', ...}]
    dataset_pairs = []
    
    for original_img_path in tqdm(all_images, desc="Extracting pairs"):
        img_filename = original_img_path.name
        
        # Split filename to extract base ID and mask size
        # Example: "953941506_medium_3_inpainted-0.png" -> ['953941506', 'medium', '3', 'inpainted-0.png']
        parts = img_filename.split("_")
        
        if len(parts) < 2:
            continue
            
        img_id = parts[0]
        mask_size_label = parts[1] 
        
        # Reconstruct the expected mask name and verify its existence
        expected_mask_name = f"mask_{img_id}_{mask_size_label}.png"
        original_mask_path = base_path / "masks" / expected_mask_name
        
        if not original_mask_path.exists():
            continue
                
        # Open image temporarily to extract width and height without loading pixels
        with Image.open(original_img_path) as img:
            width, height = img.size
            
        dataset_pairs.append({
            "img_path": original_img_path,
            "mask_path": original_mask_path,
            "filename": img_filename,
            "id": img_id,
            "mask_size": mask_size_label,
            "shape": f"{width}x{height}"
        })

    if not dataset_pairs:
        print("No valid pairs found.")
        return

    print("\n>>> Grouping by Image ID to prevent Data Leakage...")
    
    # Group all image variants by their base ID to keep them in the same split
    # Example: {'953941506': [{pair_dict_1}, {pair_dict_2}], '12345': [{pair_dict_3}]}
    grouped_by_id = {}
    for pair in dataset_pairs:
        grouped_by_id.setdefault(pair['id'], []).append(pair)
        
    unique_ids = list(grouped_by_id.keys())
    unique_ids.sort()
    
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    
    # Calculate indices for an 80/10/10 split over the unique base IDs
    n_ids = len(unique_ids)
    train_end = int(n_ids * 0.8)
    val_end = int(n_ids * 0.9)
    
    # Sets containing the IDs assigned to each split for fast O(1) lookup
    train_ids = set(unique_ids[:train_end])
    val_ids = set(unique_ids[train_end:val_end])

    print("\n>>> Creating folder structure...")
    
    # Create target directories for Train, Val, and Test
    splits = ["train", "val", "test"]
    for split in splits:
        (base_path / f"{split}" / "images").mkdir(parents=True, exist_ok=True)
        (base_path / f"{split}" / "masks").mkdir(parents=True, exist_ok=True)

    print(">>> Distributing file groups to their splits...")
    
    # Dictionary to accumulate final metadata for CSV generation
    # Example: {'train': [{'image_path': '...', 'mask_size': 'small'}, ...], 'val': [...]}
    records_by_split = {"train": [], "val": [], "test": []}
    
    for pair in tqdm(dataset_pairs, desc="Moving files"):
        # Determine target split by checking which group the base ID belongs to
        if pair['id'] in train_ids:
            split = "train"
        elif pair['id'] in val_ids:
            split = "val"
        else:
            split = "test"
            
        target_img_dir = base_path / f"{split}" / "images"
        target_mask_dir = base_path / f"{split}" / "masks"
        
        target_img_path = target_img_dir / pair["img_path"].name
        target_mask_path = target_mask_dir / pair["mask_path"].name
        
        # Physically move the unique image
        if pair["img_path"].exists():
            shutil.move(str(pair["img_path"]), str(target_img_path))
            
        # Move the shared mask (only the first image in the group will actually move it)
        if pair["mask_path"].exists():
            shutil.move(str(pair["mask_path"]), str(target_mask_path))
            
        records_by_split[split].append({
            "id": pair["id"],
            "image_path": str(target_img_path.relative_to(base_path)),
            "mask_path": str(target_mask_path.relative_to(base_path)),
            "mask_size": pair["mask_size"],
            "original_shape": pair["shape"]
        })

    print("\n>>> Generating CSV files...")
    
    # Generate, sort, and save a CSV file for each split
    for split in splits:
        if records_by_split[split]:
            df = pd.DataFrame(records_by_split[split])
            df = df.sort_values(by=['id', 'mask_size']).reset_index(drop=True)
            
            csv_target_path = base_path / f"{split}" /f"{split}.csv"
            df.to_csv(csv_target_path, index=False)
            print(f"- Saved {len(df)} records to: {csv_target_path.name}")
            
    # Clean up the original HuggingFace folders if they are strictly empty
    
    # Check for orphan masks (masks not used by any image)
    leftover_masks = list((base_path / "masks").glob("*.png"))
    all_orphans = True
    
    for mask_path in leftover_masks:
        # Estrae ID e Taglia: "mask_241480898_medium.png" -> ['mask', '241480898', 'medium.png']
        parts = mask_path.name.split("_")
        if len(parts) >= 3:
            img_id = parts[1]
            mask_size = parts[2].replace(".png", "")
            
            # Ora cerca solo le immagini che hanno ESATTAMENTE quell'ID e quella TAGLIA
            found_images = list(base_path.rglob(f"{img_id}_{mask_size}_*.png"))
            found_images = [f for f in found_images if "mask_" not in f.name]
            
            if found_images:
                print(f"- [ERRORE] La maschera {mask_path.name} ha un'immagine viva: {found_images[0].name}")
                all_orphans = False
                break
                
    if all_orphans:
        print("- All remaining masks are orphans. Eliminating the original folders.")
        shutil.rmtree(base_path / "images", ignore_errors=True)
        shutil.rmtree(base_path / "masks", ignore_errors=True)
    else:
        print("- ATTENTION: Found some non orphan masks left, keeping the folder.")

    print("\n>>> Dataset successfully grouped and standardized!")

if __name__ == "__main__":
    check_path = SCRIPT_DIR / "flickr30k" / "train"
    
    if check_path.exists():
        print(">>> The dataset has already been downloaded, split, and standardized.")
        print(">>> No action required. You can start the training!")
    else:
        download_dataset()

        # Some info about the dataset we just downloaded
        base_path = SCRIPT_DIR / "flickr30k"
        images_dir = base_path / "images"
        masks_dir = base_path / "masks"

        num_images = len(list(images_dir.glob("*.png")))
        num_masks = len(list(masks_dir.glob("*.png")))

        print("\n" + "="*40)
        print(">>> DATASET FILE COUNT")
        print(f">>> Images found: {num_images}")
        print(f">>> Masks found:  {num_masks}")
        print("="*40 + "\n")
        
        split_and_standardize()