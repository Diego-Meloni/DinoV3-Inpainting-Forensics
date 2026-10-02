import os
import cv2
import numpy as np

def cleanup_empty_masks_and_orphans():
    # Dynamically resolve paths relative to this script location
    script_dir = os.path.dirname(os.path.abspath(__file__))   # .../config_and_utilities/
    generation_dir = os.path.dirname(script_dir)              # .../01_inpainting_generation/
    project_root = os.path.dirname(generation_dir)            # .../Beyond-the-Brush/
    
    # Define absolute paths using the dynamically resolved directories
    masks_dir = os.path.join(project_root, 'dataset', 'manually_generated', 'final_masks')
    boxes_dir = os.path.join(project_root, 'dataset', 'manually_generated', 'final_boxes')
    
    # The labels file is in the same directory as this script
    labels_file = os.path.join(script_dir, 'best_mask_labels_custom_dataset.txt')
    
    deleted_count = 0
    deleted_identifiers = set()

    print("Scanning for empty masks...")

    
    # Step 1: Find and delete empty masks, and collect their unique identifiers
    for file_name in os.listdir(masks_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(masks_dir, file_name)
            
            # Read image in grayscale to easily check pixel values
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Skip if the file cannot be read properly
            if image_array is None:
                continue
            
            # Check if the maximum pixel value is 0 (completely black mask)
            if np.max(image_array) == 0:
                os.remove(file_path)
                
                # Extract the identifier string (e.g., from "mask_0000120_small.png" to "0000120_small")
                identifier = file_name.replace('mask_', '').replace('.png', '')
                deleted_identifiers.add(identifier)
                
                deleted_count += 1
                print(f"Deleted mask: {file_name}")

    # Step 2: Delete corresponding green-box images using the collected identifiers
    boxes_deleted = 0
    for identifier in deleted_identifiers:
        # Reconstruct the expected file name for the bounding box image
        box_file_name = f"mask_{identifier}_with_box.png" 
        box_file_path = os.path.join(boxes_dir, box_file_name)
        
        if os.path.exists(box_file_path):
            os.remove(box_file_path)
            boxes_deleted += 1

    # Step 3: Remove orphaned entries from the labels text file
    labels_removed = 0
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        
        valid_lines = []
        for line in lines:
            # The label file format is "{imageName}_{size} - {label}"
            # Check if the current line starts with any of our deleted identifiers
            is_orphaned = any(line.startswith(f"{identifier} -") for identifier in deleted_identifiers)
            
            if not is_orphaned:
                valid_lines.append(line)
            else:
                labels_removed += 1
                
        # Rewrite the file keeping only the valid lines
        with open(labels_file, 'w') as f:
            f.writelines(valid_lines)

    # Print final summary for the user
    print("-" * 30)
    print("Cleanup Summary:")
    print(f"Empty masks deleted: {deleted_count}")
    print(f"Orphaned box images deleted: {boxes_deleted}")
    print(f"Orphaned label entries removed: {labels_removed}")

if __name__ == "__main__":
    cleanup_empty_masks_and_orphans()