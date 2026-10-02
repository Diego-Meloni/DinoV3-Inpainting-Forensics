import numpy as np
import json
from PIL import Image
from time import time
import os
import random
import cv2
from tqdm import tqdm
import argparse
import traceback

# Setup paths relative to this script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_utils_dir = os.path.join(script_dir, 'config_and_utilities')
output_dir = os.path.join(project_root, 'dataset', 'manually_generated')

print("Paths information:")
print(f"\nScript directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"config_utils directory: {config_utils_dir}")
print(f"Output directory: {output_dir}\n")

"""
This file takes a folder of bounding boxes and pixel masks (output of 1_process_images.py), selects the most useful masks by size category, draws bounding boxes on the images. 
3 representative masks per image.
INPUTS: 
- process_images.py outputs: .json files and .npz files containing binary segmentation masks.
OUTPUTS: 
- clean binary mask png, one for each size (small, medium, large). White regions = areas to be inpainted.
- original images with green bounding boxes around the selected objects represented in masks.
- best_mask_labels_{dataset_name}.txt, text file recording which object label corresponds to each mask.
- text files containing the paths of original .jpg images and .json files corresponding to the sampled images.
"""

def sample_n_images(input_dir, n_images, dataset_name):
    
    # os.listdir: returns the list of strings representing the name of files in the input directory.
    # os.path.join: returns the correct path to files, joining the file name with the directory name.
    json_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith("json")] #list automatically filters to only .json files (file.endswith('json'))
    # if we didn't specify n_images or if it's larger that the dataset we sample all images.
    if n_images is None or n_images > len(json_files):
        print(f"You are using all the files in the input directory.")
        # we create a .txt file containing the paths of the images we sampled for reproducibility reasons.
        with open(os.path.join(config_utils_dir, f'sampled_{len(json_files)}_images_{dataset_name}.txt'), 'w') as f:
            for file in json_files:
                f.write(file+"\n")
        return json_files
    # otherwise we sample a random subset of images of length "n_images".
    else:
        json_files_sampled = random.sample(json_files, n_images)
        with open(os.path.join(config_utils_dir, f'sampled_{n_images}_images_{dataset_name}.txt'), 'w') as f:
            for file in json_files_sampled:
                f.write(file+"\n")
        return json_files_sampled


# we compute the percentage area that the mask covers of the original image. Also computes which of the 3 masks produced by SAM is the largest.
# INPUT: mask NumPy arrays of shape (num_boxes, 3, H, W).
def compute_mask_area(masks, box):
    counts=np.asarray([0,0,0], dtype=np.float32)
    masks = masks.astype(np.uint8) # convert boolean NumPy mask array to a {0,1} NumPY mask array.
    # we iterate over the 3 mask channels, i.e. the 3 masks produced by SAM.
    for chn in range(masks.shape[1]): 
        counts[chn] = masks[box][chn].sum() # mask[box] slices out the entry of only 1 box. sum() counts the number of "True"
        # counts contains the number of true pixels for each of the 3 masks.            
    areas = counts / (masks.shape[2]*masks.shape[3]) # normalize mask coverage to values [0,1].
    max_area = areas.max() # take max area value.
    chn = areas.argmax() # take the index of the highest value of areas. Corresponds to the index of the biggest mask.

    # returns the mask covering the biggest area plus the value of area covered.
    return float(max_area), int(chn)


# masks might cover disconnected objects. The function counts how many disconnected objects are inside a mask.
def compute_connected_components(image):
    num_labels, _, _, _= cv2.connectedComponentsWithStats(image) # only return number of connected regions in the mask, including the background.
    num_connected_components = num_labels - 1 # added to avoid counting the background.
    return num_connected_components


# Updates the input .json files containing information on the bounding boxes and their corresponding labels.
# adds mask size and the number of connected components to the .json files containing the bounding boxes.
def update_json_file(data, image_json_path):
    # data: .json files of bounding boxes. image_json_path: path to .json files of images, used to retrieve the compressed .nps masks.
    
    # load the .npz compressed files, retrieving the stores mask NumPy arrays (['masks']).
    masks = np.load(image_json_path.replace(".json", ".npz"))['masks']
    
    # for each bounding box in the .json file (data['boxes']) call compute_mask_area and compute_connected_components.
    for box_num, box in enumerate(data['boxes']):
        
        area, chn = compute_mask_area(masks, box_num)
        
        # Classify box into size categories based on area percentage of image
        box['area'] = area
        if 0 <= area and area <= 0.15:
            dim = 'small'
        elif 0.15 < area and area <= 0.3:
            dim = 'medium'
        elif 0.3 < area and area <= 0.6:
            dim = 'large'
        else:
            dim = 'extra_large'
        
        # adds to the singular box .json file its area, dimension and channel with max area.    
        box['area_size'] = dim
        box['channel_max_area'] = chn
        
        image_array = masks[box_num, chn, :, :].astype(np.uint8) * 255

        num_connected_components = compute_connected_components(image_array)
        box['num_connected_components'] = num_connected_components
    
    # add computed data to the .json file 
    with open(image_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    

# we select the best candidate mask for each size. We then clean the mask and save it as a .png file.
def get_top_3_masks(image_json_path, save_path):
    with open(image_json_path, 'r') as file:
        data = json.load(file)
    
    img_name = os.path.basename(data['path']).replace(".jpg", "")
    masks = np.load(image_json_path.replace('.json', '.npz'))['masks']
    
    # Initialize dict to store best mask index for each size category
    best_masks = {'small':None, 'medium':None, 'large':None}

    # Find first occurrence of each size category (limit to top 30 detections)
    for box_num, box in enumerate(data['boxes'][:30]):
        if box['area_size'] != 'extra_large' and best_masks[box['area_size']] == None:
            best_masks[box['area_size']] = box_num
        
    # Track which size categories were "borrowed" from others (for fallback logic)
    second_medium, second_large, second_small = False, False, False
    
    # Create lists of indices for each size category
    smalls = [i for i, med_box in enumerate(data['boxes'][:30]) if med_box['area_size'] == 'small']
    mediums = [i for i, med_box in enumerate(data['boxes'][:30]) if med_box['area_size'] == 'medium']
    larges = [i for i, large_box in enumerate(data['boxes'][:30]) if large_box['area_size'] == 'large']
    
    # If small not found, use second-best medium or large
    if best_masks['small'] == None:
        if len(mediums) > 1:
            best_masks['small'] = mediums[1]
            second_medium = True
        elif len(larges) > 1:
            best_masks['small'] = larges[1]
            second_large = True

    # If medium not found, use second-best large or medium        
    if best_masks['medium'] == None:
        if len(larges) > 1 and len(larges) > 2:
            if second_large:
                best_masks['medium'] = larges[2]
            elif not second_large:
                best_masks['medium'] = larges[1]
                second_large = True
        elif len(smalls) > 1:
            if best_masks['small'] != None:
                best_masks['medium'] = smalls[1]
                second_small = True

    # If large not found, use second-best medium or small
    if best_masks['large'] == None:
        if len(mediums) > 1:
            if not second_medium:
                best_masks['large'] = mediums[1]
                second_medium = True
            elif second_medium and len(mediums) > 2: # Skip first medium if already used
                best_masks['large'] = mediums[2]
        elif len(smalls) > 1:
            if second_small and len(smalls) > 2:
                best_masks['large'] = smalls[2]
            elif not second_small:
                best_masks['large'] = smalls[1]

    # Save selected masks as png
    for key, value in best_masks.items():
        if value is not None:
            chn = data['boxes'][value]['channel_max_area']
            mask_dim = key
            image_array = masks[value, chn, :, :].astype(np.uint8) * 255
            image = Image.fromarray(image_array)
            
            # Apply morphological operations to smooth and clean the mask
            # - Opening removes small noise and thin connections
            # - Closing fills small holes in the mask
            img = np.array(image)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            
            cv2.imwrite(f'{save_path}/mask_{img_name}_{mask_dim}.png', closing)
            
        else:
            print(f"Missing {key} region for image {img_name}!")
            with open(os.path.join(save_path, "missed_regions.txt"), "a") as missing:
                missing.write(f"Missing {key} region for image {img_name}\n")
    return best_masks


# draw green rectangle to mark where the selected object is. Later fed to generate_prompts.py to the vision-language model to suggest replacements.
def draw_bounding_box(image, start_point, end_point, img_name, save_path):
    image = cv2.rectangle(image, start_point, end_point, color=(0, 255, 0), thickness=3)
    cv2.imwrite(f'{save_path}/mask_{img_name}_with_box.png', image)
    

# alternative to draw_bounding_box. Produces a white bounding box on a black background.
# NOT USED.      
def mask_with_bounding_box(image, start_point, end_point, img_name, save_path):
    black_image = np.zeros_like(image)
    
    width = end_point[0] - start_point[0]
    height = end_point[1] - start_point[1]
    
    if (width*1.1) < image.shape[0] and (height*1.1) < image.shape[1]:
        new_width = int(width * 1.1)
        new_height = int(height * 1.1)
        
        new_start_point = (start_point[0] - (new_width - width) // 2, start_point[1] - (new_height - height) // 2)
        new_end_point = (end_point[0] + (new_width - width) // 2, end_point[1] + (new_height - height) // 2)
    else: 
        new_start_point = start_point
        new_end_point = end_point
        
    image = cv2.rectangle(black_image, new_start_point, new_end_point, color=(255, 255, 255), thickness=-1)
    cv2.imwrite(f'{save_path}/mask_{img_name}_with_box.png', black_image)


# Reads the original .jpg image paths stored inside each .json file and collects them into a .txt file. Necessary for downstream scripts (ex. inpaint_images_fooocus.py).
def get_images_path(sampled_json_files, dataset_name):
    # Open the .json files of the sampled images. Store the names of singular .json files into the list "json_files".
    with open(sampled_json_files, 'r') as f:
        json_files = f.readlines()
    
    # create .txt file that will contain the original images paths.
    images_txt_file = open(os.path.join(config_utils_dir, f"source_images_path_{dataset_name}.txt"), 'w')
    
    # Load single .json files in the json_files list.
    for json_file in json_files:
        with open(json_file.strip(), 'r') as f:
            data = json.load(f)
        # Write the original .jpg image path into images_text_file.
        images_txt_file.write(data['path']+"\n")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default=os.path.join(output_dir, 'tags_and_masks'), help="Directory containing the output from the mask extraction component (.json and .npz files).")
    parser.add_argument('-n', '--num_images', type=int, default=None, help="Number of samples to construct the BtB collection for the pipeline.")
    parser.add_argument('-d', '--dataset_name', type=str, default="custom_dataset", help="Name of the source dataset used.")
    parser.add_argument('-m', '--save_dir_masks', type=str, default=os.path.join(output_dir, 'final_masks'), help="Directory to save the created masks.")
    parser.add_argument('-b', '--save_dir_bb', type=str, default=os.path.join(output_dir, 'final_boxes'), help="Directory to save the images with the green bounding box.")
    return parser 

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    masks_save_path = args.save_dir_masks
    # Create/define directory to save the clean .png masks.
    os.makedirs(masks_save_path, exist_ok=True)
    
    # Create/define directory to save the images with the green bounding boxes.
    bounding_box_save_path = args.save_dir_bb
    os.makedirs(bounding_box_save_path, exist_ok=True)
    
    # .txt file containing the labels of the best masks.
    best_mask_labels_file = open(os.path.join(config_utils_dir, f'best_mask_labels_{args.dataset_name}.txt'), 'a')
    
    # 1. SAMPLE N_IMAGES FROM THE INPUT DIR
    json_files_sampled = sample_n_images(args.input_dir, args.num_images, args.dataset_name)
    
    for json_file in tqdm(json_files_sampled, desc="Processing images", bar_format=f'\033[35m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):
        json_file = json_file.strip()
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
                
            image_name = os.path.basename(data['path']).replace(".jpg", "")

            # 2. UPDATE JSON FILE 
            update_json_file(data, json_file)
            with open(json_file, 'r') as file:
                data = json.load(file)
            
            # 3. GET TOP 3 MASKS (SMALL, MEDIUM, AND LARGE)
            best_masks = get_top_3_masks(json_file, masks_save_path)
            
            for size in ['small', 'medium', 'large']:
                if best_masks[size] is not None:
                    best_mask_labels_file.write(f"{image_name}_{size} - {data['boxes'][best_masks[size]]['label']}\n")
                    start_point = (data['boxes'][best_masks[size]]['box']['xmin'], data['boxes'][best_masks[size]]['box']['ymin'])
                    end_point = (data['boxes'][best_masks[size]]['box']['xmax'], data['boxes'][best_masks[size]]['box']['ymax'])
                    img = cv2.imread(data['path'])
                    if img is None:
                        print(f"Error reading image {data['path']}. Skipping bounding box drawing.")
                    else:
                        draw_bounding_box(img, start_point, end_point, img_name=image_name+f"_{size}", save_path=bounding_box_save_path)
        
        except Exception as e:
            print(f"    ERROR processing {json_file}: {str(e)}")
            traceback.print_exc()

    # 4. CREATE A TXT FILE CONTAINING THE SOURCE IMAGES PATH
    get_images_path(os.path.join(config_utils_dir, f'sampled_{len(json_files_sampled)}_images_{args.dataset_name}.txt'), args.dataset_name)
        
    print(f"Dataset {args.dataset_name} preprocessing completed.")