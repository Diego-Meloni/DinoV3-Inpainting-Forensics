import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import json
from tqdm import tqdm
import re
from time import time
import gc

from torchvision.transforms.functional import InterpolationMode

# Setup paths relative to this script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_utils_dir = os.path.join(script_dir, 'config_and_utilities')
output_dir = os.path.join(project_root, 'dataset', 'manually_generated')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    Create image preprocessing pipeline: PIL Image -> normalized tensor (3, input_size, input_size).
    Example: 1920x1080 RGBA dog image -> (3, 448, 448) tensor with ImageNet normalization.
    Steps: RGB conversion -> resize -> to tensor -> ImageNet normalization.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find best grid layout for image patches. Input: aspect ratio + target grid options.
    Output: (cols, rows) tuple. Example: 1600x1000 image (1.6 ratio) -> (2, 1) grid.
    Logic: Matches original aspect ratio; if tied, prefers layout accommodating image area.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    # Loop through all possible grid layouts and find the one closest to original aspect ratio
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        # If this layout is closer to original aspect ratio, use it
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        # If tied on aspect ratio, prefer the layout that accommodates the image better
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """
    Split image into grid of patches preserving aspect ratio. Input: PIL Image any size.
    Output: List of (448x448) patches + optional thumbnail.
    Example: 1600x1000 image -> 2 patches (2x1 grid) + 1 thumbnail = 3 total images.
    Algorithm: Find best grid -> resize image -> crop patches -> add thumbnail if needed.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate all possible grid layouts (min_num to max_num patches total)
    # Example with min_num=1, max_num=6: (1,1), (2,1), (1,2), (2,2), (2,3), ...
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the grid layout that best matches original aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate total dimensions for the grid (each patch is image_size x image_size)
    # Example: (2, 1) grid → target_width=896, target_height=448
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize image to fit grid layout
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    # Extract individual patches from the grid
    for i in range(blocks):
        # Calculate crop box coordinates for this patch
        # For 2x1 grid: block 0 gets (0,0,448,448), block 1 gets (448,0,896,448)
        col = i % (target_width // image_size)
        row = i // (target_width // image_size)
        box = (
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    # If use_thumbnail=True and image wasn't already single patch, add downsampled full image
    # This gives model context about full image for better understanding
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    """
    Load and preprocess image for vision model. Input: image file path.
    Output: Tensor (num_patches, 3, 448, 448). Example: 'dog_image_small_with_box.png' -> (3, 3, 448, 448).
    Steps: Load -> split into patches + thumbnail -> normalize -> stack into tensor.
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    # Split image into patches (preserves aspect ratio)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    # Apply normalization to each patch
    pixel_values = [transform(image) for image in images]
    # Stack patches into single tensor: (num_patches, 3, H, W)
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def main(args):
    """
    Generate inpainting replacement suggestions using Qwen2-VL vision language model.
    Steps: Load model -> For each image: extract label -> process with processor -> generate N suggestions -> save to JSON.
    Output: JSON files per image with replacement suggestion list.
    """
    # STEP 0: Aggressive cache cleaning BEFORE loading model
    torch.cuda.empty_cache()
    gc.collect()
    
    # STEP 1: Load Qwen2-VL model
    print(f"Loading Qwen/Qwen2-VL-2B-Instruct...")
    print("(This downloads ~4GB on first run, then cached locally)")
    
    # Configure 8-bit quantization to reduce model size from ~8GB (float32) to ~1GB
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,  # Quantize the quantization constants
        bnb_8bit_quant_type="nf4"  # Use NormalFloat4 for best quality at 8-bit
    )
    
    model_path = 'Qwen/Qwen2-VL-2B-Instruct'
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("✓ Model loaded successfully (8-bit quantized)")
    
    # STEP 1.5: Clean GPU cache after loading model to free fragmented memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # STEP 2: Load the labels file
    try:
        labels_file = args.labels_file
        if not os.path.isabs(labels_file):
            labels_file = os.path.join(config_utils_dir, labels_file)
        
        with open(labels_file, 'r') as f:
            images_and_labels = f.readlines()
        imgs = [img.split(" ")[0] for img in images_and_labels]
        labels = [img.split(" ")[1].strip() for img in images_and_labels]
        print(f"✓ Loaded labels for {len(imgs)} images")
    except FileNotFoundError:
        print(f"✗ Labels file not found: {args.labels_file}")
        return

    # STEP 3: Get list of images with bounding boxes to process
    try:
        images = os.listdir(args.bounding_box_dir)
        images = [image.replace("mask_", "").replace("_with_box.png", "") for image in images]
        print(f"✓ Found {len(images)} images to process")
    except Exception as e:
        print(f"✗ Error reading images: {str(e)}")
        return
    
    if len(images) == 0:
        print("✗ No images found in bounding_box_dir")
        return
    
    # STEP 4: Main loop
    success_count = 0
    for img in tqdm(images, desc="Generating prompts", bar_format=f'\033[34m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):
        try:
            img_name = img
            
            # Find label for this image
            if img_name not in imgs:
                print(f"⚠ Warning: {img_name} not found in labels file, skipping")
                continue
                
            label = labels[imgs.index(img_name)].strip()
            
            torch.cuda.empty_cache()
            gc.collect()

            # Load image
            image_path = os.path.join(args.bounding_box_dir, f"mask_{img}_with_box.png")
            if not os.path.exists(image_path):
                print(f"⚠ Warning: Image not found {image_path}, skipping")
                continue
                
            image = Image.open(image_path).convert('RGB')

            # --- IMPORTANT: Limit dimensions to prevent GPU from running out of memory ---
            # Note: this operation keeps the original proportion of the image
            max_dimension = 1024
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension))
            # -----------------------------------------------------------------------------
            
            # Craft prompt (short and simple for Qwen)
            question = f"""You are an expert of image inpainting and object replacement design.
                            Context: There is a {label} inside the green bounding box. I need to suggest coherent replacement alternatives.
                            Your task:
                            1. Carefully analyze the image (room style, colors, surrounding objects, lighting)
                            2. Consider the object's size, shape, and material
                            3. Suggest diverse and creative replacements - different each time
                            4. Explain your specific reasoning based on visual context
                            Requirements:
                            - Each replacement must be different from previous ones
                            - Consider: style coherence, practical function, visual balance, color harmony
                            - Size similar to original object
                            - Replacements should be realistic but creative
                            Format:
                            Rationale: [2-3 specific sentences analyzing THIS image and why this replacement works]
                            Replace with: [object name, max 4 words]"""
            
            prompts = []
            
            # STEP 5: Generate N prompts
            for i in range(args.num_prompt):
                try:
                    # Format message for Qwen2-VL
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                },
                                {
                                    "type": "text",
                                    "text": question
                                }
                            ],
                        }
                    ]
                    
                    # Process inputs using official method
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device)
                    
                    # Increase diversity with each prompt
                    temperature = 0.7 + (i * 0.4)  # 0.7, 1.1, 1.5, 1.9...
                    top_p = 0.8 + (i * 0.1)  # 0.8, 0.9, 1.0
                    top_k = 50 - (i * 10)  # 50, 40, 30...

                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs, 
                            max_new_tokens=128,  # Allow more space for creativity
                            do_sample=True, 
                            temperature=min(temperature, 2.0),
                            top_p=min(top_p, 1.0),
                            top_k=max(top_k, 10),
                            repetition_penalty=1.2  # Penalizes repetition
                        )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    prompts.append(response_text.strip())
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"  ⚠ Error generating prompt {i+1}: {str(e)}")
                    continue
            
            if len(prompts) == 0:
                print(f"✗ {img_name}: No prompts generated")
                continue
            
            # STEP 6: Save to JSON
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"{img_name}.json"), 'w') as f:
                json.dump(prompts, f, indent=2)
            
            # Clean up image tensor and cache after processing this image
            del image
            torch.cuda.empty_cache()
            gc.collect()
            
            # print(f"✓ {img_name}: {len(prompts)} prompts saved")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {img}: {str(e)}")
            continue
    
    print(f"\n✓ Completed: {success_count}/{len(images)} images processed successfully")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--output_dir', type=str, default=os.path.join(output_dir, 'prompts_v2'), help='Directory to store the json file with the generated prompt for each image.')
    parser.add_argument('-b','--bounding_box_dir', type=str, default=os.path.join(output_dir, 'final_boxes'), help='Directory containing the images with the green bounding box.')
    parser.add_argument('-l','--labels_file', type=str, default=os.path.join(config_utils_dir, 'best_mask_labels_custom_dataset.txt'), help='Txt file containing the labels for each image')
    parser.add_argument('-n','--num_prompt', type=int, default=5, help='Number of prompts to generate for each image. Default is 5.')
    return parser 

if __name__ == '__main__':
    # Parse command-line arguments and run main pipeline
    parser = get_parser()
    start = time()
    main(parser.parse_args())
    elapsed_minutes = (time() - start) / 60
    print(f"Prompt generation ended in {elapsed_minutes:.2f} minutes.")