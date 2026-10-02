#!/usr/bin/env python3

import argparse
import json
import os
import sys
import importlib.util

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForMaskGeneration, AutoModelForZeroShotObjectDetection, pipeline
from ram.models import ram
from ram import get_transform

# Setup paths relative to this script location
script_dir = os.path.dirname(os.path.abspath(__file__))           # 01_inpainting_generation/
project_root = os.path.dirname(script_dir)                        # Beyond-the-Brush/
config_utils_dir = os.path.join(script_dir, 'config_and_utilities')
output_dir = os.path.join(project_root, 'dataset', 'manually_generated')
input_images_dir = os.path.join(script_dir, 'input_images')

# Import text_file_generator explicitly from config_and_utilities
text_file_gen_path = os.path.join(config_utils_dir, 'text_file_generator.py')
spec = importlib.util.spec_from_file_location("text_file_generator", text_file_gen_path)
text_file_gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text_file_gen_module)
text_file_generator = text_file_gen_module.text_file_generator

# STAGE 1: TAGGING - Recognize what objects are in the image
class Tagger:
    def __init__(self, image_size, device, ram_model_path):
        self.transform = get_transform(image_size=image_size)
        self.model = ram(pretrained=ram_model_path, # './weights/ram_swin_large_14m.pth',
                        image_size=image_size,
                        vit='swin_l')
        self.model.eval()  # Activate evaluation mode, we need the model to do inference
        self.model = self.model.to(device)
        self.device = device

    def __call__(self, images):
        # Convert PIL images to tensor and move to GPU
        image_tensor = torch.stack(tuple(self.transform(i).to(self.device) for i in images)) # Concatenates images along a new dimension, 4D tensor (batch_size, 3, image_size, image_size)

        with torch.no_grad(): # Disable gradient computation, we only need model inference
            tags, tags_chinese = self.model.generate_tag(image_tensor) # Generate tags in format: ex. "dog | car | house"
        return [x.split(' | ') for x in tags] # Return a list of tags, converts from string of type "x | y | z" to a list [x, y, z]

# STAGE 2: DETECTION - Locate objects using bounding boxes
class Detector:
    def __init__(self, device):
        model_id = "IDEA-Research/grounding-dino-tiny"

        self.object_detector = pipeline(model=model_id, task="zero-shot-object-detection", device=device) # Pipeline object of the "transformers" library, wraps needed preprocessing, postprocessing and the model and performs the specified task
        self.processor = AutoProcessor.from_pretrained(model_id) #NOT USED
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device) #NOT USED
    
    
    def __call__(self, images, tags):
        # Pipeline expects as inputs a list of dics, each with an "image" PIL image and "candidate_labels" list of strings (the tags).
        inputs = [  
            {
                "image": i,
                "candidate_labels": [t + '.' for t in ts] # Grounding DINO was trained on tags with full stops at the end.
            }
            for i, ts in zip(images, tags) # Zip pairs each image with its corresponding list of tags.
        ]
        # Returns for each image a list of detected objects with their confidence score, label and box coordinates.
        return self.object_detector(inputs) 


# Returns for each image a list of detected objects with their confidence score, label and box coordinates.
class Segmenter:
    def __init__(self, device, batch_boxes=4):
        segmenter_id = "facebook/sam-vit-base"

        self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device) # HuggingFace class from transformers library for mask-generating models.
        self.processor = AutoProcessor.from_pretrained(segmenter_id) # Preprocessing pipeline transforming input data in the required format for the SAM model.
        self.device = device
        self.batch_boxes = batch_boxes # Max number of bounding boxes processed per SAM forward pass.

    def __call__(self, image, boxes):
        # Computes number of batches of "prompts" (bounding boxes).
        n_batches = len(boxes) // self.batch_boxes + (1 if len(boxes) % self.batch_boxes != 0 else 0) 

        all_masks = np.empty((len(boxes), 3, image.size[1], image.size[0]), dtype=bool) #pre-allocates the full output array, dimensions (number of boxes, 3 (SAM always generates 3 candidate masks), height of original image, width of original image)
        
        print(f"Running {n_batches} batches")
        for i in range(n_batches):
            input_boxes = [boxes[i * self.batch_boxes : (i + 1) * self.batch_boxes]] #line of code taking as input quantized "batch_boxes" boxes to feed forward to SAM.
            inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device) #converts PIL images and bounding box coordinates into tensors expected by SAM. Sends them to the device. Returns PyTorch tensors (return_tensors="pt").
            with torch.no_grad():
                outputs = self.segmentator(**inputs) #runs the SAM forward pass.
                
            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks, # Masks raw logit masks as SAM's internal resolution.
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes
            )[0] # Upsamples masks back to original image resolution, then binarizes using a threshold the logit values ([0,1]) to produce boolean masks.
            # [0] indexes first image in sub-batch, since SAM generates 3 masks with different resolutions that all get upsampled back to original image resolution.

            all_masks[i * self.batch_boxes : min((i + 1) * self.batch_boxes, len(boxes)), :, :, :] = masks.cpu().numpy() #.cpu() moves tensors from the gpu tp the cpu, .numpy() converts it to a Numpy array.
            # Results stored in the pre-allocated array "all_masks"
            
        return all_masks


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-o', '--output-path', type=str, default=os.path.join(output_dir, 'tags_and_masks'))
    parser.add_argument('-tm', '--top-masks', type=int, default=None)
    parser.add_argument('-r', '--ram-model-path', required=True)
    parser.add_argument('-i', '--input_directory', type=str, default=input_images_dir)
    parser.add_argument('-t', '--text_file', type=str, default=os.path.join(config_utils_dir, 'input_images_paths.txt'))

    return parser


# Iterator that segments a list into fixed size chunks. 
# Allows main loop to process large list of images in batches without the need to load everything at once.
class Batched: 
    def __init__(self, base, batch_size):
        self.base = base
        self.batch_size = batch_size
        self.next_batch = 0

    def __len__(self):
        base_len = len(self.base)
        if base_len % self.batch_size != 0:
            return base_len // self.batch_size + 1
        else:
            return base_len // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch >= len(self):
            raise StopIteration

        items = self.base[self.next_batch * self.batch_size : (self.next_batch + 1) * self.batch_size]
        self.next_batch += 1

        return items


# Like "Batch", operates simultaneously on parallel lists. Iterates over "image_paths" and "all_tags" in detection phase (Grounding DINO model).
class BatchedZip: 
    def __init__(self, base, batch_size):
        self.base = base
        self.base_len = len(self.base[0])
        for x in self.base[1:]:
            assert len(x) == self.base_len
        self.batch_size = batch_size
        self.next_batch = 0

    def __len__(self):
        if self.base_len % self.batch_size != 0:
            return self.base_len // self.batch_size + 1
        else:
            return self.base_len // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch >= len(self):
            raise StopIteration

        items = [x[self.next_batch * self.batch_size : (self.next_batch + 1) * self.batch_size] for x in self.base]
        self.next_batch += 1

        return items


def load_all_json(parent_path, concat=False):
    result = []
    for item in sorted(os.listdir(parent_path)):
        with open(os.path.join(parent_path, item), 'r') as stream:
            result.append(json.load(stream)) #json.load(stream) transforms a .json file into a python object.

    # each tags file contains a list of tag lists, one per image in the batch.
    
    # Flatten aforementioned list of lists into a single list.
    if concat: 
        return [item for sub in result for item in sub]
    else:
        return result

# 3 STAGES PIPELINE
def main(args): 
    # Initialize the three AI models
    tagger = Tagger(384, args.device, args.ram_model_path)
    detector = Detector(args.device)
    segmenter = Segmenter(args.device, batch_boxes=1)

    #create the .txt file containing the paths of the input images from the input_directory specified in the parser arguments.
    text_file_generator(args.input_directory, args.text_file)
    
    # Load image paths from file
    with open(args.text_file, 'r') as stream:
        image_paths = [x.strip() for x in stream.readlines()]

    # Create output directories
    os.makedirs(os.path.join(args.output_path, 'tmp', 'tags'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'tmp', 'boxes'), exist_ok=True)
    
# STAGE 1: TAGGING
    
    for i, ps in enumerate(tqdm(Batched(image_paths, 16))): # tqdm wraps the "Batched" iterator to show a progress bar
        tags_path = os.path.join(args.output_path, 'tmp', 'tags', f'{i:07d}-tags.json') # path for .json tags output file.
        if not os.path.exists(tags_path): # makes the script resumable: doesn't recompute tags for already-computed batches.
            images = [Image.open(p).convert("RGB") for p in ps] # images are opened and turned into RGB, model requires 3-channel RGB.
            tags = tagger(images) # produce tags using tagger model.
            with open(tags_path, 'w') as stream:
                json.dump(tags, stream)

    del tagger # free RAM's GPU memory before loading next model.

# STAGE 2: DETECTION

    all_tags = load_all_json(os.path.join(args.output_path, 'tmp', 'tags'), concat=True) # load .json files containing the tag lists + image paths.
    print("Tags OK")
    
    for i, (ps, tags) in enumerate(tqdm(BatchedZip((image_paths, all_tags), 2))): # BatchedZip keeps image paths and their tags synchronized.
        boxes_path = os.path.join(args.output_path, 'tmp', 'boxes', f'{i:07d}-boxes.json') # path for .json boxes output file.
        if not os.path.exists(boxes_path):
            images = [Image.open(p).convert("RGB") for p in ps]
            boxes = detector(images, tags) # produce bounding boxes + confidence scores using detector model.
            
            with open(boxes_path, 'w') as stream:
                json.dump(boxes, stream)

    del detector # free GroundingDINO's GPU memory before loading the next model.

    all_boxes = load_all_json(os.path.join(args.output_path, 'tmp', 'boxes'), concat=True) # load .json files containing the bounding boxes + image paths.
    print("Boxes OK")

    # STAGE 3: SEGMENTATION

    # SAM is called per single image and not per batch, each image can have varying number of boxes.
    for i, (p, tags, boxes) in enumerate(zip(tqdm(image_paths), all_tags, all_boxes)):
        result_path = os.path.join(args.output_path, f'{i:07d}.json')
        if not os.path.exists(result_path):
            image = Image.open(p).convert("RGB") # convert image to RGB.
            
            # truncate the number of masks produced by truncating the bounding boxes per image.
            if args.top_masks is not None: 
                boxes = boxes[:args.top_masks]
            in_boxes = [[m['box']['xmin'], m['box']['ymin'], m['box']['xmax'], m['box']['ymax']] for m in boxes] # extracts boxes in the format [xmin, ymin, xmax, ymax].
            segmented_masks = segmenter(image, in_boxes) # produce masks using segmenter model.
            
            result = {
                'path': p,
                'tags': tags,
                'boxes': [
                    { 'score': box['score'], 'label': box['label'][:-1], 'box': box['box'] }
                    for box in boxes
                ]
            }
            
            with open(result_path, 'w') as stream:
                json.dump(result, stream, indent=2) # saves result dictionary as a .json file.
            masks_path = os.path.join(args.output_path, f'{i:07d}.npz')
            np.savez_compressed(masks_path, masks=segmented_masks) # saves boolean mask arrays in a compressed Numpy archive .npz.
    
        torch.cuda.empty_cache() # releases cached GPU memory after each image.


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
