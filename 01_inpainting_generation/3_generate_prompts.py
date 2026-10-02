import os
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import json
from tqdm import tqdm
import re
from time import time

from torchvision.transforms.functional import InterpolationMode

# Setup paths relative to this script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_utils_dir = os.path.join(script_dir, 'config_and_utilities')
output_dir = os.path.join(project_root, 'dataset', 'manually_generated')

# RGB channel-wise mean and standard deviation of the ImageNet dataset. 
# Most vision models were pre-trained on ImageNet and expect input images to be normalized using these statistics.
# this centers each channel around zero, normalizing pixel values.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


#image pre-processing, outputs formatted numerical tensor usable by the model.
def build_transform(input_size):
    """
    Create image preprocessing pipeline: PIL Image -> normalized tensor (3, input_size, input_size).
    Example: 1920x1080 RGBA dog image -> (3, 448, 448) tensor with ImageNet normalization.
    Steps: RGB conversion -> resize -> to tensor -> ImageNet normalization.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    # T.Compose chains multiple transformations into a single object.
    transform = T.Compose([

        # T.Lambda converts the image into RGB from possible other modes (ex. RGBA, greyscale...)
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),   
        
        # T.resize resizes image to fixed square. BICUBIC interpolation mode computes weighted average of 4x4 nearest pixels to downscale the image.
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        
        # T.ToTensor converts PIL image to PyTorch tensor with pixel values in the [0,1] range. Tensor format: (channels, height, width)
        T.ToTensor(),
        
        #T .Normalize applies ImageNet normalization according to the specified mean and std values.
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform



def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    finds the best tilling layout to preserve an image's natural proportion when being downscaled to a square image.
    tilling layout: dividing the original image in a grid layout whose tiles' size match the target size (square of dim. input_size).
    the tiles are fed into the model as a batch, together with a global tile obtained by downscaling the original image to the target size.
    """
    # given a set of grid layout (ex. (2,3), (4,2)...) it searches for the one whose aspect ratio most closely matches the input image's aspect ratio.
    # aspect ratio = height/width.
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    # Loop through all possible grid layouts and find the one closest to original aspect ratio
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        # difference between candidate ratio and true image ratio.
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        # tiebreaker, if ratio_diff are equal prefer larger tile grids.
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


# splits the image into a grid of tiles to avoid squashing the whole image into a square of dim (input_size, input_size)
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # generates every valid grid layout in the form (columns, rows). Total number of tiles is between min_num and max_num.
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    
    # sort by total tile count.
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target ratio using the previously defined function.
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height such that the image can be resized to perfectly fit the chosen grid layout (target_aspect_ratio).
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image to exactly fill the chosen grid layout.
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    # compute the pixel coordinates of the box containing the different tiles in which the image is divided into by the grid layout.
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image into rectangular subregions defined by the tile boxes. Add them to the processed_images list.
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    # if use_thumbnail=True a global overview tile is added at the end. Gives the model a fine detailed view (tiles) and compressed general overview (global overview tile).
    if use_thumbnail and len(processed_images) != 1:
        # simply aggressively downsize original image to obtain global overview tile.
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# returns a batch of tensors, one per tile in which we divided the image, to feed to the model.
def load_image(image_file, input_size=448, max_num=6):
    # PIL method opening the image file. Convert to RGB mode.
    image = Image.open(image_file).convert('RGB')
    # create a transformer object that will return PyTorch tensors.
    transform = build_transform(input_size=input_size)
    # call the dynamic_preprocess() function to split the image in the gird layout best representing the original image's aspect ratio.
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    # call the build_transform() function, applying the pre-processing pipeline to the loaded image. Turns the images into normalized PyTorch tensors.
    pixel_values = [transform(image) for image in images]
    # takes the list of tile tensors of shape (3,448,448) and combines them into a new tensor of shape (N_tiles, 3, 448, 448).
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def main(args):
    # LOAD THE VISUAL LANGUAGE MODEL 
    model_path = 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda() #put model in evaluation mode.
    # load tokenizer associated to the used vision-language model. Converts strings (our instructions to the model) into token IDs the model can process.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # dictionary dictating the text generation aka token sampling strategies of the language model. do_sample=True enables probabilistic token sampling, doesn't always sample token with highest probability.
    generation_config = dict(num_beams=1,max_new_tokens=512,do_sample=True,)
    
    # LOAD THE TXT FILE WITH THE LABELS FOR EACH IMAGE AND THE INPUT IMAGES
    # Format: "image_name_size - label" (e.g., "dog_image_small - dog")
    # Smart path resolution: if not absolute path, search in config_utils_dir
    labels_file = args.labels_file
    if not os.path.isabs(labels_file):
        labels_file = os.path.join(config_utils_dir, labels_file)
    
    with open(labels_file, 'r') as f:
        images_and_labels = f.readlines()
    imgs = [img.split(" - ")[0] for img in images_and_labels] #extract the image names.
    labels = [img.split(" - ")[1] for img in images_and_labels] #extract the image labels.

    # returns a list of all the names of the images in the directory.
    images = os.listdir(args.bounding_box_dir)
    # by eliminating "mask_" and "with_box.png" the names of the images pulled from the bounding box directory will now match the names of the images in the best_mask_labels.txt file.
    # ex. "mask_000001_small_with_box.png" --> "000001_small". Name in the best_mask_labels.txt file: "000001_small".
    images = [image.replace("mask_", "").replace("_with_box.png", "") for image in images]
    
    responses = [] 

    # iterates over the "images" list, containing the stripped bounding box image names. ex. "000001_small".
    for img in tqdm(images, desc="Generating prompts", bar_format=f'\033[34m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):
        img_name = img
        # "imgs" is the list containing the image_names pulled from best_mask_labels.txt. Finds the index of the "img_name" string in the "imgs" list and uses it to pull the corresponding label from the "labels" list, also pulled from best_mask_labels.txt.
        label = labels[imgs.index(img_name)]
        # call the load_image() function on the green bounded box image. Returns PyTorch tensor (N_tiles, 3, 448, 448).
        pixel_values = load_image(os.path.join(args.bounding_box_dir, f"mask_{img}_with_box.png"), max_num=6).to(torch.bfloat16).cuda()

        question = f"You are an expert of image inpainting. I want to replace the {label} inside the green box with something else, while keeping the rest of the image intact. The replacement should have similar size to the original object, as it will be inpainted on the original image. The resulting image should stay semantically coherent after the change. Can you suggest me what this replacement object could be? Please explain your chain of though for the rationale used for the proposal. Use the following template for answering, keeping the proposed replacement a very short sentence (at most four words): Rationale: XXX \n Replace with: YYY"
        
        prompts = []
        
        # PROMPT GENERATION
        
        # generate args.num_prompt prompts per image suggesting substitutions of the box bounded & labeled object in the image.
        for i in range(args.num_prompt):
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            # the responses list's elements are strings containing the image name (format "000001_small") plus the model's response.
            responses.append(f"{img_name} - {response}")
            prompts.append(f"{response}")
        # save the prompts as {img_name}.json arrays in the specified output directory.
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"{img_name}.json"), 'w') as out_file:
                json.dump(prompts, out_file, indent=2)
                
        

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default=os.path.join(output_dir, 'prompts'), help='Directory to store the json file with the generated prompt for each image.')
    parser.add_argument('-b', '--bounding_box_dir', type=str, default=os.path.join(output_dir, 'final_boxes'), help='Directory containing the images with the green bounding box.')
    parser.add_argument('-l', '--labels_file', type=str, default=os.path.join(config_utils_dir, 'best_mask_labels_custom_dataset.txt'), help='Txt file containing the labels for each image')
    parser.add_argument('-n', '--num_prompt', type=int, default=5, help='Number of prompts to generate for each image. Default is 5.')
    return parser 

if __name__ == '__main__':
    parser = get_parser()
    start = time()
    main(parser.parse_args())
    print(f"Prompt generation ended in {(time()-start)/60} minutes.")