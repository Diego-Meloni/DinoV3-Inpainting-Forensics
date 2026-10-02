import requests
import json
import os
import argparse
import subprocess
from datetime import datetime
from tqdm import tqdm
from time import time

# Setup paths relative to this script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_utils_dir = os.path.join(script_dir, 'config_and_utilities')
output_dir = os.path.join(project_root, 'dataset', 'manually_generated')
# address of locally running Fooocus API server.
HOST = 'http://127.0.0.1:8889'

# GLOBAL PARAMETERS FOR FOOOCUS
GUIDANCE_SCALE = 4.0 # control parameter for how strictly the model should follow the prompt.
INPAINT_STRENGTH = 0.9 # control parameter for how strongly the inpainting modifies the masked region.
IMAGE_SEED = 2668419004769029052

# script doesn't run the model directly, it acts as a client that calls a Fooocus API running on a remote server.
# INPUTS: original images, binary .png masks and generated prompts. 
# OUTPUT: inpainted images. Format (.jpg or .png) depends on the Fooocus API server.

# core communication function
def inpaint_outpaint(params: dict, input_image: bytes, input_mask: bytes = None) -> dict:
    # requests.post() sends an HTTP POST request to the specified url.
    response = requests.post(
    url=f"{HOST}/v1/generation/image-inpaint-outpaint",
    data=params, # generation parameters (prompt and previously defined control parameters such as guidance_scale).
    files={"input_image": input_image,
    "input_mask": input_mask})
    return response.json() #.json() ensures that it returns the response as a Python dictionary containing info on the inpainted images.
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_paths', type=str, default=os.path.join(config_utils_dir, 'source_images_path_custom_dataset.txt'), help='Txt file containing the path of source images to be inpaint.')
    parser.add_argument('-m', '--masks_path', type=str, default=os.path.join(output_dir, 'final_masks'), help='Directory containing the masks of the images.')
    parser.add_argument('-p', '--prompt', type=str, default=os.path.join(output_dir, 'prompts_v2'), help='Directory containing the prompts generated for each images.')
    parser.add_argument('-o', '--save_path', type=str, default=os.path.join(output_dir, 'output_images_fooocus'), help="Directory to store the final inpainted images.")
    parser.add_argument('-f', '--fooocus_api_dir', type=str, default=None, help='Location of Fooocus-API directory.') # NOT USED
    parser.add_argument('-j', '--save_jsons', type=bool, default=True, help='Save .json files describing inpainted images or not.')
    parser.add_argument('-s', '--save_in_directory', type=bool, default=True, help='Save .png images in the specified directory.')
    return parser 

def main(args):
    # create the output directory.
    os.makedirs(args.save_path, exist_ok=True)
    
    # "images" is a list of all the name files in the prompt directory. ex. "000001_small.json".
    images = os.listdir(args.prompt)
    
    start = time()
    
    for img in tqdm(images, desc="Inpaiting images", bar_format=f'\033[36m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):

        image_name = img[:img.rfind("_")] # returns the name of the image. ex. "000001".
        size = img[img.rfind("_")+1:img.find(".json")] # returns the size of the mask. ex. "medium".
        
        # open the .txt file containing the names of images sampled.

        images_paths = args.images_paths
        if not os.path.isabs(images_paths):
            images_paths = os.path.join(config_utils_dir, images_paths)

        with open(images_paths, "r") as f:
            paths = f.readlines()
        
        # create a list with the paths leading to the sampled images.
        images_names = [os.path.basename(path).replace(".jpg", "").strip() for path in paths]
        
        # open the directory containing the .json prompt files, store them in the prompts list. Find only prompts related to the current "img" in the for loop. ex. "000001_medium.json"
        with open(os.path.join(args.prompt, img), 'r') as file:
            prompts = json.load(file)

        # now we iterate over the prompts produced for each image.
        for i, prompt in enumerate(prompts):
            source = open(paths[images_names.index(image_name)].strip(), "rb").read() #read the images in binary mode to produce byte objects that can be fed to the inpainting model.
            mask = open(os.path.join(args.masks_path, f"mask_{image_name}_{size}.png"), "rb").read()
            
            replacement_idx = prompt.find("Replace with: ") #finds only the replacement target object inside the prompt.
            if replacement_idx ==-1:
                print(f"Unable to find the replacement in the image {image_name}_{size} for prompt {i}")
            else:
                prompt = prompt[replacement_idx+len("Replace with: "):]
                    
                save_name = f"{image_name}_{size}_{i%5}_fooocus_inpainted"
                
                # the server automatically saves the inpainted images in our output directory.
                result = inpaint_outpaint(
                    params={
                        "prompt": prompt,
                        "async_process": False, 
                        "save_name": f"{args.save_path}/{save_name}",
                        "guidance_scale": GUIDANCE_SCALE,
                        "inpaint_strength": INPAINT_STRENGTH,
                        "image_seed": IMAGE_SEED,
                    },
                    input_image=source,
                    input_mask=mask)
                # returns a list containing only 1 dictionary.
                
                
                if result[0]['finish_reason'] == "SUCCESS" and args.save_in_directory:
                    # get the image URL from result, which is a list containing a dictionary.
                    image_url = result[0]['url']
                    # request the image present at that specific server URL.
                    image_response = requests.get(f"{HOST}{image_url}" if image_url.startswith('/') else image_url)
                    save_file = os.path.join(args.save_path, f"{save_name}")
                    # save it in the directory specified in args.save_path.
                    with open(save_file, 'wb') as f:
                        f.write(image_response.content)
                else:
                    print(f'Inpainting failed for {save_name}: {result}')
                
                if args.save_jsons:
                    jsons_directory = os.path.join(output_dir, 'inpainted_images_jsons')
                    os.makedirs(jsons_directory, exist_ok=True)
                    with open(os.path.join(jsons_directory, save_name), 'w') as out_file:
                        json.dump(result, out_file, indent=2)
                
    print(f"Inpainting ended in {(time()-start)/60} minutes.")
    
if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
