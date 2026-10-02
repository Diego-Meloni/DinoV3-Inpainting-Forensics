import os
from diffusers import StableDiffusionInpaintPipeline
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from time import time

# Setup paths relative to this script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_utils_dir = os.path.join(script_dir, 'config_and_utilities')
output_dir = os.path.join(project_root, 'dataset', 'manually_generated')

GUIDANCE_SCALE = 4.0 #control parameter for how strictly the model should follow the prompt.
INPAINT_STRENGTH = 0.9 #control parameter for how strongly the inpainting modifies the masked region.

class Inpainter():
    def __init__(self, device, seed, optimization=False):
        self.model_id = 'sd-legacy/stable-diffusion-inpainting'
        if optimization:
            self.model = StableDiffusionInpaintPipeline.from_pretrained(pretrained_model_name_or_path=self.model_id, torch_dtype=torch.float16).to(device)
            # self.model.enable_model_cpu_offload() #commented because it was causing problems with "accelerate", fix at a later date.
            self.model.enable_vae_slicing()
        else:
            self.model = StableDiffusionInpaintPipeline.from_pretrained(model_id=self.model_id, torch_dtype=torch.float16).to(device)
        
        # self.generator = torch.Generator(device).manual_seed(seed) #commented because it was causing problems with "cuda", fix at a later date.
            
    def inpaint_image(self, image, mask_image, prompt):
        output = self.model(
            prompt = prompt,
            image = image,
            mask_image = mask_image,
            strength = INPAINT_STRENGTH,
            guidance_scale = GUIDANCE_SCALE,
        )
        return output.images[0]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_paths', type=str, default=os.path.join(config_utils_dir, 'source_images_path_custom_dataset.txt'), help='Txt file containing the path of source images to be inpaint.')
    parser.add_argument('-m', '--masks_path', type=str, default=os.path.join(output_dir, 'final_masks'), help='Directory containing the masks of the images.')
    parser.add_argument('-p', '--prompt', type=str, default=os.path.join(output_dir, 'prompts_v2'), help='Directory containing the prompts generated for each images.')
    parser.add_argument('-o', '--save_path', type=str, default=os.path.join(output_dir, 'output_images_st_diff'), help="Directory to store the final inpainted images.")
    return parser 


def main(args):
    #create the Inpainter object responsible for inpainting using the 'sd-legacy/stable-diffusion-inpainting' HuggingFace model.
    inpainter = Inpainter('cuda', 2668419004769029052, optimization=True)
    
    #create the output directory.
    os.makedirs(args.save_path, exist_ok=True)
    
    #"images" is a list of all the name files in the prompt directory. ex. "000001_small.json".
    images = os.listdir(args.prompt)
    
    start = time()
    
    for img in tqdm(images, desc=f"Inpaiting images using {inpainter.model_id} ...", bar_format=f'\033[36m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):

        image_name = img[:img.rfind("_")] #returns the name of the image. ex. "000001".
        size = img[img.rfind("_")+1:img.find(".json")] #returns the size of the mask. ex. "medium".
        
        # open the .txt file containing the names of images sampled.
        images_paths = args.images_paths
        if not os.path.isabs(images_paths):
            images_paths = os.path.join(config_utils_dir, images_paths)

        with open(images_paths, "r") as f:
            paths = f.readlines()
        
        #create a list with the paths leading to the sampled images.
        images_names = [os.path.basename(path).replace(".jpg", "").strip() for path in paths]
        
        #open the directory containing the .json prompt files, store them in the prompts list. Find only prompts related to the current "img" in the for loop. ex. "000001_medium.json"
        with open(os.path.join(args.prompt, img), 'r') as file:
            prompts = json.load(file)

        #now we iterate over the prompts produced for each image.
        for i, prompt in enumerate(prompts):
            source = (Image.open(paths[images_names.index(image_name)].strip()).convert('RGB')).resize((512, 512)) #convert the image to RGB and resize it to (512, 512) so that it can be fed to the model.
            mask = (Image.open(os.path.join(args.masks_path, f"mask_{image_name}_{size}.png")).convert('L')).resize((512, 512)) #convert the mask to L and resize it to (512, 512) so that it can be fed to the model.
            
            replacement_idx = prompt.find("Replace with: ") #finds only the replacement target object inside the prompt.
            if replacement_idx ==-1:
                print(f"Unable to find the replacement in the image {image_name}_{size} for prompt {i}")
            else:
                prompt = prompt[replacement_idx+len("Replace with: "):]
                    
                save_name = f"{image_name}_{size}_{i%5}_stablediffusion_inpainted-0.png"
                
                #the server automatically saves the inpainted images in our output directory.
                inpainted_image = inpainter.inpaint_image(
                    image=source,
                    mask_image=mask,
                    prompt=prompt
                )
                
                output_path = os.path.join(args.save_path, save_name)
                inpainted_image.save(output_path, format='PNG')
                
                
    print(f"Inpainting ended in {(time()-start)/60} minutes.")
    
if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())