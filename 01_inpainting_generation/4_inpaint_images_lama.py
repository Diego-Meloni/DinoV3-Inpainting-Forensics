import os
from simple_lama_inpainting import SimpleLama
import json
import argparse
import torch

# --- TRUCCO PER AGGIRARE IL BUG DI PYTORCH 2.x ---
_original_load = torch.jit.load
torch.jit.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'map_location': 'cpu'})
# -------------------------------------------------

from PIL import Image
from tqdm import tqdm
from time import time

# Setup paths relative to this script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_utils_dir = os.path.join(script_dir, 'config_and_utilities')
output_dir = os.path.join(project_root, 'dataset', 'manually_generated')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_paths', type=str, default=os.path.join(config_utils_dir, 'source_images_path_custom_dataset.txt'), help='Txt file containing the path of source images to be inpaint.')
    parser.add_argument('-m', '--masks_path', type=str, default=os.path.join(output_dir, 'final_masks'), help='Directory containing the masks of the images.')
    parser.add_argument('-o', '--save_path', type=str, default=os.path.join(output_dir, 'output_images_lama'), help="Directory to store the final inpainted images.")
    return parser 


def main(args):
    #create the SimpleLama object responsible for inpainting.
    simple_lama = SimpleLama()
    
    #create the output directory.
    os.makedirs(args.save_path, exist_ok=True)
    
    #"images" is a list of all the name files in the mask directory. ex. "mask_000001_small.png".
    images = os.listdir(args.masks_path)
        
        
    start = time()
    
    for img in tqdm(images, desc='Inpainting images using SimpleLama...', bar_format=f'\033[36m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):
        for substring in ['_small', '_medium', '_large']:
            pos = img.rfind(substring)
            if pos != -1:
                image_name = img[5: pos]
                size = img[pos+1: img.rfind('.png')]
                break
        

        images_paths = args.images_paths
        if not os.path.isabs(images_paths):
            images_paths = os.path.join(config_utils_dir, images_paths)

        #open the .txt file containing the names of images sampled.    
        with open(images_paths, "r") as f:
            paths = f.readlines()
            
        #create a list with the paths leading to the sampled images.
        images_names = [os.path.basename(path).replace(".jpg", "").strip() for path in paths]
        
        source = (Image.open(paths[images_names.index(image_name)].strip()).convert('RGB'))
        mask = (Image.open(os.path.join(args.masks_path, img)).convert('L'))
        
        inpainted_image = simple_lama(source, mask)
        
        save_name = f'{image_name}_{size}_lama_inpainted-0.png'
        
        output_path = os.path.join(args.save_path, save_name)
        
        inpainted_image.save(output_path, format='PNG') 
                
    print(f"Inpainting ended in {(time()-start)/60} minutes.")
    
if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())