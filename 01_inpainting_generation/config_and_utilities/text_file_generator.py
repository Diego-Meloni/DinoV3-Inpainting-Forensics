import os
import argparse

def text_file_generator(input_dir, output_file):
    # Ensure output folder exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    count = 0
    with open(output_file, 'w') as file:
        for image in os.listdir(input_dir):
            if image.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Write the absolute path of the image
                image_path = os.path.abspath(os.path.join(input_dir, image))
                file.write(f'{image_path}\n')
                count += 1
                
    print(f">>> Found {count} images.")
    print(f">>> File successfully generated at: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a .txt file with the paths of the input images.")
    parser.add_argument('-i', '--input_dir', type=str, 
                        default="01_inpainting_generation/input_images",
                        help="Folder containing input images")
    # Default name is input_images_paths.txt
    parser.add_argument('-o', '--output_file', type=str, 
                        default="01_inpainting_generation/config_and_utilities/input_images_paths.txt", 
                        help="Path and name of the output .txt file")
    
    args = parser.parse_args()

    text_file_generator(args.input_dir, args.output_file)