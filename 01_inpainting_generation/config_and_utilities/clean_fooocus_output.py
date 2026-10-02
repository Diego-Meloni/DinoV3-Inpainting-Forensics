import os

def clean_error_files():
    # Dynamically resolve paths relative to this script location
    script_dir = os.path.dirname(os.path.abspath(__file__))   # .../config_and_utilities/
    generation_dir = os.path.dirname(script_dir)              # .../01_inpainting_generation/
    project_root = os.path.dirname(generation_dir)            # .../Beyond-the-Brush/
    
    # Define the directory containing the Fooocus outputs using the resolved project root
    target_dir = os.path.join(project_root, 'dataset', 'manually_generated', 'output_images_fooocus')
    
    deleted_count = 0
    print("Scanning for extensionless error files...")

    # Iterate through all files in the target directory
    for file_name in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file_name)
        
        # Ensure we are only operating on files, not directories
        if os.path.isfile(file_path):
            # os.path.splitext splits the file path into a root and an extension.
            # If the extension is an empty string, it's one of our API error files.
            _, file_extension = os.path.splitext(file_name)
            
            if file_extension == '':
                os.remove(file_path)
                print(f"Deleted error file: {file_name}")
                deleted_count += 1
                
    print("-" * 30)
    print(f"Cleanup complete. Total error files removed: {deleted_count}")

if __name__ == "__main__":
    clean_error_files()