# Beyond-the-Brush - Running procedure

## Prerequisites Setup

### Environment and Requirements.txt file modifications

To begin with the installation and configuration process some adjustments are needed:
- Create the virtual environment as defined by the authors in their github repo, but use python 3.10.20 instead of the original one to avoid dependencies errors.
- Before installing the libraries in the requirements.txt file, we need to modify it in the following way:
   - Change the version of the `accelerate` library to 1.13.0
   - Remove the `clip` library (We will install and updated version)
   - Remove `nvidia_nccl-cu12` (linux library, creates problems)
   - Remove `torch` and `torchvision` (We need the right version for our GPU, see notes later)
   - Remove `triton` (linux library)
   - Execute the following commands:
      ```bash
      # 1. Install the necessary packages to compile CLIP
      pip install "setuptools<70.0.0" wheel

      # 2. Install CLIP forcing the usage of the above installed packages
      pip install git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1 --no-build-isolation

      # Install the rest from the clean requirements file
      pip install -r requirements.txt

      ```

- Some files need some little modifications, like post_processing.py, since authors were using linux, which manages folders differently (\ instead of /), creating some parsing problems which impact the output.
- Changed the file `generate_prompts.py` to `generate_prompts_v2`.py since the original model was too heavy, slow and inefficient.

### Install Required Libraries

**Commands (run in conda environment):**
```bash
conda activate btb

# Install PyTorch with CUDA support (adjust CUDA version if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies (for step 3: Prompt generation)
pip install qwen-vl-utils

# Install/upgrade accelerate for GPU memory management (if needed)
pip install --upgrade accelerate
```

**Important Notes:**
- Replace `cu126` with your CUDA version (check with `nvidia-smi`) if you have problems verify GPU support.

### Verify GPU Support for PyTorch
**What it does:** Checks if your NVIDIA GPU is correctly configured to work with PyTorch for fast processing.

**Command:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Expected Output:**
- `True` = Your GPU is ready ✓
- `False` = You need to configure CUDA (see troubleshooting below)

**GPU troubleshooting (if output is False):**

1. Check your CUDA version:
   ```cmd
   nvidia-smi
   ```
   Look for "CUDA Version" in the top right (e.g., CUDA 12.6)

2. Uninstall PyTorch:
   ```bash
   pip uninstall torch torchvision
   ```

3. Reinstall PyTorch with the correct CUDA version:
   - For CUDA 12.6: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
   - For CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

---
### Installing the dataset
To test our detection model we need a dataset of inpainted images, we are going to download the results of the Flickr30k inpainting from the original Btb authors, since the whole collection would be too heavy for our machines.

We need to install the dataset from huggingface so we need to do the following procedure:

1. Install the necessary library (should be alreayd installed within the virtual environment)
   ```cmd
   pip install huggingface_hub
   ```
2. Go inside the folder `dataset` and run the python file `download_flickr_30k.py`, it will download the entire flickr30k inpainted dataset and the masks. Alternative datasets like `SAGI-D` (around 24GB) can be downloaded using their dedicated installer, but Flickr30k is recommended for its contained dimensions and relatively good quality inpaintings.

### Generate Image List file (Windows PowerShell/CMD)

In order to run the script we need some images in our dataset, and a text file containing the full paths of all the images we want to inpaint. Generating this file for large datasets manually would be unfeasible so we use the following commands. 

*Note: Make sure your terminal is currently located inside the `01_inpainting_generation` folder before running this command.*

**Command:**

```cmd
dir /b /s <image_directory>\*.jpg <image_directory>\*.png > <output_directory>\input_images_paths.txt
```

**Example:** If your images are in a folder called `test_images`:
```cmd
dir /b /s input_images\*.jpg input_images\*.png > config_and_utilities\input_images_paths.txt
```

*If you are in the main project folder you can just run the command adding the path of the main folder:*

```cmd
dir /b /s 01_inpainting_generation\input_images\*.jpg 01_inpainting_generation\input_images\*.png > 01_inpainting_generation\config_and_utilities\input_images_paths.txt
```

**Output:** A file called `input_images_paths.txt` with one image path per line:
```
C:\Users\utente\Projects\IdentifAI_Stage\Beyond-the-Brush\images\photo1.jpg
C:\Users\utente\Projects\IdentifAI_Stage\Beyond-the-Brush\images\photo2.jpg
C:\Users\utente\Projects\IdentifAI_Stage\Beyond-the-Brush\images\photo3.png
```

The output result must be placed in the `01_inpainting_generation/config_and_utilities` folder. In alternative, if the images are added in the `01_inpainting_generation/input_images` folder, this file can be automatically generated in the correct place also using the auxiliary script `text_file_generator.py` placed in the `config_and_utilities` folder.

Simply run:
```bash
python 01_inpainting_generation/config_and_utilities/text_file_generator.py
```

---



# Running the Main Script





## 1 - Image Processing Stage

### Parameters

| Parameter | What it means | Example |
|-----------|---------------|----------|
| `-r/--ram-model-path` | Path to RAM model weights (REQUIRED) | `weights/ram_swin_large_14m.pth` |
| `-d/--device` | Device to use ('cuda' for NVIDIA GPU, 'cpu') | `cuda` |
| `-i/--input_directory` | Directory with source images (default: `input_images/`) | `01_inpainting_generation/input_images` |
| `-t/--text_file` | Text file to list all image paths (default: `input_images.txt`) | `01_inpainting_generation/config_and_utilities/input_images_paths.txt` |
| `-o/--output-path` | Output directory for tags and masks (default: `tags_and_masks`) | `dataset/manually_generated/tags_and_masks` |
| `-tm/--top-masks` | Number of top masks to keep per image (optional, default: None) | `3` |

### Command Example
```bash
python 01_inpainting_generation/1_process_images.py -r weights/ram_swin_large_14m.pth -tm 3
```

**Note**: All other parameters have defaults and don't need to be specified.

### Output

After running, your `dataset/manually_generated/tags_and_masks/` will contain:
```
dataset/manually_generated/tags_and_masks/
├── tmp/                          # Temporary files (can be deleted after completion)
│   ├── tags/
│   │   ├── 0000000-tags.json
│   │   ├── 0000001-tags.json
│   │   └── ...
│   └── boxes/
│       ├── 0000000-boxes.json
│       ├── 0000001-boxes.json
│       └── ...
├── 0000000.json                  # Final results (tags + boxes + metadata)
├── 0000000.npz                   # Segmentation masks (pixel-level details)
├── 0000001.json
├── 0000001.npz
└── ...
```
- `.json` files: Text-based data (image paths, detected objects, confidence scores)
- `.npz` files: Binary arrays (visual masks showing exactly where each object is)

### "CUDA out of memory" error
If you get a memory error, reduce the batch sizes in `process_images.py`:
- Try to reduce the number of objects the program should identify (ex: `-t 3` instead of `-t 10`)
- Line ~187: Change `Batched(image_paths, 64)` to `Batched(image_paths, 16)`
- Line ~200: Change `BatchedZip(..., 4)` to `BatchedZip(..., 2)`


### Script takes too long
If processing is slow:
1. Verify GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check GPU usage while script runs: Open Task Manager → Performance → GPU
3. If GPU usage is low, you may be running on CPU by mistake, see GPU troubleshooting above

---








## 2 - Post-Processing Stage

### Overview
The `post_processing.py` script filters and organizes the results from `process_images.py`:
- Analyzes mask quality (area percentage, connectivity)
- Selects best small/medium/large objects per image
- Generates visualizations with green bounding boxes
- Prepares data for the inpainting stage

### Parameters

| Parameter | What it means | Example |
|-----------|---------------|---------|
| `-i/--input_dir` | Directory with JSON+NPZ files from Phase 1 (default: `tags_and_masks`) | `dataset/manually_generated/tags_and_masks` |
| `-m/--save_dir_masks` | Directory to save PNG mask files (default: `final_masks`) | `dataset/manually_generated/final_masks` |
| `-b/--save_dir_bb` | Directory to save bounding box images (default: `final_boxes`) | `dataset/manually_generated/final_boxes` |
| `-n/--num_images` | Number of images to sample (optional, default: None = all) | `None` or `4` |
| `-d/--dataset_name` | Dataset name for output file prefixes (default: `custom_dataset`) | `TestDataset` |

### Command Example
```bash
python 01_inpainting_generation/2_post_processing.py
```


### Output Files

The script generates:
- **Mask PNGs**: `dataset/manually_generated/final_masks/mask_<imagename>_small.png`, `mask_<imagename>_medium.png`, `mask_<imagename>_large.png`
- **Boxed images**: `dataset/manually_generated/final_boxes/mask_<imagename>_with_box.png` (image with green bounding box overlay)
- **Text files in** `01_inpainting_generation/config_and_utilities/`: 
  - `sampled_N_images_TestDataset.txt` - List of processed JSON files
  - `source_images_path_TestDataset.txt` - Paths to original images
  - `best_mask_labels_TestDataset.txt` - Image names + labels + sizes

---

### Auxiliary Script: empty_mask_fixer.py (Optional Cleanup)

Due to the morphological operations in the post processing stage, some extremely thin objects might be erased, resulting in completely black masks. Processing these empty masks in the next stages wastes compute resources. 

This auxiliary script automatically scans the `final_masks` directory, deletes any completely empty (black) `.png` files, and safely removes their orphaned bounding box images and entries from the `best_mask_labels.txt` file.

**Command Example:**
```bash
python 01_inpainting_generation/config_and_utilities/empty_mask_fixer.py
```

---


## 3 - Prompt Generation Stage (Qwen2-VL - Recommended)

### Overview
The `generate_prompts_v2.py` script uses **Qwen2-VL-2B-Instruct** model to generate replacement suggestions for objects in images, this is an alternative model to the original used by the authors, which is:
- **Faster**: ~15 seconds per image (much faster than Mini-InternVL)
- **Efficient**: Only 2B parameters, runs on 16GB GPU
- **Diverse outputs**: We can control the diversity of the suggestion prompts it generates by adjusting temperature and sampling parameters


### Model Setup
Huggingface link to the model: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct

```
# Install project dependencies (for step 3: Prompt generation)
pip install qwen-vl-utils

# Install/upgrade accelerate for GPU memory management (if needed)
pip install --upgrade accelerate
```

### Parameters

| Parameter | What it means | Example |
|-----------|---------------|---------|
| `--bounding_box_dir` | Directory containing images with green bounding boxes (from post_processing.py) | `dataset/manually_generated/final_boxes` |
| `--labels_file` | Text file with object labels (generated by post_processing.py) | `best_mask_labels_TestDataset.txt` |
| `--output_dir` | Where to save JSON files with generated prompts | `dataset/manually_generated/prompts_v2` |
| `--num_prompt` | How many diverse replacement suggestions per image | `5` |

**Command Example**
```bash
python 01_inpainting_generation/3_generate_prompts_v2.py -n 3
```

On first execution the model will automatically be downloaded from HuggingFace in a local directory of your pc (~4GB). Subsequent runs skip this step.



### Output Files

The script generates JSON files (one per image) with detailed replacement suggestions:

Example output (`IMG-20191022-WA0000_large.json`):
```json
[
  "Rationale: The living room setting with warm lighting suggests a need for comfort. The green box shows a small decorative item area. A ceramic vase would add elegance while maintaining the room's aesthetic balance.\nReplace with: Ceramic vase with golden trim",
  
  "Rationale: Looking at the room's modern furniture and neutral color palette, a functional replacement that complements the space would be appropriate. The size is ideal for a table decoration.\nReplace with: Wooden picture frame",
  
  "Rationale: Given the cozy interior design and available space in the green box, an item that adds visual interest would enhance the scene. A potted plant fits perfectly within the context.\nReplace with: Small potted succulent plant"
]
```

### How Diversity Works (Qwen2-VL)

The script generates diverse prompts by varying generation parameters for each iteration:

| Iteration | Temperature | Top-P | Top-K | Purpose |
|-----------|-------------|-------|-------|---------|
| 1 | 0.7 | 0.8 | 50 | Conservative, safe suggestions |
| 2 | 1.1 | 0.9 | 40 | Balanced creativity |
| 3 | 1.5 | 1.0 | 30 | More creative alternatives |
| 4+ | 1.9+ | 1.0 | 10 | Highly creative, diverse options |

- **Temperature**: Controls randomness (lower = conservative, higher = creative)
- **Top-P**: Controls vocabulary distribution (higher = more variety)
- **Top-K**: Limits candidate tokens (lower = more focused)
- **Repetition Penalty**: Discourages repeating similar suggestions


### Troubleshooting

**Error: "Using a device_map, torch.device context manager... requires accelerate"**
```bash
pip install --upgrade accelerate
```
---







## 3.5 - Prompt Generation Stage (Mini-InternVL - Legacy)

### Overview (Legacy Model)
The `generate_prompts.py` script uses **Mini-InternVL-Chat-4B** model to generate replacement suggestions for objects in images:
- Loads a Mini-InternVL-Chat model (4B parameters)
- For each image with green bounding box: asks the model for replacement object suggestions
- Generates multiple diverse suggestions per image
- Saves results as JSON files (one per image)

**Note**: This is the original model used by the authors. It is slower and more expensive than the alternative we found, so in our project we use **generate_prompts_v2.py** (Qwen2-VL) instead.

### Parameters (Same as step 3)

| Parameter | What it means | Example |
|-----------|---------------|---------|
| `--bounding_box_dir` | Directory containing images with green bounding boxes (from post_processing.py) | `dataset/manually_generated/final_boxes` |
| `--labels_file` | Text file with object labels (generated by post_processing.py) | `best_mask_labels_TestDataset.txt` |
| `--output_dir` | Where to save JSON files with generated prompts | `dataset/manually_generated/prompts` |
| `--num_prompt` | How many replacement suggestions to generate per image | `5` |

### Command Example
```bash
python 01_inpainting_generation/3_generate_prompts.py -n 3
```


### Output Files

The script generates:
- **Prompt JSONs**: One JSON file per image (e.g., `dog_image_small.json`)
- Each JSON contains a list of replacement suggestions with explanations

Example output (`dog_image_small.json`):
```json
[
  "Rationale: A cat is similar in size and domesticated character\nReplace with: cat",
  "Rationale: A small dog breed maintains scene coherence\nReplace with: poodle",
  "Rationale: A rabbit has comparable dimensions and fits indoor setting\nReplace with: rabbit",
  "Rationale: A duck maintains animal continuity\nReplace with: duck",
  "Rationale: A fox has similar predatory posture\nReplace with: fox"
]
```

### Important Notes

- **GPU Memory**: Ensure sufficient GPU memory (model uses ~8GB). If out of memory errors occur, reduce batch size in code
- **Model Download**: First run downloads the Mini-InternVL model (~4.5GB) - this may take time
- **Processing Time**: ~60-100 seconds per image depending on image size and GPU speed (slower than Qwen2-VL)
- **Internet**: Requires internet connection for first-time model download

---




## 4 - Inpainting Stage (3 Options)

This phase has **3 alternativess**, each with completely different dependencies. Run only ONE of these three options per time.

**⚠️ IMPORTANT**: Each option requires its own isolated Conda environment to avoid dependency conflicts.

---

### Option A: Fooocus-API (`4_inpaint_images_fooocus.py`) - **Highest Quality**

Fooocus uses SDXL (Stable Diffusion XL). It runs as an external API server.

#### Setup (One-Time Only)

**1. Clone Fooocus-API Repository**
```bash
cd ..
git clone https://github.com/mrhan1993/Fooocus-API.git
cd Fooocus-API
```

**2. Create Fooocus Environment**
```bash
conda create -n fooocus_env python=3.10 -y
conda activate fooocus_env
pip install -r requirements.txt # We are in the Foocus folder
# Install torch and torchvision in the correct version for the hardware used
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

#### Execution

**Terminal 1: Start Fooocus Server** (leave running in background)
```bash
cd Fooocus-API
conda activate fooocus_env
python main.py --port 8889
# Wait for: "Uvicorn running on http://127.0.0.1:8889"
```

**Terminal 2: Run Inpainting Script**
```bash
cd Beyond-the-Brush
conda activate btb
python 01_inpainting_generation/4_inpaint_images_fooocus.py
```

#### Input
- Original source images (paths from: `01_inpainting_generation/config_and_utilities/source_images_path_custom_dataset.txt`)
- Binary masks: `dataset/manually_generated/final_masks/`
- Prompts: `dataset/manually_generated/prompts_v2/`

#### Output
- **Inpainted images**: `dataset/manually_generated/output_images_fooocus/`
- **Response JSONs**: `dataset/manually_generated/inpainted_images_jsons/` (server metadata)

#### Parameters
- `-i/--images_paths`: Source images list (default: `source_images_path_custom_dataset.txt`)
- `-m/--masks_path`: Masks directory (default: `dataset/manually_generated/final_masks`)
- `-p/--prompt`: Prompts directory (default: `dataset/manually_generated/prompts_v2`)
- `-o/--save_path`: Output directory (default: `dataset/manually_generated/output_images_fooocus`)


#### Auxiliary Script: clean_fooocus_output.py (Error Cleanup)

During long inpainting sessions (like processing hundreds of images), the Fooocus API might occasionally timeout or overload. When this happens, it returns a text based error (like "Internal Server Error") instead of an image, which gets saved as a file without an extension.

This script scans the Fooocus output directory and automatically removes these invalid, extensionless error files, leaving your dataset perfectly clean with only the valid `.png` inpainted images.

**Command Example:**
```bash
python 01_inpainting_generation/config_and_utilities/clean_fooocus_output.py
```

---

### Option B: Simple Lama (`4_inpaint_images_lama.py`) - **Fast & Lightweight**

Lama is a lightweight, self-contained model. No external server required. **Fastest option**.

#### Setup (One-Time Only)

```bash
conda create -n lama_env python=3.11 -y
conda activate lama_env
# Install torch and torchvision in the correct version for the hardware used
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/requirements_lama.txt
```

#### Execution

```bash
cd Beyond-the-Brush
conda activate lama_env
python 01_inpainting_generation/4_inpaint_images_lama.py
```

#### Input
- Original source images (paths from: `config_and_utilities/source_images_path_TestDataset.txt`)
- Binary masks: `dataset/manually_generated/final_masks/`

#### Output
- **Inpainted images**: `dataset/manually_generated/output_images_lama/`

#### Parameters
- `-i/--images_paths`: Source images list (default: `source_images_path_custom_dataset.txt`)
- `-m/--masks_path`: Masks directory (default: `dataset/manually_generated/final_masks`)
- `-o/--save_path`: Output directory (default: `dataset/manually_generated/output_images_lama`)

---

### Option C: Stable Diffusion Legacy (`4_inpaint_images_stable_diffusion.py`) - **Good Quality**

Stable Diffusion v1.5 inpainting model from Hugging Face. Good balance between quality and speed.

#### Setup (One-Time Only)

```bash
conda create -n sd_env python=3.11 -y
conda activate sd_env
# Install torch and torchvision in the correct version for the hardware used
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/requirements_stable_diffusion.txt
```

#### Execution

```bash
cd Beyond-the-Brush
conda activate sd_env
python 01_inpainting_generation/4_inpaint_images_stable_diffusion.py
```

#### Input
- Original source images (paths from: `01_inpainting_generation/config_and_utilities/source_images_path_custom_dataset.txt`)
- Binary masks: `dataset/manually_generated/final_masks/`
- Prompts: `dataset/manually_generated/prompts_v2/`

#### Output
- **Inpainted images**: `dataset/manually_generated/output_images_st_diff/`

#### Parameters
- `-i/--images_paths`: Source images list (default: `source_images_path_custom_dataset.txt`)
- `-m/--masks_path`: Masks directory (default: `dataset/manually_generated/final_masks`)
- `-p/--prompt`: Prompts directory (default: `dataset/manually_generated/prompts_v2`)
- `-o/--save_path`: Output directory (default: `dataset/manually_generated/output_images_st_diff`)

---

### Complete Pipeline Example

Run all phases sequentially:

```bash
# Phase 1: Extract masks
python 01_inpainting_generation/1_process_images.py -r weights/ram_swin_large_14m.pth -tm 3

# Phase 2: Post-process masks
python 01_inpainting_generation/2_post_processing.py

# Phase 3: Generate prompts
python 01_inpainting_generation/3_generate_prompts_v2.py

# Phase 4: Choose ONE inpainting backend
# Option A (Fooocus - requires separate terminal with server)
python 01_inpainting_generation/4_inpaint_images_fooocus.py

# Option B (Lama)
conda activate lama_env
python 01_inpainting_generation/4_inpaint_images_lama.py

# Option C (Stable Diffusion)
conda activate sd_env
python 01_inpainting_generation/4_inpaint_images_stable_diffusion.py
```

---

### Pipeline Order (Complete!)
0. **setup the environment**
   - Create the virtual environment
   - Modify the requirements.txt file
   - Install the necessary libraries
   - Verify GPU support
   - Create `images_list.txt` file
1. **process_images.py**
   - Tags images
   - Detects objects
   - Segments objects
2. **post_processing.py**
   - Analyzes and filters masks
   - Creates visualizations
   - Prepares data for inpainting
3. **generate_prompts_v2.py** (Recommended) OR **generate_prompts.py** (Original)
   - Generates replacement suggestions
   - Outputs JSON files with prompts
4. **Choose ONE inpainting backend**:
   - **4_inpaint_images_fooocus.py** (Best quality, requires external API server)
   - **4_inpaint_images_lama.py** (Fast & lightweight, local)
   - **4_inpaint_images_stable_diffusion.py** (Good quality, local)

  