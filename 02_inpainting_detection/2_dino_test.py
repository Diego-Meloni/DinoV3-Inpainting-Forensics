import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC

# Dataset and models
from dinolizer_models import (
    DinoCSVDataset, JointTransform,
    DinoLizer, DinoLizerMLP, DinoLizerCNN, 
    DinoLizerConvNext, DinoLizerConvNextMLP, DinoLizerConvNextCNN
    )

# Utilities
from dinolizer_utils import (    
    denormalize_tensor_to_image, plot_visual_summary, plot_visual_summary_thesis
)

from monai.inferers import sliding_window_inference

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ===================================================================================
#                                      ARGUMENTS
# ===================================================================================
def get_parser():
    parser = argparse.ArgumentParser(description="Test Dinolizer with Metrics")
    # ======== Dataset path (points to a .csv file with iamges and mask paths) ========
    parser.add_argument('-d', '--dataset_dir', type=str, default='dataset/flickr30k', help='Base dataset directory containing test.csv')
    parser.add_argument('--csv', type=str, default='test/test.csv', help='Relative path to the CSV file within dataset_dir')
    # ======== Model configuration (directories for backbone weights and classificator checkpoint, model type) ========
    parser.add_argument('-md', '--model_dir', type=str, default='weights/dinov3-vits', help='Directory of local DINO weights')
    parser.add_argument('-w', '--weights', type=str, required=True, default='models/checkpoints/dinolizer/dinolizer_best.pth',help='Path to trained checkpoint (.pth)')
    parser.add_argument('--model_type', type=str, default=None, help='Model class to use: DinoLizer, DinoLizerMLP, DinoLizerCNN, DinoLizerConvNext, DinoLizerConvNextCNN. If not set, reads from checkpoint.')
    
    # Type of approach
    parser.add_argument('--resize', type=int, nargs='?', const=320, default=None,
                    help='Enable resize-based training/testing with optional size (default: 320 if flag given without value). Mutually exclusive with --patch_based.')
    parser.add_argument('--patch_based', type=int, nargs='?', const=512, default=None,
                    help='Enable patch-based training/testing with optional patch size (default: 512 if flag given without value). Mutually exclusive with --resize. Must be divisible by 16.')
    
    # ======== Test parameters ========
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for binarizing mask')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('-ts', '--test_samples', type=int, default=5, help='Number of random images to include in visual summary plot')
    
    # ======== Reset test environment ========
    parser.add_argument('--clean_test', action='store_true',
                    help='Delete previous Dinolizer test outputs (heatmaps, masks, plots, CSV) before running.')
    return parser

# Command for Base dinolizer [Patch based, multi loss, normalize]
# python 02_inpainting_detection/dinolizer/2_dino_test.py --patch_based 512 -bs 8 -md weights/dinov3-vits --dataset_dir dataset --csv combined_sagi_flickr/combined_domain_test.csv -w models/checkpoints/dinolizer/dinolizer_best.pth
# python 02_inpainting_detection/dinolizer/2_dino_test.py --resize 640 -bs 8 -md weights/dinov3-vits --dataset_dir dataset --csv combined_sagi_flickr/combined_domain_test.csv -w models/checkpoints/dinolizer/dinolizer_best.pth --clean_test

# Command for DinolizerConvNext (tiny weights) [Patch based, multi loss, normalize]
# python 02_inpainting_detection/dinolizer/2_dino_test.py --patch_based 512 -bs 8 -md weights/dinov3-convnet-tiny --dataset_dir dataset/SAGI-D -w models/checkpoints/dinolizer/dinolizer_best.pth 
# python 02_inpainting_detection/dinolizer/2_dino_test.py --resize 640 -bs 8 -md weights/dinov3-convnet-tiny --dataset_dir dataset/SAGI-D -w models/checkpoints/dinolizer/dinolizer_best.pth

# Command for DinolizerConvNext (small weights) [Patch based, multi loss, normalize]
# python 02_inpainting_detection/dinolizer/2_dino_test.py --patch_based 512 -bs 8 -md weights/dinov3-convnet-small --dataset_dir dataset/SAGI-D -w models/checkpoints/dinolizer/dinolizer_best.pth 
# python 02_inpainting_detection/dinolizer/2_dino_test.py --resize 640 -bs 8 -md weights/dinov3-convnet-small --dataset_dir dataset/SAGI-D -w models/checkpoints/dinolizer/dinolizer_best.pth 

# ===================================================================================
#                                      TEST LOGIC
# ===================================================================================
def main(args):
    #  ======== PATHS SETUP ========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nScript directory: {script_dir}")

    detection_dir = os.path.dirname(script_dir)

    project_root = os.path.dirname(detection_dir)
    print(f"Project directory: {project_root}")

    DATASET_DIR = Path(project_root) / args.dataset_dir
    WEIGHTS_PATH = os.path.join(project_root, args.weights)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test outputs paths   
    OUTPUT_BASE = os.path.join(project_root, "predicted_masks", "dinolizer")
    CSV_PATH = os.path.join(OUTPUT_BASE, "test_metrics_results.csv")
    HEATMAP_DIR = os.path.join(OUTPUT_BASE, "heatmaps")
    MASK_DIR = os.path.join(OUTPUT_BASE, "masks")

    # Reset test environment if needed
    if args.clean_test:
        import shutil
        if os.path.exists(OUTPUT_BASE):
            print(f"\n>>> [NOTE] --clean_test active --> clearing previous Dinolizer test outputs at {OUTPUT_BASE}")
            shutil.rmtree(OUTPUT_BASE)
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ======== CHECKPOINT WEIGHT LOADING ========
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[ERROR] Weights file not found: {WEIGHTS_PATH}")
        return
        
    print(f"\n>>> Loading weights from: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)


    # ======== SMART ARGUMENT RESOLUTION (read from checkpoint if not explicitly given) ========
    # --resize / --patch_based: if user didn't specify either, read training mode from checkpoint
    user_gave_resize  = args.resize is not None
    user_gave_patches = args.patch_based is not None

    if user_gave_resize and user_gave_patches:
        print("[ERROR] --resize and --patch_based are mutually exclusive. Choose only one.")
        return

    if not user_gave_resize and not user_gave_patches:
        # Nothing specified by user — read from checkpoint (retrocompatible)
        ckpt_patch_based = checkpoint.get('patch_based', None)
        ckpt_patch_size  = checkpoint.get('patch_size', None)

        if ckpt_patch_based is None:
            # Old checkpoint with no tracking info at all — fall back to safe default
            print(">>> Checkpoint has no patch_based info (old format). Defaulting to --resize 320.")
            args.is_patch_based = False
            RESIZE_SIZE = 320
            PATCH_SIZE  = None
        elif ckpt_patch_based:
            args.is_patch_based = True
            PATCH_SIZE  = ckpt_patch_size if ckpt_patch_size is not None else 512
            RESIZE_SIZE = None
            print(f">>> Read from checkpoint: patch-based [{PATCH_SIZE}×{PATCH_SIZE}]")
        else:
            args.is_patch_based = False
            RESIZE_SIZE = checkpoint.get('resize_size', 320)  # see Bug 1b below — needs to be saved in training too
            PATCH_SIZE  = None
            print(f">>> Read from checkpoint: resize-based [{RESIZE_SIZE}×{RESIZE_SIZE}]")
    else:
        # User explicitly specified — use their choice, but warn if it disagrees with checkpoint
        args.is_patch_based = user_gave_patches
        RESIZE_SIZE = args.resize if not args.is_patch_based else None
        PATCH_SIZE  = args.patch_based if args.is_patch_based else None

        ckpt_patch_based = checkpoint.get('patch_based', None)
        if ckpt_patch_based is not None and ckpt_patch_based != args.is_patch_based:
            train_mode = "patch-based" if ckpt_patch_based else "resize-based"
            user_mode  = "patch-based" if args.is_patch_based else "resize-based"
            print(f"[WARNING] You specified {user_mode}, but checkpoint was trained {train_mode}. Proceeding with your choice — results may be unreliable.")

        ckpt_size = checkpoint.get('patch_size' if args.is_patch_based else 'resize_size', None)
        user_size = PATCH_SIZE if args.is_patch_based else RESIZE_SIZE
        if ckpt_size is not None and ckpt_size != user_size:
            print(f"[WARNING] You specified size={user_size}, but checkpoint was trained with size={ckpt_size}. Proceeding with your choice.")

    if args.is_patch_based and PATCH_SIZE % 16 != 0:
        print(f"[ERROR] patch size {PATCH_SIZE} is not divisible by 16. Required by DINO's patch grid.")
        return

    # --- model_dir: read from checkpoint if user didn't specify it (retrocompatible) ---
    user_gave_model_dir = '-md' in os.sys.argv or '--model_dir' in os.sys.argv
    if not user_gave_model_dir:
        ckpt_model_dir = checkpoint.get('model_dir', None)
        if ckpt_model_dir is not None:
            args.model_dir = ckpt_model_dir
            print(f">>> Read from checkpoint: model_dir = {ckpt_model_dir}")
        else:
            print(f">>> Checkpoint has no model_dir info (old format). Using CLI default: {args.model_dir}")
    LOCAL_MODEL_PATH = os.path.join(project_root, args.model_dir)

    if args.is_patch_based:
        print(f">>> Final test mode: Patch-based [{PATCH_SIZE}×{PATCH_SIZE}] | batch size forced to 1")
    else:
        print(f">>> Final test mode: Resize-based [{RESIZE_SIZE}×{RESIZE_SIZE}]")



    # ======== SMART MODEL INITIALIZATION ========
    # Use the specified model, if not specified use the checkpoint one (or default to CNN)
    model_type_str = args.model_type or checkpoint.get('model_type', 'DinoLizerCNN')
    model_classes = {
        'DinoLizer': DinoLizer, 
        'DinoLizerMLP': DinoLizerMLP, 
        'DinoLizerCNN': DinoLizerCNN,
        'DinoLizerConvNext': DinoLizerConvNext,
        'DinoLizerConvNextMLP': DinoLizerConvNextMLP,
        'DinoLizerConvNextCNN': DinoLizerConvNextCNN,
        }

    if model_type_str not in model_classes:
        print(f"\n[ERROR] Unknown model_type '{model_type_str}'. Choose from: {list(model_classes.keys())}")
        return
    model = model_classes[model_type_str](LOCAL_MODEL_PATH).to(DEVICE)
    print(f"\n>>> Initializing {model_type_str} on {DEVICE} for the test.")
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    if args.model_type and checkpoint.get('model_type') and checkpoint.get('model_type') != args.model_type:
        print(f"[WARNING] --model_type={args.model_type} selected, but checkpoint was trained with {checkpoint.get('model_type')}")

    model.eval()


    # ======== TRANSFORMER & DATALOADER SETUP ========
    
    # If training is patch based, then resize is set to False (and otherwise)
    do_resize = not args.is_patch_based

    # Read CSV directly to get paths before dataset initialization
    csv_path = DATASET_DIR / args.csv
    test_csv = pd.read_csv(csv_path)
    
    # If we resize the image, track the original size
    # Our predictions will be rescaled for comparability
    if do_resize:
        print(">>> Reading original image sizes...")
        original_sizes = {}
        for img_path in test_csv['image_path']:
            with Image.open(DATASET_DIR / img_path) as img:
                w, h = img.size
                original_sizes[img_path] = (h, w)

    # Init DataLoader (Strictly no augmentation for testing)
    test_transform = JointTransform(
        apply_resize=do_resize,
        size=(RESIZE_SIZE, RESIZE_SIZE),  # only used when apply_resize=True, irrelevant otherwise
        preprocess=True,
        augment=False, 
        prob=0.0, 
        normalize=True,)
    
    test_dataset = DinoCSVDataset(csv_file= csv_path, base_dir=DATASET_DIR, transform=test_transform)

    # If we use sliding window inference (patch based training), 
    # then all images have different dimensions --> We use batch size 1
    if args.is_patch_based and args.batch_size > 1:
        print("[Warning] - batch size has been set to 1 in order to run sliding window inference, the test will continue normally.")
    batch_size = args.batch_size if not args.is_patch_based else 1
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # ======== METRICS AND PLOTS SETUP ========
    
    # Metrics accumulators
    auroc_metric = BinaryAUROC(thresholds=200).to(DEVICE)

    total_TP = 0.0
    total_FP = 0.0
    total_FN = 0.0
    
    # Variables for visual summary storage
    plot_images = []
    plot_gt_masks = []
    plot_pred_masks = [] 
    plot_heatmaps = [] 
    plot_filenames = []

    # Randomly select absolute indices to intercept during the loop (saves RAM)
    random.seed(42) # Fixed seed to always get the same random images across runs
    target_plot_indices = set(random.sample(range(len(test_dataset)), min(args.test_samples, len(test_dataset))))

    # ========================================
    #           EVALUATION LOOP
    # ========================================
    print("\n>>> Starting Test Evaluation...")

    # Smart warning: check if inference mode matches training mode
    ckpt_patch_based = checkpoint.get('patch_based', None)
    if ckpt_patch_based is not None and ckpt_patch_based != args.is_patch_based:
        train_mode = "patch-based" if ckpt_patch_based else "resize-based"
        test_mode  = "patch-based" if args.is_patch_based else "resize-based"
        print(f"[WARNING] Model was trained {train_mode} but you are testing {test_mode}. Results may be unreliable.")

    with torch.no_grad():
        # Use enumerate to keep track of the batch index
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing", colour='red')):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE, dtype=torch.float32)
            
            # Calculate the actual starting index in the CSV for this specific batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + images.size(0)
            
            # Extract the original filenames from the dataframe (only the filename part of the path)
            filenames = [Path(p).name for p in test_dataset.data.iloc[start_idx:end_idx]['image_path']]
            
            if args.is_patch_based:
                # ======== SLIDING WINDOW INFERENCE (patch based training) ========
                # We are not resizing images, dataloader can't get images of different sizes
                # --> We force batch size = 1.
                # We still use a "batch approach" to save some code later
                
                probs_list = []     # Predicted heatmaps of the batch (original image dimension)
                masks_list = []     # Predicted binary masks of the batch (original image dimension)
                
                for i in range(images.size(0)):
                    # Take one image and add a batch dimension: [C, H, W] -> [1, C, H, W]
                    # sliding_window_inference requires a batch dimension even for single images
                    single_image = images[i].unsqueeze(0)

                    with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                        # sliding_window_inference slides a (patch_size x patch_size) window
                        # across the full image, runs the model on each crop, then blends
                        # the overlapping predictions using Gaussian weighting (reduces seam artifacts).

                        # Output shape: [batch_size=1, 1, H, W] — same spatial size as the input image
                        logits = sliding_window_inference(
                            inputs=single_image,
                            roi_size=(PATCH_SIZE, PATCH_SIZE),
                            sw_batch_size=4,        # process 4 windows at once for efficiency
                            predictor=model,
                            overlap=0.5,            # overlap=0.5 means each window overlaps 50% with its neighbors
                            mode='gaussian',
                            sw_device=torch.device(DEVICE),
                            device=torch.device(DEVICE),
                        )

                    # Align mask to logits resolution if compressed masks differ
                    if masks.shape[-2:] != logits.shape[-2:]:
                        masks = F.interpolate(masks, size=logits.shape[-2:], mode='nearest')

                    # Logits are unbound, bring them into [0, 1] range with sigmoid
                    probs_list.append(torch.sigmoid(logits).squeeze(0))   # .squeeze(0) removes first dimension channel [B, 1, H, W] -> [1, H, W]
                    masks_list.append(masks[i])     # [1, H, W], already original size

            else:
                # ======== RESIZE PATH: batch inference ========
                # inference at RESIZE_SIZE X RESIZE_SIZE, upsample to original size
                # to ensure comparability with other models

                probs_list = []     # Predicted heatmaps of the batch (original image dimension)
                masks_list = []     # Predicted binary masks of the batch (original image dimension)

                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    raw_logits = model(images)    # [B, 1, RESIZE_SIZE, RESIZE_SIZE]

                for i in range(images.size(0)):
                    img_path = test_dataset.data.iloc[start_idx + i]['image_path']
                    orig_h, orig_w = original_sizes[img_path]

                    # Upsample logits to image size (unbounded values interpolate cleaner than probs)
                    logits_up = F.interpolate(
                        raw_logits[i].unsqueeze(0).float(),  # [1, 1, RESIZE_SIZE, RESIZE_SIZE]
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)                              # [1, orig_H, orig_W]

                    # Upsample true masks to the size of the original image
                    mask_up = F.interpolate(
                        masks[i].unsqueeze(0).float(),        # [1, 1, RESIZE_SIZE, RESIZE_SIZE]
                        size=(orig_h, orig_w),
                        mode='nearest'
                    ).squeeze(0) 
                
                    probs_list.append(torch.sigmoid(logits_up))  # [1, orig_H, orig_W]
                    masks_list.append(mask_up)                   # [1, orig_H, orig_W]


            # ======== Shared logic for output creation and metric update ========
            # Both paths now have probs_list and masks_list of length B (1 for patch, bs for resize)
            probs_np = []
            bin_np   = []
            for i in range(len(probs_list)):
                mask  = masks_list[i]                          # [1, H, W] true masks
                prob  = probs_list[i]                          # [1, H, W] heatmaps prediction
                pred  = (prob >= args.threshold).float()       # [1, H, W] binary prediction

                # --- Update global accumulators ---
                # Flatten tensors to calculate global pixel overlap easily
                p_flat = pred.view(-1)
                m_flat = mask.view(-1)
                
                total_TP += (p_flat * m_flat).sum().item()
                total_FP += (p_flat * (1 - m_flat)).sum().item()
                total_FN += ((1 - p_flat) * m_flat).sum().item()
                auroc_metric.update(prob.unsqueeze(0), mask.long().unsqueeze(0))    # Update AUROC

                probs_np.append(prob.squeeze(0).cpu().numpy())  # (H, W)
                bin_np.append(pred.squeeze(0).cpu().numpy())    # (H, W)


            # --- Save heatmaps and masks to disk, intercept for plot ---
            for i in range(images.size(0)):
                global_idx = start_idx + i # Calculate the absolute row index in the CSV
                base_name = filenames[i].replace('.png', '').replace('.jpg', '')
                
                # Save Heatmap
                plt.imsave(os.path.join(HEATMAP_DIR, f"{base_name}_heatmap.png"), probs_np[i], cmap='jet', vmin=0, vmax=1)
                
                # Save Binary Mask (Scale 0/1 to 0/255 for standard image viewing)
                Image.fromarray((bin_np[i] * 255).astype(np.uint8)).save(os.path.join(MASK_DIR, f"{base_name}_mask.png"))
                
                # Intercept the pre-selected random images for the summary plot
                
                if global_idx in target_plot_indices:
                    # For resize path: load original image and mask directly for display
                    if not args.is_patch_based:
                        img_path  = test_dataset.data.iloc[global_idx]['image_path']
                        mask_path = test_dataset.data.iloc[global_idx]['mask_path']
                        orig_img  = np.array(Image.open(DATASET_DIR / img_path).convert("RGB"))
                        orig_mask = np.array(Image.open(DATASET_DIR / mask_path).convert("L")) / 255.0

                        plot_images.append(orig_img)
                        plot_gt_masks.append(orig_mask)
                    else:
                        plot_images.append(denormalize_tensor_to_image(images[i]))
                        plot_gt_masks.append(masks[i].squeeze().cpu().numpy())
                    
                    plot_pred_masks.append(bin_np[i])
                    plot_heatmaps.append(probs_np[i])
                    plot_filenames.append(filenames[i])


    # ======== COMPUTE GLOBAL MEASURES ========
    eps = 1e-6  # Small epsilon term to prevent division by 0
    precision = total_TP / (total_TP + total_FP + eps)
    recall    = total_TP / (total_TP + total_FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = total_TP / (total_TP + total_FP + total_FN + eps)
    dice      = 2 * total_TP / (2 * total_TP + total_FP + total_FN + eps)

    final_metrics = {
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'iou':       iou,
        'dice':      dice,
        'auroc':     auroc_metric.compute().item()
    }

    
    print("\n" + "="*40)
    print(" FINAL TEST METRICS")
    print("="*40)
    for k, v in final_metrics.items():
        print(f"- {k.upper():<10}: {v:.4f}")
    
    # ======== SAVE METRICS AND TEST DETAILS ON CSV FILE ========
    # Read model config from checkpoint for the metadata line
    ckpt_meta = {
        'model_type':     checkpoint.get('model_type',     model_type_str),
        'patch_based':    args.is_patch_based,
        'patch_size':     PATCH_SIZE,
        'resize_size':    RESIZE_SIZE,
        'dataset':        args.dataset_dir,
        'model_dir':      args.model_dir,
        'multi_loss':     checkpoint.get('multi_loss',     'unknown'),
        'normalize_loss': checkpoint.get('normalize_loss', 'unknown'),
        'threshold':      args.threshold,
    }
    # Initial line for the csv containing test details
    meta_str = ",".join([f"{k}={v}" for k, v in ckpt_meta.items()])

    # Save the csv (details, header, values)
    with open(CSV_PATH, 'w') as f:
        f.write(f"# {meta_str}\n")
        f.write("precision,recall,f1,iou,dice,auroc\n")
        f.write(",".join([f"{final_metrics[k]:.6f}" for k in ['precision','recall','f1','iou','dice','auroc']]) + "\n")
    print(f"\n>>> Metrics saved to: {CSV_PATH}")

    # ======== GENERATE VISUAL SUMMARY PLOT ========
    if plot_images:
        print(">>> Generating Visual Summary Plot...")
        summary_plot_path = os.path.join(OUTPUT_BASE, "summary_plot.png")
        thesis_summary_plot_path = os.path.join(OUTPUT_BASE, "thesis_summary_plot.png")

        # Build checkpoint_meta dict for the plot (reuse what you built for the CSV)
        checkpoint_metadata = {
            'model_type':     checkpoint.get('model_type',     model_type_str),
            'patch_based':    args.is_patch_based,
            'patch_size':     PATCH_SIZE,
            'resize_size':    RESIZE_SIZE,
            'dataset':        args.dataset_dir,
            'model_dir':      args.model_dir,
            'multi_loss':     checkpoint.get('multi_loss',     False),
            'normalize_loss': checkpoint.get('normalize_loss', False),
            'alpha':          checkpoint.get('alpha',          None),
            'gamma':          checkpoint.get('gamma',          None),
            'learning_rate':  checkpoint.get('learning_rate',  None),
            'batch_size':     checkpoint.get('batch_size',     None),
            'best_epoch':     checkpoint.get('best_epoch',     None),
        }
        
        plot_visual_summary(
            images=plot_images,
            gt_masks=plot_gt_masks,
            pred_masks=plot_pred_masks,
            heatmaps=plot_heatmaps,
            filenames=plot_filenames,
            threshold=args.threshold,
            metrics=final_metrics,
            checkpoint_meta=checkpoint_metadata,
            save_path=summary_plot_path,
            num_samples=args.test_samples       # Passed directly from the parser arguments
        )

        plot_visual_summary_thesis(
                    images=plot_images,
                    gt_masks=plot_gt_masks,
                    pred_masks=plot_pred_masks,
                    heatmaps=plot_heatmaps,
                    filenames=plot_filenames,
                    save_path=thesis_summary_plot_path,
                    num_samples=2,       # Passed directly from the parser arguments
                )

if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())