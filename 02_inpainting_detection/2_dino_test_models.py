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

def get_parser():
    parser = argparse.ArgumentParser(description="Test Dinolizer with Metrics per Generator")
    parser.add_argument('-d', '--dataset_dir', type=str, default='dataset/flickr30k', help='Base dataset directory containing test.csv')
    parser.add_argument('--csv', type=str, default='test/test.csv', help='Relative path to the CSV file within dataset_dir')
    parser.add_argument('-md', '--model_dir', type=str, default='weights/dinov3-vits', help='Directory of local DINO weights')
    parser.add_argument('-w', '--weights', type=str, required=True, default='models/checkpoints/dinolizer/dinolizer_best.pth',help='Path to trained checkpoint (.pth)')
    parser.add_argument('--model_type', type=str, default=None, help='Model class to use')
    
    parser.add_argument('--resize', type=int, nargs='?', const=320, default=None, help='Enable resize-based testing')
    parser.add_argument('--patch_based', type=int, nargs='?', const=512, default=None, help='Enable patch-based testing')
    
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for binarizing mask')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('-ts', '--test_samples', type=int, default=5, help='Number of random images to include in visual summary plots')
    
    parser.add_argument('--clean_test', action='store_true', help='Delete previous Dinolizer test outputs')
    return parser

def main(args):
    # ======== PATHS SETUP ========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detection_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(detection_dir)
    
    DATASET_DIR = Path(project_root) / args.dataset_dir
    WEIGHTS_PATH = os.path.join(project_root, args.weights)

    OUTPUT_BASE = os.path.join(project_root, "predicted_masks", "dinolizer")
    CSV_PATH = os.path.join(OUTPUT_BASE, "test_metrics_per_generator.csv")
    HEATMAP_DIR = os.path.join(OUTPUT_BASE, "heatmaps")
    MASK_DIR = os.path.join(OUTPUT_BASE, "masks")

    if args.clean_test:
        import shutil
        if os.path.exists(OUTPUT_BASE):
            shutil.rmtree(OUTPUT_BASE)
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ======== CHECKPOINT LOADING ========
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[ERROR] Weights file not found: {WEIGHTS_PATH}")
        return
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)

    # ======== ARGUMENT RESOLUTION ========
    user_gave_resize  = args.resize is not None
    user_gave_patches = args.patch_based is not None

    if not user_gave_resize and not user_gave_patches:
        ckpt_patch_based = checkpoint.get('patch_based', None)
        ckpt_patch_size  = checkpoint.get('patch_size', None)
        if ckpt_patch_based is None:
            args.is_patch_based = False
            RESIZE_SIZE, PATCH_SIZE = 320, None
        elif ckpt_patch_based:
            args.is_patch_based = True
            PATCH_SIZE, RESIZE_SIZE = (ckpt_patch_size if ckpt_patch_size else 512), None
        else:
            args.is_patch_based = False
            RESIZE_SIZE, PATCH_SIZE = checkpoint.get('resize_size', 320), None
    else:
        args.is_patch_based = user_gave_patches
        RESIZE_SIZE = args.resize if not args.is_patch_based else None
        PATCH_SIZE  = args.patch_based if args.is_patch_based else None

    user_gave_model_dir = '-md' in os.sys.argv or '--model_dir' in os.sys.argv
    if not user_gave_model_dir and checkpoint.get('model_dir', None) is not None:
        args.model_dir = checkpoint.get('model_dir')
    LOCAL_MODEL_PATH = os.path.join(project_root, args.model_dir)

    # ======== MODEL INIT ========
    model_type_str = args.model_type or checkpoint.get('model_type', 'DinoLizerCNN')
    model_classes = {
        'DinoLizer': DinoLizer, 'DinoLizerMLP': DinoLizerMLP, 'DinoLizerCNN': DinoLizerCNN,
        'DinoLizerConvNext': DinoLizerConvNext, 'DinoLizerConvNextMLP': DinoLizerConvNextMLP, 'DinoLizerConvNextCNN': DinoLizerConvNextCNN,
    }
    model = model_classes[model_type_str](LOCAL_MODEL_PATH).to(DEVICE)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    # ======== DATALOADER SETUP ========
    do_resize = not args.is_patch_based

    csv_full_path = DATASET_DIR / args.csv
    test_csv = pd.read_csv(csv_full_path)
    
    if do_resize:
        original_sizes = {}
        for img_path in test_csv['image_path']:
            with Image.open(DATASET_DIR / img_path) as img:
                w, h = img.size
                original_sizes[img_path] = (h, w)

    test_transform = JointTransform(
        apply_resize=do_resize, size=(RESIZE_SIZE, RESIZE_SIZE), 
        preprocess=True, augment=False, prob=0.0, normalize=True
    )
    test_dataset = DinoCSVDataset(csv_file=csv_full_path, base_dir=DATASET_DIR, transform=test_transform)
    batch_size = args.batch_size if not args.is_patch_based else 1
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # ======== METRICS SETUP (GLOBAL + LOCAL) ========

    # Read the 'inpainting_model' column from the CSV
    if 'inpainting_model' in test_csv.columns:
        # Take the unique values (lowercase for compatibility)
        GENERATORS = [str(g).lower() for g in test_csv['inpainting_model'].unique() if pd.notna(g)]
    else:
        print("[WARNING] 'inpainting_model' column not found in the CSV. Fallback on default SAGI-D generators (brushnet, hdpainter, powerpaint, inpaintanything, removeanything).")
        GENERATORS = ['brushnet', 'hdpainter', 'powerpaint', 'inpaintanything', 'removeanything']
    
    # Global Accumulators
    auroc_global = BinaryAUROC(thresholds=200).to(DEVICE)
    TP_global, FP_global, FN_global = 0.0, 0.0, 0.0
    
    # Per-Generator Accumulators
    auroc_gen = {gen: BinaryAUROC(thresholds=200).to(DEVICE) for gen in GENERATORS}
    TP_gen = {gen: 0.0 for gen in GENERATORS}
    FP_gen = {gen: 0.0 for gen in GENERATORS}
    FN_gen = {gen: 0.0 for gen in GENERATORS}
    # For dataset distribution informations
    COUNT_gen = {gen: 0 for gen in GENERATORS}

    # Plot Storage
    plot_data_global = {'images': [], 'gt': [], 'pred': [], 'heatmaps': [], 'files': []}
    plot_data_gen = {gen: {'images': [], 'gt': [], 'pred': [], 'heatmaps': [], 'files': []} for gen in GENERATORS}


    random.seed(42)
    target_plot_indices = set(random.sample(range(len(test_dataset)), min(args.test_samples, len(test_dataset))))

    # ======== EVALUATION LOOP ========
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing", colour='red')):
            images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.float32)
            start_idx = batch_idx * batch_size
            filenames = [Path(p).name for p in test_dataset.data.iloc[start_idx:start_idx+images.size(0)]['image_path']]
            
            probs_list, masks_list = [], []
            
            if args.is_patch_based:
                for i in range(images.size(0)):
                    single_image = images[i].unsqueeze(0)
                    with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                        logits = sliding_window_inference(
                            inputs=single_image, roi_size=(PATCH_SIZE, PATCH_SIZE), sw_batch_size=4,
                            predictor=model, overlap=0.5, mode='gaussian', 
                            sw_device=torch.device(DEVICE), device=torch.device(DEVICE)
                        )
                    if masks.shape[-2:] != logits.shape[-2:]:
                        masks = F.interpolate(masks, size=logits.shape[-2:], mode='nearest')
                    probs_list.append(torch.sigmoid(logits).squeeze(0))
                    masks_list.append(masks[i])
            else:
                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    raw_logits = model(images)
                for i in range(images.size(0)):
                    img_path = test_dataset.data.iloc[start_idx + i]['image_path']
                    orig_h, orig_w = original_sizes[img_path]
                    logits_up = F.interpolate(raw_logits[i].unsqueeze(0).float(), size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0)
                    mask_up = F.interpolate(masks[i].unsqueeze(0).float(), size=(orig_h, orig_w), mode='nearest').squeeze(0) 
                    probs_list.append(torch.sigmoid(logits_up))
                    masks_list.append(mask_up)

            probs_np, bin_np = [], []
            for i in range(len(probs_list)):
                mask, prob = masks_list[i], probs_list[i]
                pred = (prob >= args.threshold).float()
                
                p_flat, m_flat = pred.view(-1), mask.view(-1)
                tp, fp, fn = (p_flat * m_flat).sum().item(), (p_flat * (1 - m_flat)).sum().item(), ((1 - p_flat) * m_flat).sum().item()

                # Update Global
                TP_global += tp; FP_global += fp; FN_global += fn
                auroc_global.update(prob.unsqueeze(0), mask.long().unsqueeze(0))

                # Identify Generator & Update Local

                # Finds out the model from the .csv file using the global index
                if 'inpainting_model' in test_dataset.data.columns:
                    gen_name = str(test_dataset.data.iloc[start_idx + i]['inpainting_model']).lower()
                else:
                    # Fallback to finding the generator from the file name
                    current_file = filenames[i].lower()
                    gen_name = next((g for g in GENERATORS if g in current_file), 'unknown')
                
                if gen_name in GENERATORS:
                    TP_gen[gen_name] += tp; FP_gen[gen_name] += fp; FN_gen[gen_name] += fn
                    auroc_gen[gen_name].update(prob.unsqueeze(0), mask.long().unsqueeze(0))
                    COUNT_gen[gen_name] += 1

                probs_np.append(prob.squeeze(0).cpu().numpy())
                bin_np.append(pred.squeeze(0).cpu().numpy())

            # Output and Plot interception
            for i in range(images.size(0)):
                global_idx = start_idx + i 
                base_name = filenames[i].replace('.png', '').replace('.jpg', '')
                
                plt.imsave(os.path.join(HEATMAP_DIR, f"{base_name}_heatmap.png"), probs_np[i], cmap='jet', vmin=0, vmax=1)
                Image.fromarray((bin_np[i] * 255).astype(np.uint8)).save(os.path.join(MASK_DIR, f"{base_name}_mask.png"))
                
                # Fetch original data if needed for plots
                orig_img, orig_mask = None, None
                def get_display_data():
                    if not args.is_patch_based:
                        ip = test_dataset.data.iloc[global_idx]['image_path']
                        mp = test_dataset.data.iloc[global_idx]['mask_path']
                        return np.array(Image.open(DATASET_DIR / ip).convert("RGB")), np.array(Image.open(DATASET_DIR / mp).convert("L")) / 255.0
                    else:
                        return denormalize_tensor_to_image(images[i]), masks[i].squeeze().cpu().numpy()

                # # Intercept for Global Plot
                # if global_idx in target_plot_indices:
                #     orig_img, orig_mask = get_display_data()
                #     plot_data_global['images'].append(orig_img); plot_data_global['gt'].append(orig_mask)
                #     plot_data_global['pred'].append(bin_np[i]); plot_data_global['heatmaps'].append(probs_np[i]); plot_data_global['files'].append(filenames[i])
                
                # Intercept for Per-Generator Plot
                current_file = filenames[i].lower()
                gen_name = next((g for g in GENERATORS if g in current_file), 'unknown')
                if gen_name != 'unknown' and len(plot_data_gen[gen_name]['files']) < args.test_samples:
                    if orig_img is None: orig_img, orig_mask = get_display_data()
                    plot_data_gen[gen_name]['images'].append(orig_img); plot_data_gen[gen_name]['gt'].append(orig_mask)
                    plot_data_gen[gen_name]['pred'].append(bin_np[i]); plot_data_gen[gen_name]['heatmaps'].append(probs_np[i]); plot_data_gen[gen_name]['files'].append(filenames[i])


    # ======== COMPUTE METRICS ========
    def calc_metrics(tp, fp, fn, auroc_obj):
        eps = 1e-6
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        return {'precision': p, 'recall': r, 'f1': f1, 'iou': iou, 'dice': dice, 'auroc': auroc_obj.compute().item()}


    # ======== PRINT DATASET DISTRIBUTION ========
    print("\n" + "="*40)
    print(" DATASET DISTRIBUTION")
    print("="*40)
    total_images = sum(COUNT_gen.values())
    print(f"- Total Test Images : {total_images}")
    for gen, count in COUNT_gen.items():
        if count > 0:
            print(f"  * {gen.ljust(16)}: {count} images ({count/total_images*100:.1f}%)")
            
    # Avoid computing losses on generators with no images
    active_generators = [g for g in GENERATORS if COUNT_gen[g] > 0]

    results = {}
    results['GLOBAL'] = calc_metrics(TP_global, FP_global, FN_global, auroc_global)
    for gen in active_generators:
        if (TP_gen[gen] + FP_gen[gen] + FN_gen[gen]) > 0:
            results[gen] = calc_metrics(TP_gen[gen], FP_gen[gen], FN_gen[gen], auroc_gen[gen])

    print("\n" + "="*40 + "\n FINAL TEST METRICS\n" + "="*40)

    print("\n" + "="*40 + "\n FINAL TEST METRICS\n" + "="*40)
    for k, v in results['GLOBAL'].items(): print(f"- {k.upper():<10}: {v:.4f}")
    
    # ======== SAVE CSV ========
    ckpt_meta = {
        'model_type': checkpoint.get('model_type', model_type_str),
        'patch_based': args.is_patch_based,
        'patch_size': PATCH_SIZE,
        'resize_size': RESIZE_SIZE,
        'dataset': args.dataset_dir,
        'model_dir': args.model_dir,
        'threshold': args.threshold,
    }
    with open(CSV_PATH, 'w') as f:
        f.write("# " + ",".join([f"{k}={v}" for k, v in ckpt_meta.items()]) + "\n")
        f.write("split,precision,recall,f1,iou,dice,auroc\n")
        for key, metrics in results.items():
            f.write(f"{key}," + ",".join([f"{metrics[m]:.6f}" for m in ['precision','recall','f1','iou','dice','auroc']]) + "\n")
    print(f"\n>>> Metrics saved to: {CSV_PATH}")

    # ======== GENERATE PLOTS ========
    print("\n>>> Generating Visual Summary Plots...")
    if plot_data_global['files']:
        plot_visual_summary(
            images=plot_data_global['images'], gt_masks=plot_data_global['gt'], pred_masks=plot_data_global['pred'], heatmaps=plot_data_global['heatmaps'], filenames=plot_data_global['files'],
            threshold=args.threshold, metrics=results['GLOBAL'], checkpoint_meta=ckpt_meta,
            save_path=os.path.join(OUTPUT_BASE, "summary_plot_GLOBAL.png"), num_samples=args.test_samples
        )
        plot_visual_summary_thesis(
            images=plot_data_global['images'], gt_masks=plot_data_global['gt'], pred_masks=plot_data_global['pred'], heatmaps=plot_data_global['heatmaps'], filenames=plot_data_global['files'],
            save_path=os.path.join(OUTPUT_BASE, "thesis_summary_plot_GLOBAL.pdf"), num_samples=args.test_samples
        )
    for gen in active_generators:
        if plot_data_gen[gen]['files']:
            plot_visual_summary_thesis(
                images=plot_data_gen[gen]['images'], gt_masks=plot_data_gen[gen]['gt'], pred_masks=plot_data_gen[gen]['pred'], heatmaps=plot_data_gen[gen]['heatmaps'], filenames=plot_data_gen[gen]['files'],
                # threshold=args.threshold, metrics=results[gen], checkpoint_meta=ckpt_meta,
                save_path=os.path.join(OUTPUT_BASE, f"summary_plot_{gen}.png"), num_samples= 1
            )

if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())