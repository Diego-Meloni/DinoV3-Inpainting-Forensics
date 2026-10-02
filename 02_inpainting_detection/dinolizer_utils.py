import os
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ===================================================================================
#                                 UTILITY FUNCTIONS
# ===================================================================================

# ======== Training file utilities ========
def calculate_dice(probs, masks, threshold=0.5, eps=1e-6, return_components=False):
    """
    Computes the Dice Coefficient using the exact same logic
    as the testing script to ensure mathematical consistency.
    If return_components=True, also returns TP, FP, FN for computing
    precision, recall, f1, and iou externally.
    """
    preds_bin = (probs >= threshold).float()
    
    p_flat = preds_bin.view(-1)
    m_flat = masks.view(-1)
    
    TP = (p_flat * m_flat).sum()
    FP = (p_flat * (1 - m_flat)).sum()
    FN = ((1 - p_flat) * m_flat).sum()
    
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    
    if return_components:
        return dice, TP, FP, FN
    return dice

# Function to calculate the dice loss
def compute_dice_loss(probs, masks, eps=1e-6):
    """
    Computes the continuous Dice Loss for backpropagation.
    Unlike the binary calculate_dice metric, this uses raw probabilities
    so gradients can flow backward to update the weights.
    """
    p_flat = probs.view(-1)
    m_flat = masks.view(-1)
    
    intersection = (p_flat * m_flat).sum()
    dice = (2. * intersection + eps) / (p_flat.sum() + m_flat.sum() + eps)
    # eps is a very small term added to avoid divisions by 0

    # Loss is 1 - Dice (we want to minimize the loss, so maximize the dice)
    return 1.0 - dice


def print_loss_config(args):
    print("\n>>> Loss Configuration:")
    if not args.multi_loss:
        print("    - Loss: Dice only")
        if args.normalize_loss:
            print("    - [WARNING] --normalize_loss is ignored when --multi_loss is False (single loss needs no normalization)")
        if args.alpha != 0.6 or args.gamma != 1.5:
            print(f"    - [WARNING] --alpha={args.alpha} and --gamma={args.gamma} are set but ignored: Focal Loss is not active")
    else:
        print("    - Loss: Focal + Dice")
        if args.alpha == 0.6 and args.gamma == 1.5:
            print(f"    - Focal loss: using default parameters (alpha={args.alpha}, gamma={args.gamma})")
        else:
            print(f"    - Focal loss: alpha={args.alpha}, gamma={args.gamma}")
        if args.normalize_loss:
            print("    - Normalization: EMA (Exponential Moving Average) — each loss is divided by a running")
            print("      estimate of its own magnitude, keeping both terms near 1.0 so neither dominates gradients.")
            print("      Checkpointing based on val_dice_loss improvement")
        else:
            print("    - Normalization: none — raw sum, AdamW optimizer handles gradient scaling implicitly")
            print("      Checkpointing based on raw val_loss improvement")
    print()



def add_details_to_checkpoint(checkpoint_path: str, new_details: dict):
    """
    Loads an existing checkpoint, merges in new key-value pairs, and saves it back.
    Use this to retroactively add metadata (e.g. model_type) to already-saved checkpoints.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    checkpoint.update(new_details)
    torch.save(checkpoint, checkpoint_path)
    print(f">>> Updated checkpoint: {checkpoint_path} with keys: {list(new_details.keys())}")


def extract_epoch_details(checkpoint_path: str) -> dict:
    """
    Loads a checkpoint and returns a clean dictionary of all relevant fields
    in a readable format. Safe: missing keys return None instead of crashing.
    """
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    return {
        # Training identity
        'model_type':       cp.get('model_type',        None),
        'model_dir':        cp.get('model_dir',          None),
        'dataset':          cp.get('dataset',           None),
        'patch_based':      cp.get('patch_based',       None),
        'patch_size':       cp.get('patch_size',        None),
        'resize_size':      cp.get('resize_size',        None),
        # Training config
        'multi_loss':       cp.get('multi_loss',        None),
        'normalize_loss':   cp.get('normalize_loss',    None),
        'augmentation':     cp.get('augmentation',      None),
        'learning_rate':    cp.get('learning_rate',     None),
        'batch_size':       cp.get('batch_size',        None),
        'alpha':            cp.get('alpha',             None),
        'gamma':            cp.get('gamma',             None),
        # Progress
        'epoch':            cp.get('epoch',             None),
        'best_epoch':       cp.get('best_epoch',        None),
        # Training losses
        'train_loss':       cp.get('train_loss',        None),
        'train_dice_loss':  cp.get('train_dice_loss',   None),
        'train_focal_loss': cp.get('train_focal_loss',  None),
        # Validation losses
        'val_loss':         cp.get('val_loss',          None),
        'val_dice_loss':    cp.get('val_dice_loss',     None),
        'val_focal_loss':   cp.get('val_focal_loss',    None),
        # Validation metrics
        'val_dice_score':   cp.get('val_dice_score',    None),
        'val_precision':    cp.get('val_precision',     None),
        'val_recall':       cp.get('val_recall',        None),
        'val_f1':           cp.get('val_f1',            None),
        'val_iou':          cp.get('val_iou',           None),
        'val_auroc':        cp.get('val_auroc',         None),
        # Best tracked values
        'best_val_loss':      cp.get('best_val_loss',      None),
        'best_val_dice_loss': cp.get('best_val_dice_loss', None),
    }


def save_details_json(details: dict, save_dir: str):
    """
    Writes a details.json file safely: first writes to details.json.temp,
    then renames it to details.json, overwriting the previous one.
    """
    import json
    temp_path  = os.path.join(save_dir, "details.json.temp")
    final_path = os.path.join(save_dir, "details.json")
    with open(temp_path, 'w') as f:
        json.dump(details, f, indent=4)
    os.replace(temp_path, final_path)

# ======== Test file utilities ========
def denormalize_tensor_to_image(tensor):
    """Converts a normalized PyTorch tensor back to a format viewable by matplotlib/PIL."""
    tensor = tensor.cpu().clone().detach()
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)
    tensor = tensor.numpy().transpose(1, 2, 0) 
    return (tensor * 255).astype(np.uint8)


def plot_visual_summary(images, gt_masks, pred_masks, heatmaps, filenames,
                        threshold, metrics, checkpoint_meta, save_path, num_samples=5):
    if not images:
        return

    n_samples = min(len(images), num_samples)

    # Compact layout: taller info row, tight image rows
    fig = plt.figure(figsize=(16, 2.5 + 3.2 * n_samples))
    gs = fig.add_gridspec(
        n_samples + 1, 4,
        height_ratios=[1] + [1] * n_samples,
        hspace=0.12, wspace=0.08
    )

    # --- INFO PANEL ---
    ax_left  = fig.add_subplot(gs[0, :3])
    ax_right = fig.add_subplot(gs[0, 3:])
    ax_left.axis('off')
    ax_right.axis('off')

    dataset_name = os.path.basename(checkpoint_meta.get('dataset', 'unknown'))
    loss_str = "Dice + Focal" if checkpoint_meta.get('multi_loss') else "Dice loss"
    if checkpoint_meta.get('multi_loss') and checkpoint_meta.get('alpha') is not None:
        loss_str += f" (α={checkpoint_meta.get('alpha')}, γ={checkpoint_meta.get('gamma')})"
    norm_str = "EMA" if checkpoint_meta.get('normalize_loss') else "Raw sum (AdamW)"

    # Type of training
    training_size = checkpoint_meta.get('patch_size') if checkpoint_meta.get('patch_based') else checkpoint_meta.get('resize_size')
    mode_str = f"Sliding window inference [Patch-based train {training_size}×{training_size}]" if checkpoint_meta.get('patch_based') else f"Resize-based [{training_size}×{training_size}]"
    weights_str = os.path.basename(checkpoint_meta.get('model_dir', 'unknown')) if checkpoint_meta.get('model_dir') else 'unknown'
    y = 0.97
    line_height = 0.09

    # Left table title
    # Title background bar (purple)
    ax_left.add_patch(FancyBboxPatch((0.01, 0.88), 0.898, 0.113,
        boxstyle="square,pad=0.0",
        transform=ax_left.transAxes,
        facecolor="#7d4bfd", edgecolor='none', alpha=1.0, zorder=1))

    # White title text on top of the title bar
    ax_left.text(0.45, 0.97, "MODEL DETAILS", transform=ax_left.transAxes,
            fontsize=14, fontfamily='monospace', fontweight='bold',
            verticalalignment='top', horizontalalignment='center',
            color='white', zorder=2)
    y -= line_height

    # Separation line
    ax_left.axhline(y=y, xmin=0.01, xmax=0.91,color="#202122", linewidth=2)
    y -= line_height * 0.6  # smaller gap since it's just a line
    
    # Left table lines 
    lines = [
        ("Architecture", True,  f": {checkpoint_meta.get('model_type', 'unknown')}"),
        ("Weights",      True,  f": {weights_str}"),
        ("Mode",         True,  f": {mode_str}"),
        ("Dataset",      True,  f": {dataset_name}"),
        ("Loss",         True,  f": {loss_str}"),
        ("Normalization",True,  f": {norm_str}"),
        ("LR / Batch",   True,  f": {checkpoint_meta.get('learning_rate','?')} / {checkpoint_meta.get('batch_size','?')}"),
        ("Best epoch",   True,  f": {checkpoint_meta.get('best_epoch','?')}   Threshold: {threshold}"),
    ]

    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)

    for item in lines:
        if len(item) == 2:  # title or separator line
            label, bold = item
            ax_left.text(0.04, y, label, transform=ax_left.transAxes,
                        fontsize=12, fontfamily='monospace',
                        fontweight='bold' if bold else 'normal',
                        verticalalignment='top')
        else:  # key: value line
            label, bold, value = item
            # Print bold key
            t = ax_left.text(0.04, y, label, transform=ax_left.transAxes,
                            fontsize=10, fontfamily='monospace',
                            fontweight='bold', verticalalignment='top')
            # Print normal value right after — fixed offset for alignment
            ax_left.text(0.04 + len("Normalization") * 0.018, y, value,
                        transform=ax_left.transAxes,
                        fontsize=10, fontfamily='monospace',
                        fontweight='normal', verticalalignment='top')
        y -= line_height

    # Background
    ax_left.add_patch(FancyBboxPatch((0.01, 0.1), 0.9, 0.9,
        boxstyle="square,pad=0.0",
        transform=ax_left.transAxes,
        facecolor="#f5f1db", edgecolor="#202122", linewidth=2, alpha=0.95, zorder=0))

    # Metrics table
    metric_labels = ['Precision', 'Recall', 'F1', 'IOU', 'Dice', 'AUROC']
    metric_keys   = ['precision', 'recall', 'f1', 'iou', 'dice', 'auroc']
    metric_vals   = [[label, f"{metrics[k]:.4f}"] for label, k in zip(metric_labels, metric_keys)]


    table = ax_right.table(
        cellText=metric_vals,
        colLabels=['Metric', 'Score'],
        loc='center',
        cellLoc='center',
        bbox=[0.05, 0.1, 0.88, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#7d4bfd")
            cell.set_text_props(fontweight='bold', color='white')
        if c == 0 and r > 0:
            cell.set_text_props(fontweight='bold')

    # --- Column headers on first image row only ---
    for col, title in enumerate(['Original', 'Ground Truth', 'Predicted Mask', 'Predicted Heatmap']):
        ax = fig.add_subplot(gs[1, col])
        ax.set_title(title, fontsize=14, fontweight='bold', pad=3)
        ax.axis('off')

    # --- Image rows ---
    for i in range(n_samples):
        row = i + 1

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(images[i])
        ax0.set_xlabel(filenames[i], fontsize=5.5, labelpad=1)
        ax0.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax0.spines.values(): spine.set_visible(False)

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(gt_masks[i], cmap='gray', vmin=0, vmax=1)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(pred_masks[i], cmap='gray', vmin=0, vmax=1)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[row, 3])
        im = ax3.imshow(heatmaps[i], cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax3, fraction=0.04, pad=0.02)
        ax3.axis('off')

    plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f">>> Summary plot saved to: {save_path}")


def plot_visual_summary_thesis(images, gt_masks, pred_masks, heatmaps, filenames, save_path, num_samples=4):
    """
    Academic version of the visual summary. 
    Removes headers, metrics, and styling to provide a clean grid 
    suitable for direct inclusion in a LaTeX document.
    """
    if not images:
        return

    n_samples = min(len(images), num_samples)

    # Compact layout: only image rows
    fig = plt.figure(figsize=(12, 3.0 * n_samples))
    gs = fig.add_gridspec(
        n_samples + 1, 4,
        height_ratios=[0.2] + [1] * n_samples, # Tiny row for headers
        hspace=0.05, wspace=0.05 # Very tight spacing
    )

    # --- Column headers ---
    for col, title in enumerate(['Original', 'Ground Truth', 'Predicted Mask', 'Predicted Heatmap']):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, title, fontsize=16, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # --- Image rows ---
    for i in range(n_samples):
        row = i + 1

        # Original Image
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(images[i])
        ax0.axis('off') # Remove filename and axes entirely for a clean look

        # Ground Truth Mask
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(gt_masks[i], cmap='gray', vmin=0, vmax=1)
        ax1.axis('off')

        # Predicted Binary Mask
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(pred_masks[i], cmap='gray', vmin=0, vmax=1)
        ax2.axis('off')

        # Predicted Heatmap
        ax3 = fig.add_subplot(gs[row, 3])
        im = ax3.imshow(heatmaps[i], cmap='jet', vmin=0, vmax=1)
        # Keep colorbar but make it smaller
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        ax3.axis('off')

    # Save as high-res PNG or PDF
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f">>> Academic summary plot saved to: {save_path}")