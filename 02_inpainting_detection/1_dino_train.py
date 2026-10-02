import os
import shutil
import argparse
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

# Sliding window for sliding window inference
from monai.inferers import sliding_window_inference

# Importing dino classes & utility functions
from dinolizer_models import (
    # Dataset
    DinoCSVDataset, JointTransform,
    # Models
    DinoLizer, DinoLizerMLP, DinoLizerCNN,
    DinoLizerConvNext, DinoLizerConvNextMLP, DinoLizerConvNextCNN
    )

from dinolizer_utils import (
    # Utility functions
    calculate_dice, compute_dice_loss, print_loss_config,
    extract_epoch_details, save_details_json
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ===================================================================================
#                                      ARGUMENTS
# ===================================================================================
def get_parser():
    parser = argparse.ArgumentParser(description="Train Dinolizer Head")
    # ======== Dataset path (points to the folder containing the CSVs) ========
    parser.add_argument('-d', '--dataset_dir', type=str, default='dataset/flickr30k', help='Base dataset directory')
    parser.add_argument('--train_csv', type=str, default='train/train.csv', help='Relative path to the train CSV file within dataset_dir')
    parser.add_argument('--val_csv', type=str, default='val/val.csv', help='Relative path to the val CSV file within dataset_dir')
    # ======== Model configuration (directory of backbone weights and type of classifier on top) ========
    parser.add_argument('-md', '--model_dir', type=str, default='weights/dinov3-vits', help='DINO weights directory')
    parser.add_argument('--model_type', type=str, default=None, help='Model class to use: DinoLizer, DinoLizerMLP, DinoLizerCNN, DinoLizerConvNext, DinoLizerConvNextCNN. If not set, reads from checkpoint.')
    # Type of approach
    parser.add_argument('--resize', type=int, nargs='?', const=320, default=None,
                    help='Enable resize-based training/testing with optional size (default: 320 if flag given without value). Mutually exclusive with --patch_based.')
    parser.add_argument('--patch_based', type=int, nargs='?', const=512, default=None,
                    help='Enable patch-based training/testing with optional patch size (default: 512 if flag given without value). Mutually exclusive with --resize. Must be divisible by 16.')

    # ======== Training parameters ========
    # Augmentation argument (0.0 means no augmentation, >0 enables it with that probability)
    parser.add_argument('-a', '--augmentation', type=float, default=0.0, help='Augmentation probability (0.0 to 1.0)')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Total number of epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-vf', '--val_freq', type=int, default=5, help='Report loss on validation loss every N epochs')
    
    # ======== Parameters for ablation studies ========
    parser.add_argument('--multi_loss', action='store_true',
                        help='Use Focal + Dice loss combination. If False, only Dice loss is used (paper baseline).')
    parser.add_argument('--normalize_loss', action='store_true',
                        help='Normalize losses with EMA (Exponential Moving Average) scaling. Only meaningful with --multi_loss.')
    # Focal loss parameters (only if multi loss is active)
    parser.add_argument('--alpha', type=float, default=0.6, help='Focal Loss Alpha')
    parser.add_argument('--gamma', type=float, default=1.5, help='Focal Loss Gamma')
    
    # ======== Automatically reset training environment to perform a new cycle ========
    parser.add_argument('--retrain', action='store_true', help='Delete all checkpoints and start from scratch')
    return parser

# Command for Base dinolizer [Patch based, multi loss, normalize]
# python 02_inpainting_detection/dinolizer/1_dino_train.py --patch_based -e 20 -bs 8 -lr 1e-4 -a 0.3 --multi_loss --normalize_loss --dataset_dir dataset/SAGI-D --model_type DinoLizer

# Command for DinolizerConvNext (tiny weights) [Patch based, multi loss, normalize] then [Resize, multi loss, normalize]
# python 02_inpainting_detection/dinolizer/1_dino_train.py --patch_based 512 -e 20 -bs 8 -lr 1e-4 -a 0.3 --multi_loss --normalize_loss --dataset_dir dataset/SAGI-D --model_dir "weights/dinov3-convnet-tiny" --model_type DinoLizerConvNext --retrain
# python 02_inpainting_detection/dinolizer/1_dino_train.py --resize 640 -e 20 -bs 8 -lr 1e-4 -a 0.3 --multi_loss --normalize_loss --dataset_dir dataset/SAGI-D --model_dir "weights/dinov3-convnet-tiny" --model_type DinoLizerConvNext --retrain

# Command for DinolizerConvNext (small weights) [Patch based, multi loss, normalize] then [Resize, multi loss, normalize]
# python 02_inpainting_detection/dinolizer/1_dino_train.py --patch_based 512 -e 20 -bs 8 -lr 1e-4 -a 0.3 --multi_loss --normalize_loss --dataset_dir dataset/SAGI-D --model_dir "weights/dinov3-convnet-small" --model_type DinoLizerConvNext --retrain
# python 02_inpainting_detection/dinolizer/1_dino_train.py --resize 640 -e 20 -bs 8 -lr 1e-4 -a 0.3 --multi_loss --normalize_loss --dataset_dir dataset/SAGI-D --model_dir "weights/dinov3-convnet-small" --model_type DinoLizerConvNext --retrain

# ===================================================================================
#                            TRAINING SETUP AND LOGIC
# ===================================================================================
def main(args):
    # Useful paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nScript dir: {script_dir}")
    detection_dir = os.path.dirname(script_dir)
    
    project_root = os.path.dirname(detection_dir)
    print(f"Project root: {project_root}")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOCAL_MODEL_PATH = os.path.join(project_root, args.model_dir) 
    DATASET_DIR = Path(project_root) / args.dataset_dir
    SAVE_DIR = os.path.join(project_root, "models", "checkpoints", "dinolizer")



    # Remove the existing checkpoints of the model if the retrain flag is on
    if args.retrain and os.path.exists(SAVE_DIR):
        print("\n>>> Retrain flag active: clearing checkpoint directory...")
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)


    # --- Resolve resize vs patch_based, mutually exclusive, default to resize ---
    if args.resize is not None and args.patch_based is not None:
        print("[ERROR] --resize and --patch_based are mutually exclusive. Choose only one.")
        return

    if args.resize is None and args.patch_based is None:
        args.resize = 320  # default behavior when neither is specified
        print(">>> Neither --resize nor --patch_based specified. Defaulting to --resize 320.")

    args.is_patch_based = args.patch_based is not None
    RESIZE_SIZE = args.resize if not args.is_patch_based else None
    PATCH_SIZE  = args.patch_based if args.is_patch_based else None

    if args.is_patch_based and PATCH_SIZE % 16 != 0:
        print(f"[ERROR] --patch_based={PATCH_SIZE} is not divisible by 16. Required by DINO's patch grid.")
        return

    if args.is_patch_based:
        print(f">>> Mode: Patch-based [{PATCH_SIZE}×{PATCH_SIZE}] | Validation/test use sliding window inference (batch size forced to 1)")
    else:
        print(f">>> Mode: Resize-based [{RESIZE_SIZE}×{RESIZE_SIZE}]")


    # ======== TRANSFORMER SETUP ========
    print(f"\n>>> Initializing Dataloaders and Model on {DEVICE}...")
    # If training is patch based, then resize is set to False (and otherwise)
    do_resize = not args.is_patch_based

    train_transform = JointTransform(
        apply_resize=do_resize,
        size= (RESIZE_SIZE, RESIZE_SIZE),       # Applied only if do_resize = True
        preprocess=True,
        augment=(args.augmentation > 0), # Augmentation True if it's > 0
        prob=args.augmentation,
        normalize= True,
    )

    val_transform = JointTransform(
        apply_resize=do_resize,
        size= (RESIZE_SIZE, RESIZE_SIZE),       # Applied only if do_resize = True
        preprocess=True,
        augment=False, # Validation has no augmentation
        prob=0.0,
        normalize= True,
    )


    # ======== DATASET SETUP ========
    # The transformer with the augmentation is applied in this phase
    train_dataset = DinoCSVDataset(
        csv_file=DATASET_DIR / args.train_csv, 
        base_dir=DATASET_DIR, 
        transform=train_transform, 
        use_patches=args.is_patch_based, 
        patch_size=PATCH_SIZE, 
        num_patches=10)

    val_dataset = DinoCSVDataset(
        csv_file=DATASET_DIR / args.val_csv, 
        base_dir=DATASET_DIR, 
        transform=val_transform, 
        use_patches=False, 
        patch_size=PATCH_SIZE, 
        num_patches=10)


    # ======== DATALOADER SETUP ========
    # If the model is trained with patches validation is done with sliding window inference
    # since images have different dimensions we need batch size=1 to work with sliding window 
    val_batch_size = 1 if args.is_patch_based else args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)  # num_workers allows CPU to read images while GPU is training, avoiding bottlenecks
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=2)

   # ======== USEFUL INFORMATION FOR DEBUG =====
    print_loss_config(args)

    # ======== SMART MODEL INITIALIZATION ========
    # Use the specified model, if not specified use the checkpoint one (or default to CNN)
    model_type_str = args.model_type
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
    print(f"\n[TRAINING] - Model: {model_type_str} - {'patch-based' if args.is_patch_based else 'resize'} approach")

    # Initialization, could be modified in resume logic section
    start_epoch = 0
    best_epoch = 0 


    # ======== CSV SETUP FOR METRICS LOGGING ========
    csv_log_path = os.path.join(SAVE_DIR, "training_metrics.csv")
    if not os.path.exists(csv_log_path) or args.retrain:
        with open(csv_log_path, mode='w') as f:
            # Metadata block at the top, above the column headers
            f.write(f"# model={type(model).__name__},patch_based={args.is_patch_based},multi_loss={args.multi_loss},")
            f.write(f"normalize_loss={args.normalize_loss},augmentation={args.augmentation},")
            f.write(f"dataset={args.dataset_dir},lr={args.learning_rate},batch_size={args.batch_size}\n")

            # Training losses
            f.write("epoch,train_loss,train_focal_loss,train_dice_loss,")
            # Validation losses and metrics
            f.write("val_loss,val_focal_loss,val_dice_loss,val_dice_score,")
            f.write("val_precision,val_recall,val_f1,val_iou,val_auroc\n")


    # ======== OPTIMIZER AND SCALER ========
    # Using AdamW with weight decay instead of Adam to heavily reduce overfitting
    optimizer = optim.AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    # The scaler is used to multiply the loss to avoid torch.autocast from rounding up at 0 the gradients.
    # very important especially when the model gets better and gradients update by little amounts
    scaler = torch.amp.GradScaler('cuda' if DEVICE == 'cuda' else 'cpu')


    # ======== SCHEDULER AND EARLY STOPPING LOGIC (based on validation loss results) ========
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    # We only check validation loss every -vf (validation frequency, parameter) epochs, so we count checks
    checks_without_improvement = 0
    EARLY_STOPPING_PATIENCE = 2 # If checks every 5 epochs and EARLY_STOPPING_PATIENCE = 2 --> early stop at 10 epochs with no improvement


    # ======== LOSS INITIALIZATION ========
    # EMA running means (only used when --multi_loss --normalize_loss are both true)
    focal_running_mean = None
    dice_running_mean  = None
    EMA_MOMENTUM = 0.98

    # Losses are initialized at infinity (so whatever the first is, it's the new best one)
    # Checkpointing metric — always track both, use the right one depending on config
    best_val_loss      = float('inf')   # used when: not normalize_loss (raw loss is comparable)
    best_val_dice_loss = float('inf')   # used when: normalize_loss (combined loss not comparable, use dice component)


    # ======== RESUME TRAINING LOGIC ========
    checkpoints = [f for f in os.listdir(SAVE_DIR) if f.startswith('dinolizer_epoch_') and f.endswith('.pth')]
    if checkpoints and not args.retrain:
        # Sort the list of available checkpoints and takes the last one (highest epoch)
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = os.path.join(SAVE_DIR, checkpoints[-1])

        print(f">>> Found checkpoint! Resuming training from: {latest_checkpoint}")

        # Extract information from the ckecpoint
        checkpoint_data = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint_data['model_state_dict'])

        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        start_epoch = checkpoint_data['epoch']

        # Extract best losses safely, defaulting to infinity (e.g. for first epochs)
        best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
        best_val_dice_loss = checkpoint_data.get('best_val_dice_loss', float('inf'))
        best_epoch = checkpoint_data.get('best_epoch', None)

        # Always load EMA means unconditionally — if the key is missing they stay None
        focal_running_mean = checkpoint_data.get('focal_running_mean', None)
        dice_running_mean  = checkpoint_data.get('dice_running_mean',  None)

    else:
        if args.retrain:
            print (">>> Retrain active. Starting training from scratch.")
        else:
            print(">>> No checkpoint found. Starting training from scratch.")

    # Safety check
    if start_epoch >= args.epochs:
        print(f">>> Model has already completed {start_epoch} epochs. Increase --epochs to continue.")
        return

    # -------------------------------------------------------------------------
    # EPOCH LOOP
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        
        # ======== TRAINING PHASE ========
        # Set the model in training state
        model.train()

        # Initializing losses for the epoch
        train_loss = 0.0
        train_focal_loss = 0.0 
        train_dice_loss_val = 0.0 
        
        # Training loop
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]", colour = 'yellow')
        for images, masks in train_loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.float32)

            if args.is_patch_based:
                # Flatten patches at the dimension of the batch
                images = images.view(-1, 3, PATCH_SIZE, PATCH_SIZE)
                masks = masks.view(-1, 1, PATCH_SIZE, PATCH_SIZE)

            # Cancel gradients of previous batches to only focus on current one
            optimizer.zero_grad()

            # torch.autocast changes the way numbers are rounded up, using 32 bit precision for 
            # more important operations and 16 bits precision for other operations
            # Loss calculation (combined loss obtained summing focal and dice losses)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                
                logits = model(images)              # Raw probabilities (output of the model)
                probs = torch.sigmoid(logits)       # Binary predictions (sigmoid with threshold 0.5)
                
                # ======== LOSS MANAGEMENT ========
                dice_l = compute_dice_loss(probs, masks)    # Dice Loss (Main loss of our model)

                # Manage losses
                if args.multi_loss:
                    # Focal Loss (Pixel-level classification focus)
                    focal = sigmoid_focal_loss(logits, masks, alpha=args.alpha, gamma=args.gamma, reduction='mean')
                
                    if args.normalize_loss:
                        # Update EMA running means
                        if focal_running_mean is None:
                            focal_running_mean = focal.item()
                            dice_running_mean  = dice_l.item()
                        else:
                            focal_running_mean = EMA_MOMENTUM * focal_running_mean + (1 - EMA_MOMENTUM) * focal.item()
                            dice_running_mean  = EMA_MOMENTUM * dice_running_mean  + (1 - EMA_MOMENTUM) * dice_l.item()
                        focal_normalized = focal / (focal_running_mean + 1e-6)
                        dice_normalized  = dice_l / (dice_running_mean  + 1e-6)
                        
                        # Combined loss with normalization, will always gravitate around 2
                        loss = focal_normalized + dice_normalized
                    else:
                        # Combined loss, no normalization(raw) — AdamW handles implicit gradient scaling
                        loss = focal + dice_l
                
                else:
                    # Single loss scenario (we use pure dice)
                    focal = None
                    loss  = dice_l



            # Scaler is used to ensure gradient calculations work well 
            # even with torch.autocast and 16 bits precision
            scaler.scale(loss).backward()   # Increases loss multiplying it by a big number, then makes backpropagation calculating gradients
            scaler.step(optimizer)          # Divides the gradients by the initial big number
            scaler.update()                 # Adjust the big number for the next epoch
            

            # Update total training losses (sum loss on every batch of the train loader)
            train_loss              += loss.item()  
            train_dice_loss_val     += dice_l.item()
            train_focal_loss        += focal.item() if focal is not None else 0.0

            train_loop.set_postfix(loss=loss.item())
        
        # Find the average loss across the whole epoch
        epoch_avg_train_focal_loss = train_focal_loss / len(train_loader)
        epoch_avg_train_dice_loss = train_dice_loss_val / len(train_loader)
        epoch_avg_train_loss = train_loss / len(train_loader)


        # ======== VALIDATION PHASE ========
        # NOTE: If the validation frequency argument -vf is set to 0 we only make a check on the last epoch
        # When we have validation frequency we can use things like early stopping or learning rate scheduler

        # Initialize to None to prevent crashes when skipping validation
        epoch_avg_val_loss = None 
        epoch_avg_val_dice_loss = None
        epoch_avg_val_focal_loss = None

        epoch_val_precision = None
        epoch_val_recall    = None
        epoch_val_f1        = None
        epoch_val_iou       = None

        epoch_avg_val_dice_score = None
        current_aurc = None

        # Flag for validation epochs
        is_val_epoch = False

        # Finds out if it's a validation epoch (an epoch where we check on validation set)
        if args.val_freq > 0 and (epoch + 1) % args.val_freq == 0:
            is_val_epoch = True
        elif (epoch + 1) == args.epochs:
            # We always check results on the validation set at the last epoch
            is_val_epoch = True

        # ======== VALIDATION LOOP  ========    
        if is_val_epoch:
            # Set model in evaluatin mode
            model.eval()

            # Initializing losses for the epoch
            val_loss = 0.0
            val_focal_loss = 0.0
            val_dice_loss = 0.0 

            # Initializing metrics (dice score, auroc)
            val_dice_score = 0.0
            auroc_metric = BinaryAUROC(thresholds=200).to(DEVICE)
            # Accumulators for precision, recall, f1
            val_TP = 0.0    # True Positives
            val_FP = 0.0    # False Positives
            val_FN = 0.0    # False Negatives
            
            # Validation dataloader
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]", colour = 'blue')
            
            with torch.no_grad():
                for images, masks in val_loop:
                    images = images.to(DEVICE)
                    masks  = masks.to(DEVICE, dtype=torch.float32)

                    if args.is_patch_based:
                        # ======== SLIDING WINDOW INFERENCE (patch based training) ========
                        # We are not resizing images and dataloader can't get images of different sizes
                        # --> We force batch size = 1.
                        # We still use a "batch approach" to save some code later

                        logits_list = []     # Raw predictions of the model, unbounded values (original image dimension)
                        probs_list  = []     # Predicted heatmaps of the batch (original image dimension)
                        masks_list  = []     # Predicted binary masks of the batch (original image dimension)

                        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                            # images ha shape [1, C, H, W] a risoluzione originale
                            # (val_loader ha batch_size=1 e apply_resize=False in patch mode)
                        
                            logits = sliding_window_inference(
                                inputs=images,
                                roi_size=(PATCH_SIZE, PATCH_SIZE),
                                sw_batch_size=4,
                                predictor=model,
                                overlap=0.5,
                                mode='gaussian',
                                sw_device=torch.device(DEVICE),
                                device=torch.device(DEVICE),
                            )

                        # Align mask to logits resolution if compressed masks differ
                        if masks.shape[-2:] != logits.shape[-2:]:
                            masks = F.interpolate(masks, size=logits.shape[-2:], mode='nearest')

                        logits_list.append(logits.squeeze(0))                # [1, H, W] - Raw model predictions, unbound values
                        probs_list.append(torch.sigmoid(logits).squeeze(0))  # [1, H, W] - Heatmap predictions of the model (logits after sigmoid, [0, 1] range of values)
                        masks_list.append(masks.squeeze(0))                  # [1, H, W] - True binary mask

                    else:
                        # ======== RESIZE PATH ========
                        # Inference at RESIZE_SIZE X RESIZE_SIZE, compute loss and metrics at RESIZE_SIZE X RESIZE_SIZE
                        # NOTE: unlike the test file, we do NOT upsample to original size here.
                        # Validation metrics are computed at RESIZE_SIZE X RESIZE_SIZE — consistent across epochs.
                        logits_list = []
                        probs_list  = []
                        masks_list  = []

                        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                            logits = model(images)  # [B, 1, RESIZE_SIZE, RESIZE_SIZE]

                        for i in range(images.size(0)):
                            logits_list.append(logits[i].float())                # [1, RESIZE_SIZE, RESIZE_SIZE]
                            probs_list.append(torch.sigmoid(logits[i].float()))  # [1, RESIZE_SIZE, RESIZE_SIZE]
                            masks_list.append(masks[i])                          # [1, RESIZE_SIZE, RESIZE_SIZE]

                    # ======== SHARED: loss and metrics logic is the same wether we resize or not ========
                    for i in range(len(probs_list)):
                        logit = logits_list[i].unsqueeze(0) # [1, 1, H, W] - Raw unbounded predictions, used for focal_loss 
                        prob = probs_list[i].unsqueeze(0)   # [1, 1, H, W] - Heatmap prediction of the model
                        mask = masks_list[i].unsqueeze(0)   # [1, 1, H, W] - Original mask

                        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                            # ======== LOSS MANAGEMENT ========
                            dice_l = compute_dice_loss(prob, mask)

                            if args.multi_loss:
                                # sigmoid_focal_loss expects logits (automatically applies sigmoid to input)
                                focal = sigmoid_focal_loss(logit, mask, alpha=args.alpha, gamma=args.gamma, reduction='mean')
                                if args.normalize_loss:
                                    focal_normalized = focal / (focal_running_mean + 1e-6)
                                    dice_normalized  = dice_l / (dice_running_mean  + 1e-6)
                                    loss = focal_normalized + dice_normalized
                                else:
                                    loss = focal + dice_l
                            else:
                                focal = None
                                loss  = dice_l

                        # --- Update global accumulators for metrics ---
                        dice_score_batch, TP, FP, FN = calculate_dice(prob, mask, return_components=True)
                        auroc_metric.update(prob, mask.long())
                        val_TP += TP.item()
                        val_FP += FP.item()
                        val_FN += FN.item()

                        # --- Update global accumulators for losses ---
                        val_loss      += loss.item()
                        val_dice_loss += dice_l.item()
                        val_focal_loss += focal.item() if focal is not None else 0.0
                        val_dice_score += dice_score_batch.item()

                    val_loop.set_postfix(loss=loss.item())


            # ======== Find out average values of Losses for the validation epoch ========
            # Total number of individual images processed
            total_val_images = len(val_loader.dataset)

            # Average value of losses for the validation epoch
            epoch_avg_val_loss       = val_loss       / total_val_images
            epoch_avg_val_focal_loss = val_focal_loss / total_val_images
            epoch_avg_val_dice_loss  = val_dice_loss  / total_val_images

            # Find out average values of Metrics for the validation epoch
            epoch_avg_val_dice_score = val_dice_score / total_val_images
            eps = 1e-6  # Small term to avoid division by 0
            epoch_val_precision = val_TP / (val_TP + val_FP + eps)
            epoch_val_recall    = val_TP / (val_TP + val_FN + eps)
            epoch_val_f1        = 2 * epoch_val_precision * epoch_val_recall / (epoch_val_precision + epoch_val_recall + eps)
            epoch_val_iou       = val_TP / (val_TP + val_FP + val_FN + eps)
            current_aurc        = auroc_metric.compute().item()

            print(f">>> Epoch {epoch+1} Results - Train Loss: {epoch_avg_train_loss:.4f} | Val Loss: {epoch_avg_val_loss:.4f}")


            # -------------------------------------------------------------------------
            # SCHEDULER & CHECKPOINTING LOGIC (Driven by Validation Loss, only if validation freq is > 0)
            # -------------------------------------------------------------------------
            # If val freq is 0 we only check validation loss at last epoch, 
            # so there is no point in using a scheduler
            if args.val_freq > 0:

                scheduler.step(epoch_avg_val_loss)
        
                # Relative MIN_DELTA (0.5% improvement required to count as best epoch)
                RELATIVE_IMPROVEMENT = 0.005

                if args.multi_loss and args.normalize_loss:
                    # Combined loss is not comparable across runs due to EMA normalization;
                    # use the raw dice loss component instead (always on the same 0-1 scale)
                    current_best = best_val_dice_loss
                    current_val  = epoch_avg_val_dice_loss
                    # Threshold to count as real improvement, handles the initial case where loss is infinite
                    required_improvement = 0 if current_best == float('inf') else (current_best * RELATIVE_IMPROVEMENT) 

                    is_improvement = current_val < (current_best - required_improvement)
                else:
                    # We use raw loss (multiple or single, same principle)
                    current_best = best_val_loss
                    current_val  = epoch_avg_val_loss
                    required_improvement = 0 if current_best == float('inf') else (current_best * RELATIVE_IMPROVEMENT) 

                    is_improvement = current_val < (current_best - required_improvement)


                if is_improvement:

                    if args.multi_loss and args.normalize_loss:
                        best_val_dice_loss = current_val
                    else:
                        best_val_loss = current_val

                    best_epoch = epoch + 1  # track which epoch was actually the best

                    # NOTE: We only check validation loss on validation epochs (depends on validation freq argument -vf)
                    # Our patience for early stopping is based on checks (1 check = -vf epochs)

                    # If a new best is found reset the counter for patience
                    checks_without_improvement = 0  
                
                    # Save the new Best Model
                    best_checkpoint_path = os.path.join(SAVE_DIR, "dinolizer_best.pth")
                    torch.save({
                        # General details about the model
                        'model_type':     type(model).__name__,
                        'dataset':        args.dataset_dir,
                        'model_dir':      args.model_dir,
                        'patch_based':    args.is_patch_based,
                        'patch_size':     PATCH_SIZE if args.is_patch_based else None,
                        'resize_size':    RESIZE_SIZE if not args.is_patch_based else None,
                        'multi_loss':     args.multi_loss,
                        'normalize_loss': args.normalize_loss,
                        'augmentation':   args.augmentation,
                        'learning_rate':  args.learning_rate,
                        'batch_size':     args.batch_size,
                        # Details for consistency
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # Current epoch training losses
                        'train_loss': epoch_avg_train_loss,
                        'train_dice_loss': epoch_avg_train_dice_loss,
                        'train_focal_loss': epoch_avg_train_focal_loss,
                        # Current epoch validation losses
                        'val_loss': epoch_avg_val_loss,
                        'val_dice_loss': epoch_avg_val_dice_loss,
                        'val_focal_loss': epoch_avg_val_focal_loss,
                        # Validation metrics (None if not a val epoch)
                        'val_dice_score': epoch_avg_val_dice_score,
                        'val_precision':  epoch_val_precision if is_val_epoch else None,
                        'val_recall':     epoch_val_recall    if is_val_epoch else None,
                        'val_f1':         epoch_val_f1        if is_val_epoch else None,
                        'val_iou':        epoch_val_iou       if is_val_epoch else None,
                        'val_auroc':      current_aurc        if is_val_epoch else None,
                        # Global best losses
                        'best_val_loss': best_val_loss,                 # Best loss we track if normalization is not used
                        'best_val_dice_loss': best_val_dice_loss,       # Best loss we track if normalization is used
                        'best_epoch': best_epoch,
                        # EMA parameters
                        'focal_running_mean': focal_running_mean,
                        'dice_running_mean':  dice_running_mean,
                        # Focal loss parameters
                        'alpha': args.alpha,
                        'gamma': args.gamma,
                    }, best_checkpoint_path)
                    print(f">>> New best epoch! Metric: {current_val:.4f} (prev: {current_best:.4f}) — saved dinolizer_best.pth")
                else:
                    checks_without_improvement += 1
                    print(f">>> No improvement. Current: {current_val:.4f} | Best epoch: {best_epoch:.4f} | Best: {current_best:.4f} | Required delta: {required_improvement:.4f} | Early stopping: {checks_without_improvement}/{EARLY_STOPPING_PATIENCE}")


            # Early stopping if we had too many checks without improvement (0.05% needed)
            if checks_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(">>> Early stopping triggered! Training stopped.")
                break
        
        
        # If this is not a validation epoch
        else:
            print(f">>> Epoch {epoch+1} Results - Train Loss: {epoch_avg_train_loss:.4f} | Val Loss: Skipped")
        
        
        # ======== UPDATE CSV LOG ========
        with open(csv_log_path, mode='a') as f:
            def _fmt(v): return f"{v:.6f}" if v is not None else ""
            
            f.write(
                f"{epoch+1},"
                # Training losses
                f"{_fmt(epoch_avg_train_loss)},{_fmt(epoch_avg_train_focal_loss)},{_fmt(epoch_avg_train_dice_loss)},"
                # Validation losses
                f"{_fmt(epoch_avg_val_loss)},{_fmt(epoch_avg_val_focal_loss)},{_fmt(epoch_avg_val_dice_loss)},"
                # Validation metrics
                f"{_fmt(epoch_avg_val_dice_score)},"
                f"{_fmt(epoch_val_precision)},{_fmt(epoch_val_recall)},{_fmt(epoch_val_f1)},{_fmt(epoch_val_iou)},"
                f"{_fmt(current_aurc if is_val_epoch else None)}\n"
            )        


        # ======== SAVE THE CURRENT EPOCH STATE ========
        checkpoint_path = os.path.join(SAVE_DIR, f"dinolizer_epoch_{epoch+1}.pth")

        # Save separate metrics. best_val_loss safely stays 'inf' until a real check happens.
        torch.save({
            # General details about the model
            'model_type':     type(model).__name__,
            'dataset':        args.dataset_dir,
            'model_dir':      args.model_dir,
            'patch_based':    args.is_patch_based,
            'patch_size':     PATCH_SIZE if args.is_patch_based else None,
            'resize_size':    RESIZE_SIZE if not args.is_patch_based else None,
            'multi_loss':     args.multi_loss,
            'normalize_loss': args.normalize_loss,
            'augmentation':   args.augmentation,
            'learning_rate':  args.learning_rate,
            'batch_size':     args.batch_size,
            # Details for consistency
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Current epoch training losses
            'train_loss': epoch_avg_train_loss,
            'train_dice_loss': epoch_avg_train_dice_loss,
            'train_focal_loss': epoch_avg_train_focal_loss,
            # Current epoch validation losses
            'val_loss': epoch_avg_val_loss,
            'val_dice_loss': epoch_avg_val_dice_loss,
            'val_focal_loss': epoch_avg_val_focal_loss,
            # Global best losses
            'best_val_loss': best_val_loss,             # Best loss we track if normalization is not used
            'best_val_dice_loss': best_val_dice_loss,   # Best loss we track if normalization is used
            'best_epoch': best_epoch,
            # Validation metrics (None if not a val epoch)
            'val_dice_score': epoch_avg_val_dice_score,
            'val_precision':  epoch_val_precision if is_val_epoch else None,
            'val_recall':     epoch_val_recall    if is_val_epoch else None,
            'val_f1':         epoch_val_f1        if is_val_epoch else None,
            'val_iou':        epoch_val_iou       if is_val_epoch else None,
            'val_auroc':      current_aurc        if is_val_epoch else None,
            # EMA parameters
            'focal_running_mean': focal_running_mean,
            'dice_running_mean':  dice_running_mean,
            # Focal loss parameters
            'alpha': args.alpha,
            'gamma': args.gamma,
        }, checkpoint_path)


        # ======== Update details.json with details of the new epoch we just saved ========
        epoch_details = extract_epoch_details(checkpoint_path)
        save_details_json(epoch_details, SAVE_DIR)
        
        # ======== ROTATE THE LAST SAVED EPOCH (Keep only the last one we just saved for disk space safety) ========
        old_checkpoint_path = os.path.join(SAVE_DIR, f"dinolizer_epoch_{epoch}.pth")
        if os.path.exists(old_checkpoint_path):
            os.remove(old_checkpoint_path)

if __name__ == '__main__':

    parser = get_parser()
    main(parser.parse_args())