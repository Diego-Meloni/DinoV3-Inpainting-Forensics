import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F_VIS

import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoModel
import warnings

# ===================================================================================
#                 PATCH GETTER, JOINT TRANSFORMER, DINOLIZER DATASET [CSV]
# ===================================================================================

class JointTransform():
    def __init__(self, size=(512, 512), augment=True, apply_resize= True, preprocess=True, prob=0.3, normalize=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.size = size
        self.augment = augment
        self.apply_resize = apply_resize
        self.preprocess = preprocess
        self.prob = prob
        self.normalize = normalize
        
        #all pre-trained PyTorch models expect inputs normalized with these mean and std values, given as default in the __init__ input arguments.
        self.normalizer = T.Normalize(mean, std)
        
        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        
        self.gaussian_blur = T.GaussianBlur(kernel_size=(3,3), sigma=(0.5, 2.0))
        
    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        img_resize = T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC)
        mask_resize = T.Resize(self.size, interpolation=T.InterpolationMode.NEAREST)
        image = img_resize(image)
        mask = mask_resize(mask)
        
        return image, mask
        
    def _random_horizontal_flip(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if torch.rand(1) < self.prob:
            image = F_VIS.horizontal_flip(image)
            mask = F_VIS.horizontal_flip(mask)
            
        return image, mask
    
    def _random_vertical_flip(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if torch.rand(1) < self.prob:
            image = F_VIS.vertical_flip(image)
            mask = F_VIS.vertical_flip(mask)
        
        return image, mask
    
    def _random_rotate(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        #small rotation, maximum angle is 15 degrees; fill colour is black (0) for both.
        angle = float(torch.empty(1).uniform_(-15, 15))
        image  = F_VIS.rotate(image,  angle, fill=0)
        mask = F_VIS.rotate(mask, angle, fill=0)
        return image, mask
    
    def _random_crop(self, image:Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        #crop then resize back to original input size
        scale = float(torch.empty(1).uniform_(0.75, 1.0))
        crop_size = int(self.size[0] * scale)
        i, j, h, w = T.RandomCrop.get_params(image, (crop_size, crop_size))
        image  = F_VIS.resized_crop(image,  i, j, h, w, self.size, interpolation=T.InterpolationMode.BICUBIC)
        mask = F_VIS.resized_crop(mask, i, j, h, w, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, mask
    
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        # if the mask is smaller (or bigger) than the original image, we resize it.
        if image.size != mask.size and not self.apply_resize:
            mask = mask.resize(image.size, resample=Image.NEAREST)
        
        # resize the image and the mask.
        if self.apply_resize:
            image, mask = self._resize(image, mask)
        
        #if self.augment = True apply joint transformations of both image and mask.
        if self.augment:
            image, mask = self._random_horizontal_flip(image, mask)
            image, mask = self._random_vertical_flip(image, mask)
            # image, mask = self._random_crop(image, mask)
            if torch.rand(1) < self.prob:
                image, mask = self._random_rotate(image, mask)
            if torch.rand(1) < self.prob:
                image = self.color_jitter(image)
            if torch.rand(1) < self.prob:
                image = self.gaussian_blur(image)
            
            
        if self.preprocess:
            if type(image) != torch.Tensor:
                image = F_VIS.to_tensor(image)
            if self.normalize:
                image = self.normalizer(image)
            if type(mask) != torch.Tensor:
                mask = F_VIS.to_tensor(mask)
                mask = (mask > 0.5).float()
            
        return image, mask


class PatchGetter():
    def __init__(self, patch_size=256, num_patches=16, stride=128, ratios={'boundary': 0.5, 'fully_inpaint': 0.3, 'background': 0.2}):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.ratios = ratios
        self.stride = stride
        
    def unfold_classify(self, image: torch.Tensor, mask: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        C, H, W = image.shape
        C_mask, _, _ = mask.shape
        pad_h = max(0, self.patch_size - H)
        pad_w = max(0, self.patch_size - W)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h))
            mask  = F.pad(mask,  (0, pad_w, 0, pad_h))
            _, H, W = image.shape
            
        edge_pad_h = (self.stride - (H - self.patch_size) % self.stride) % self.stride
        edge_pad_w = (self.stride - (W - self.patch_size) % self.stride) % self.stride
        if edge_pad_h > 0 or edge_pad_w > 0:
            image = F.pad(image, (0, edge_pad_w, 0, edge_pad_h))
            mask  = F.pad(mask,  (0, edge_pad_w, 0, edge_pad_h))
            
        img_unfold = torch.nn.functional.unfold(image.unsqueeze(0), kernel_size=self.patch_size, stride=self.stride)
        mask_unfold = torch.nn.functional.unfold(mask.unsqueeze(0), kernel_size=self.patch_size, stride=self.stride)
        
        #take the value of the number of patches to later use it as the batch dimesion.
        N = img_unfold.shape[-1]
        
        img_patches  = img_unfold.squeeze(0).permute(1, 0).view(N, C, self.patch_size, self.patch_size)
        mask_patches  = mask_unfold.squeeze(0).permute(1, 0).view(N, C_mask, self.patch_size, self.patch_size)
        
        #classify patches using percentage of foreground and background inside the mask.
        #low foreground: foreground < 20%
        #mid foreground: 40% < foreground < 60%
        #high foreground: foreground > 80%
        
        mask_flat = mask_patches.view(N, -1)
        mask_ratios = mask_flat.mean(dim=1)
        is_low_foreground = (mask_ratios <= 0.2)
        is_mid_foreground = (mask_ratios > 0.2) & (mask_ratios < 0.8)
        is_high_foreground = (mask_ratios >= 0.8)
        
        return {
            'boundary': (img_patches[is_mid_foreground], mask_patches[is_mid_foreground]),
            'fully_inpainted': (img_patches[is_high_foreground], mask_patches[is_high_foreground]),
            'background': (img_patches[is_low_foreground], mask_patches[is_low_foreground]),
        }
        
    def sample_patches(self, images: torch.Tensor, masks: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        #sample k patches from images and masks using replacement leveraging torch.randint.
        idx = torch.randint(len(images), (k,))
        return images[idx], masks[idx]

    def get_patches(self, image: torch.Tensor, mask: torch.Tensor):
        #extract and classify all patches in the image using unfold_classify function.
        patch_dict = self.unfold_classify(image, mask)
        boundary_imgs, boundary_masks = patch_dict['boundary']
        full_imgs, full_masks = patch_dict['fully_inpainted']
        bg_imgs, bg_masks = patch_dict['background']
        
        #compute per-category sample counts from self.ratios.
        num_boundary      = math.floor(self.num_patches * self.ratios['boundary'])
        num_full_inpaint  = math.floor(self.num_patches * self.ratios['fully_inpaint'])
        num_background    = math.floor(self.num_patches * self.ratios['background'])
        remainder = self.num_patches - (num_boundary + num_full_inpaint + num_background)
        num_boundary += remainder
        
        non_empty_imgs  = [t for t in [boundary_imgs, full_imgs, bg_imgs]  if len(t) > 0]
        non_empty_masks = [t for t in [boundary_masks, full_masks, bg_masks] if len(t) > 0]
        pool_imgs  = torch.cat(non_empty_imgs,  dim=0)
        pool_masks = torch.cat(non_empty_masks, dim=0)
        
        #sample from each category, fall back to non_empty if category is empty.
        def resolve(cat_images, cat_masks):
            return (cat_images, cat_masks) if len(cat_images) > 0 else (pool_imgs, pool_masks)
        
        bou_imgs, bou_masks = self.sample_patches(*resolve(boundary_imgs, boundary_masks), num_boundary)
        fi_imgs, fi_masks = self.sample_patches(*resolve(full_imgs, full_masks), num_full_inpaint)
        bk_imgs, bk_masks = self.sample_patches(*resolve(bg_imgs, bg_masks), num_background)

        image = torch.cat([bou_imgs,  fi_imgs,  bk_imgs],  dim=0)
        mask  = torch.cat([bou_masks, fi_masks, bk_masks], dim=0)
        
        return image, mask


class DinoCSVDataset(Dataset):
    """
    Reads image paths from the CSV and loads them into RAM only when requested.
    Expected CSV columns: ['image_path', 'mask_path']
    """
    def __init__(self, csv_file, base_dir, transform=None, use_patches=False, patch_size=256, num_patches=10):
        self.data = pd.read_csv(csv_file)
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # --- Warning logic ---
        # If our transformer has the attribute 'apply_resize' set True with Patch training
        if self.use_patches and getattr(self.transform, 'apply_resize', False):
            warnings.warn(
                "\n\n[WARNING] You have parameter use_patches=True but the transformer is applying resize (apply_resize=True).\n"
                "The image will be rescaled BEFORE extracting the patches.\n"
                "For big images this could be fine, but if images are low resolution it "
                "Would be better to set apply_resize=False on the transformer.\n"
            )
        
        if self.use_patches:
            self.stride = self.patch_size // 2
            # For each image we get num_patches(default=10) patches of size 256x256
            self.patch_getter = PatchGetter(
                patch_size= self.patch_size, 
                num_patches=self.num_patches, 
                stride=self.stride)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract relative paths and resolve them to absolute paths
        img_rel_path = self.data.iloc[idx]['image_path']
        mask_rel_path = self.data.iloc[idx]['mask_path']
 
        img_path = self.base_dir / img_rel_path
        mask_path = self.base_dir / mask_rel_path
        
        # Open files using PIL (required by JointTransform)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image_tensor, mask_tensor = self.transform(image, mask)

        if self.use_patches:
            # Extract num_patches(default=10) balanced patches and return them
            patches_img, patches_mask = self.patch_getter.get_patches(image_tensor, mask_tensor)
            return patches_img, patches_mask
            
        return image_tensor, mask_tensor


# ===================================================================================
#                         DINOLIZER ARCHITECTURES [Transformer]
# ===================================================================================

# ======== DinoLizer base architecture (linear head) ========
class DinoLizer(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        # The dino model outputs a vector of hidden_dim values (ex 768)
        # describing the (16x16) patch
        hidden_dim = self.backbone.config.hidden_size 

        # Keep the layers of the dino model frozen
        for param in self.backbone.parameters():
            param.requires_grad = False
                 
        # --- SIMPLIFIED HEAD ---
        # This is the approach of the original paper.
        # We project directly from DINO features to the 1D heatmap logit.
        self.head = nn.Sequential(
            nn.Dropout(p=0.25), # Deactivate a quarter of neurons per batch 
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, pixel_values):
        # Pass our batch of images/patches into the dino transformer
        outputs = self.backbone(pixel_values=pixel_values)

        # b = batch size (ex b=4 --> 4 images, if 16 images per patch --> 64 patches for a batch);
        # h, w are the dimension we choose for our transformer (ex: 320, 320)  
        b, c, h, w = pixel_values.shape

        # dino cuts images in 16x16 squares
        grid_h, grid_w = h // 16, w // 16   # 320 // 16 = 20 --> 20 x 20 grid
        num_squares = grid_h * grid_w   # 400 squares
        
        # Dino outputs more than our 400 squares, in the beginning it has some cls tokens, register tokens for memory,...
        # we don't want to consider them, therefore 
        # for all images in the batch (:), take only the last 400 items (-num_squares), and take all their features(:)
        square_tokens = outputs.last_hidden_state[:, -num_squares:, :] 

        # We pass the 400 squares in a classification head which outputs fake/real scores
        logits = self.head(square_tokens)
        # We convert the full vector in a 20x20 grid of probabilities
        raw_heatmap = logits.view(b, 1, grid_h, grid_w)

        # When using sliding window / squares we want to upsample directly in our predictions (heatmap same size as original img)
        heatmap = F.interpolate(raw_heatmap, size=(h, w), mode='bilinear', align_corners=False)
        return heatmap


# ======== DinoLizer MPL ========
class DinoLizerMLP(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        # The dino model outputs a vector of hidden_dim values (ex 768)
        # describing the (16x16) patch
        hidden_dim = self.backbone.config.hidden_size 

        # Keep the layers of the dino model frozen
        for param in self.backbone.parameters():
            param.requires_grad = False

        # We use more than a single linear layer: Linear -> Activation -> Linear
        self.head = nn.Sequential(
            nn.Dropout(p=0.3), # Drops 30% of features to prevent memorization and overfitting
            # Input of hidden_dim and output of hidden_dim // 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), # Activation function, smoother than Relu
            nn.Dropout(p=0.3), # Additional regularization before final output
            nn.Linear(hidden_dim // 2, 1)   # Outputs a single number determining the probability of patch inpainted
        )
        
    def forward(self, pixel_values):
        # Pass our batch of images/patches into the dino transformer
        outputs = self.backbone(pixel_values=pixel_values)

        # b = batch size (ex b=4 --> 4 images, if 16 images per patch --> 64 patches for a batch);
        # h, w are the dimension we choose for our transformer (ex: 320, 320)  
        b, c, h, w = pixel_values.shape

        # dino cuts images in 16x16 squares
        grid_h, grid_w = h // 16, w // 16   # 320 // 16 = 20 --> 20 x 20 grid
        num_squares = grid_h * grid_w   # 400 squares
        
        # Dino outputs more than our 400 squares, in the beginning it has some cls tokens, register tokens for memory,...
        # we don't want to consider them, therefore 
        # for all images in the batch (:), take only the last 400 items (-num_squares), and take all their features(:)
        square_tokens = outputs.last_hidden_state[:, -num_squares:, :] 

        # We pass the 400 squares in a classification head which outputs fake/real scores
        logits = self.head(square_tokens)
        # We convert the full vector in a 20x20 grid of probabilities
        raw_heatmap = logits.view(b, 1, grid_h, grid_w)

        # When using sliding window / squares we want to upsample directly in our predictions (heatmap same size as original img)
        heatmap = F.interpolate(raw_heatmap, size=(h, w), mode='bilinear', align_corners=False)
        return heatmap
    
        # We used to return the 16x16 raw_heatmap and rescale it during training/inference
        # Now we do it directly in the forward of the model   
        # return raw_heatmap
    

# ======== DinoLizer CNN ========
class DinoLizerCNN(nn.Module):
    """
    Uses a convolutional neural network (CNN) to maintain some
    spatial information about neighboring patches of pixels. 
    """
    def __init__(self, model_dir):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        
        # 100% Frozen Backbone: We return to the stable baseline
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        hidden_dim = self.backbone.config.hidden_size 

        # --- SPATIAL HEAD ---
        # We use a "bottleneck" structure to help reduce overfitting:
        # 1. A 1x1 Conv reduces DINO's massive feature channels (e.g. 384) down to a tiny 64.
        # 2. A 3x3 Conv performs spatial reasoning on this compressed representation.
        # This reduces the parameter count by 90%, physically preventing memorization.
        self.head = nn.Sequential(
            # Stage 1: channel reduction (1x1 = per-patch, no spatial mixing yet)
            nn.Conv2d(hidden_dim, 64, kernel_size=1),   # ~24,576 params for dim=384
            nn.GELU(),
            nn.Dropout2d(p=0.3),
            # Stage 2: spatial context (now we mix neighboring patches)
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # 18,432 params
            nn.GELU(),
            nn.Dropout2d(p=0.3),
            # Stage 3: final prediction
            nn.Conv2d(32, 1, kernel_size=1)              # 32 params
        )
        
    
    def forward(self, pixel_values):
        # Pass our batch of images/patches into the dino transformer
        outputs = self.backbone(pixel_values=pixel_values)

        # b = batch size (ex b=4 --> 4 images, if 16 images per patch --> 64 patches for a batch);
        # h, w are the dimension we choose for our transformer resize/patches (ex: 320, 320)
        b, c, h, w = pixel_values.shape

        # dino cuts images in 16x16 squares
        grid_h, grid_w = h // 16, w // 16   # 320 // 16 = 20 --> 20 x 20 grid
        num_squares = grid_h * grid_w       # 400 squares
        
        # Dino actually outputs more than our 400 squares, in the beginning it has some cls tokens, register tokens for memory,...
        # we don't want to consider them, therefore 
        # for all images in the batch (:), take only the last 400 items (-num_squares), and take all their features(:)
        square_tokens = outputs.last_hidden_state[:, -num_squares:, :] 
        
        # --- Spatial transformation for conv2d ---
        # From: [batch, 400, 768] (flat list of tokens)
        # To:  [batch, 768, 20, 20] (spatial grid)
        square_grid = square_tokens.transpose(1, 2).view(b, -1, grid_h, grid_w)
        
        # passes the grid into a classification head
        raw_heatmap = self.head(square_grid) 

        
        # When using sliding window / patches we want to upsample directly in our predictions (heatmap same size as original img)
        heatmap = F.interpolate(raw_heatmap, size=(h, w), mode='bilinear', align_corners=False)
        return heatmap
        
        # We used to return the 16x16 raw_heatmap and rescale it during training/inference
        # Now we do it directly in the forward of the model
        # return raw_heatmap


# ===================================================================================
#                         DINOLIZER CONVNEXT ARCHITECTURES [CNN]
# ===================================================================================

class DinoLizerConvNext(nn.Module):
    """
    DinoLizer variant using DINOv3 ConvNext backbone.
    Key differences from DinoLizer (ViT):
    - hidden_dim comes from config.hidden_sizes[-1] (last stage output, 768 for both tiny and small)
    - Effective patch size is 32x32 instead of 16x16
    - Only 1 non-spatial token instead of 5 (1 cls + 4 registers)
    """
    def __init__(self, model_dir):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

        # Last element of hidden_sizes is the output feature dimension of the final stage
        hidden_dim = self.backbone.config.hidden_sizes[-1]  # 768 for both tiny and small

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        b, c, h, w = pixel_values.shape

        # ConvNext uses 32x32 patches instead of ViT's 16x16
        grid_h, grid_w = h // 32, w // 32
        num_squares = grid_h * grid_w

        # Skip the first token (non-spatial), take the last num_squares spatial tokens
        square_tokens = outputs.last_hidden_state[:, -num_squares:, :]

        logits = self.head(square_tokens)
        raw_heatmap = logits.view(b, 1, grid_h, grid_w)

        heatmap = F.interpolate(raw_heatmap, size=(h, w), mode='bilinear', align_corners=False)
        return heatmap


class DinoLizerConvNextMLP(nn.Module):
    """
    DinoLizer MLP head variant using DINOv3 ConvNext backbone.
    Same as DinoLizerMLP but adapted for ConvNext's 32x32 patch grid
    and 768-dimensional token embeddings.
    """
    def __init__(self, model_dir):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

        # ConvNext uses hidden_sizes list, not hidden_size scalar
        hidden_dim = self.backbone.config.hidden_sizes[-1]  # 768

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        b, c, h, w = pixel_values.shape

        # ConvNext effective patch size is 32x32
        grid_h, grid_w = h // 32, w // 32
        num_squares = grid_h * grid_w

        # Skip first token (non-spatial), take spatial tokens
        square_tokens = outputs.last_hidden_state[:, -num_squares:, :]

        logits = self.head(square_tokens)
        raw_heatmap = logits.view(b, 1, grid_h, grid_w)

        heatmap = F.interpolate(
            raw_heatmap, size=(h, w),
            mode='bilinear', align_corners=False
        )
        return heatmap


class DinoLizerConvNextCNN(nn.Module):
    """
    DinoLizer CNN head variant using DINOv3 ConvNext backbone.
    Same as DinoLizerCNN but adapted for ConvNext's 32x32 patch grid.
    """
    def __init__(self, model_dir):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

        hidden_dim = self.backbone.config.hidden_sizes[-1]  # 768

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        b, c, h, w = pixel_values.shape

        grid_h, grid_w = h // 32, w // 32
        num_squares = grid_h * grid_w

        square_tokens = outputs.last_hidden_state[:, -num_squares:, :]

        # Reshape from [B, num_squares, hidden_dim] to [B, hidden_dim, grid_h, grid_w]
        square_grid = square_tokens.transpose(1, 2).view(b, -1, grid_h, grid_w)

        raw_heatmap = self.head(square_grid)

        heatmap = F.interpolate(raw_heatmap, size=(h, w), mode='bilinear', align_corners=False)
        return heatmap