import torch
import numpy as np
import supervision as sv
import cv2
import gc
import torchvision
from PIL import Image
import torch.nn.functional as F
import os
import joblib
from utils import ImageProcessor
from verifier import EnsembleVerifier
from tqdm import tqdm
import torch.optim as optim
from torch.cuda.amp import GradScaler
# Grounding DINO Imports
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict

from transformers import CLIPProcessor, CLIPModel

# SAM 2 Imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# For DinoV2
from torchvision import transforms
from torchvision.ops import box_iou

from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV

import warnings
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Suppress all warnings
warnings.filterwarnings("ignore")

class ModelManager:
    """
    Manages VRAM resources. Ensures only one model is on the GPU at a time,
    but prevents unnecessary unloading/reloading if the same model is requested twice.
    """
    def __init__(self, gd_model, sam_model, sam_predictor, dino_model, clip_model, clip_processor, device="cuda"):
        self.device = device
        self.cpu = "cpu"
        
        self.models = {
            "gd": gd_model,
            "sam": sam_model,
            "dino": dino_model,
            "clip": clip_model
        }
        self.sam_predictor = sam_predictor
        self.clip_processor = clip_processor
        self.current_key = None

    def _flush_vram(self):
        torch.cuda.empty_cache()
        gc.collect()

    def switch_to(self, key):
        """
        Switches the active model on the GPU.
        key: 'gd', 'sam', or 'dino'
        """
        if self.current_key == key:
            model = self.models[key]
            try:
                if next(model.parameters()).device.type == self.device:
                    return
            except:
                pass
        
        if self.current_key is not None:
            print(f"    [VRAM] Offloading {self.current_key}...")
            self.models[self.current_key].to(self.cpu)
            
            if self.current_key == "sam":
                self.sam_predictor.reset_predictor()
            
            self.current_key = None
            self._flush_vram()

        print(f"    [VRAM] Loading {key}...")
        self.models[key].to(self.device)
        
        if key == "sam":
            self.sam_predictor.model = self.models[key]
            
        self.current_key = key
        
    def invalidate(self, key):
        """
        Manually marks a model as 'not on GPU' (used when force_cpu moves it).
        """
        if self.current_key == key:
            self.current_key = None

class Augmentor:
    """
    Generates 4 views of every crop to make the model robust to 
    lighting and orientation changes.
    """
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def augment(self, images):
        """
        Input: Tensor (N, 3, 224, 224)
        Output: Tensor (N, 3, 224, 224) with variations
        """
        if not images: return []
        return [self.transforms(img) for img in images]

class LoRALinear(torch.nn.Module):
    """
    Wraps a frozen Linear layer with a low-rank adapter.
    Forward: W·x + scale * (B·A)·x
    Only lora_A and lora_B have requires_grad=True.
    B is zero-initialized so the adapter is a no-op at epoch 0 —
    pretrained DINOv2 features are fully preserved at the start of training.
    """
    def __init__(self, linear: torch.nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.rank   = rank
        self.scale  = alpha / rank

        in_f  = linear.in_features
        out_f = linear.out_features

        self.lora_A = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(rank, in_f))
        )
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scale

class AnnotateEngine:
    def __init__(self, device="cuda"):
        self.device = device
        self.cpu = "cpu"
        
        self.gd_config = "weights/GroundingDINO_SwinB_cfg.py"
        self.gd_weights = "weights/groundingdino_swinb_cogcoor.pth"
        self.sam_config = "sam2_hiera_s.yaml"
        self.sam_weights = "weights/sam2_hiera_small.pt"

        print("[Init] Loading model architectures...")
        gd_model = load_model(self.gd_config, self.gd_weights, device=self.cpu)
        sam_model = build_sam2(self.sam_config, self.sam_weights, device=self.cpu)
        sam_predictor = SAM2ImagePredictor(sam_model)

        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        dino_model.to(self.cpu) # Keep on CPU until needed
        dino_model.eval()
        
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        clip_model.to(self.cpu)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

        self.model_manager = ModelManager(gd_model, sam_model, sam_predictor, dino_model, clip_model, clip_processor, device)
        self.STANDARD_IMAGE_SCALE = 1280

        self.dino_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.size_thresholds = {
            'tiny': 0.0002,        # Default: < 0.02% of image
            'small': 0.002,        # Default: 0.02% - 0.2%
            'normal_max': 0.015,   # Default: 0.2% - 1.5%
            'large': 0.05,         # Default: 1.5% - 5%
            'very_large': 0.05     # Default: > 5%
        }

        self.threshold_adjustments = {
            'tiny': -0.2,
            'small': -0.1,
            'normal': 0.0,
            'large': +0.05,
            'very_large': +0.1
        }

        self.context_factors = {
            'tiny': 5.0,
            'small': 4.0,
            'normal': 3.0,
            'large': 2.5,
            'very_large': 2.0
        }
        self._dino_cache = {}   # caches patch tokens per image to avoid redundant forward passes

        print("[Init] Ready.")

    def _get_gd_transform(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def calibrate_size_thresholds(self, verified_data):
        """
        Automatically calibrate size thresholds based on training data.
        Call this after HITL annotation, before training the verifier.
        
        Args:
            verified_data: List of dicts with 'pos' and 'neg' boxes
        """
        print("\n--> [Calibration] Analyzing training data to set size thresholds...")
        
        all_pos_areas = []
        for item in verified_data:
            if len(item['pos']) > 0:
                pos_boxes = item['pos']
                if isinstance(pos_boxes, torch.Tensor):
                    pos_boxes = pos_boxes.numpy()
                
                areas = pos_boxes[:, 2] * pos_boxes[:, 3]
                all_pos_areas.extend(areas.tolist())
        
        if len(all_pos_areas) == 0:
            print("    [Warning] No positive samples found. Using default thresholds.")
            return
        
        all_pos_areas = np.array(all_pos_areas)
        
        median_area = np.median(all_pos_areas)
        p25 = np.percentile(all_pos_areas, 25)  # 25th percentile
        p75 = np.percentile(all_pos_areas, 75)  # 75th percentile
        min_area = np.min(all_pos_areas)
        max_area = np.max(all_pos_areas)
        
        print(f"    [Stats] Positive sample areas:")
        print(f"      Count: {len(all_pos_areas)}")
        print(f"      Min: {min_area:.6f}")
        print(f"      25th percentile: {p25:.6f}")
        print(f"      Median: {median_area:.6f}")
        print(f"      75th percentile: {p75:.6f}")
        print(f"      Max: {max_area:.6f}")
        
        # Tiny: Anything smaller than half the 25th percentile
        self.size_thresholds['tiny'] = max(p25 * 0.5, min_area * 0.8)
        
        # Small: From tiny to 25th percentile
        self.size_thresholds['small'] = p25
        
        # Normal max: Up to 75th percentile
        self.size_thresholds['normal_max'] = p75
        
        # Large: From 75th percentile to 2x median
        self.size_thresholds['large'] = max(median_area * 2.0, p75 * 1.5)
        
        # Very large: Anything beyond 3x median (likely FP)
        self.size_thresholds['very_large'] = median_area * 3.0
        
        print(f"    [Calibrated Thresholds]:")
        print(f"      Tiny:       < {self.size_thresholds['tiny']:.6f}")
        print(f"      Small:      {self.size_thresholds['tiny']:.6f} - {self.size_thresholds['small']:.6f}")
        print(f"      Normal:     {self.size_thresholds['small']:.6f} - {self.size_thresholds['normal_max']:.6f}")
        print(f"      Large:      {self.size_thresholds['normal_max']:.6f} - {self.size_thresholds['large']:.6f}")
        print(f"      Very Large: > {self.size_thresholds['very_large']:.6f}")
        
        # Show how many training samples fall into each category
        tiny_count = (all_pos_areas < self.size_thresholds['tiny']).sum()
        small_count = ((all_pos_areas >= self.size_thresholds['tiny']) & 
                    (all_pos_areas < self.size_thresholds['small'])).sum()
        normal_count = ((all_pos_areas >= self.size_thresholds['small']) & 
                        (all_pos_areas < self.size_thresholds['normal_max'])).sum()
        large_count = ((all_pos_areas >= self.size_thresholds['normal_max']) & 
                    (all_pos_areas < self.size_thresholds['very_large'])).sum()
        very_large_count = (all_pos_areas >= self.size_thresholds['very_large']).sum()
        
        print(f"    [Training Distribution]:")
        print(f"      Tiny:       {tiny_count}/{len(all_pos_areas)} ({100*tiny_count/len(all_pos_areas):.1f}%)")
        print(f"      Small:      {small_count}/{len(all_pos_areas)} ({100*small_count/len(all_pos_areas):.1f}%)")
        print(f"      Normal:     {normal_count}/{len(all_pos_areas)} ({100*normal_count/len(all_pos_areas):.1f}%)")
        print(f"      Large:      {large_count}/{len(all_pos_areas)} ({100*large_count/len(all_pos_areas):.1f}%)")
        print(f"      Very Large: {very_large_count}/{len(all_pos_areas)} ({100*very_large_count/len(all_pos_areas):.1f}%)")

    def _precompute_tuning_cache(self, verified_data, context_scales_grid):
        """
        Pre-computes features for a GRID of context scales.
        Includes ROBUST cropping logic to prevent PIL errors.
        """
        print(f"\n--> [Cache] Pre-computing features for scales: {context_scales_grid}")
        self.model_manager.switch_to('dino')
        
        cache = []
        
        # Calculate total for progress bar
        total_items = 0
        for item in verified_data:
            n_p = len(item['pos'])
            n_n = min(len(item['neg']), max(10, n_p * 3))
            total_items += (n_p + n_n)

        processed = 0
        
        for item in verified_data:
            if len(item['pos']) == 0 and len(item['neg']) == 0: continue
            
            image_source, _ = ImageProcessor.load_image(item['path'], self.STANDARD_IMAGE_SCALE)
            img_h, img_w, _ = image_source.shape
            
            n_pos = len(item['pos'])
            n_neg_sample = min(len(item['neg']), max(10, n_pos * 3))
            
            boxes_pos = item['pos'] if isinstance(item['pos'], np.ndarray) else item['pos'].numpy()
            boxes_neg = item['neg'][:n_neg_sample] if isinstance(item['neg'], np.ndarray) else item['neg'][:n_neg_sample].numpy()
            
            if n_pos == 0 and n_neg_sample == 0: continue
            
            arrays_to_stack = []
            if n_pos > 0:
                # Ensure 2D shape (N, 4) just in case
                if boxes_pos.ndim == 1: boxes_pos = boxes_pos.reshape(-1, 4)
                arrays_to_stack.append(boxes_pos)
                
            if n_neg_sample > 0:
                if boxes_neg.ndim == 1: boxes_neg = boxes_neg.reshape(-1, 4)
                arrays_to_stack.append(boxes_neg)
            
            if not arrays_to_stack: continue # Should be caught above, but safety first
            
            all_boxes = np.vstack(arrays_to_stack)
                
            labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg_sample)])
            
            # --- 1. Compute Static Features ---
            obj_crops, _ = self._get_dual_crops(image_source, all_boxes) 
            
            feat_obj = self.extract_dino_features(obj_crops, force_cpu=False).cpu()
            geom = self._extract_geometric_features(all_boxes, image_source.shape).cpu()
            texture = self._extract_texture_features(obj_crops).cpu()
            
            # --- 2. Determine Size Category ---
            areas_norm = all_boxes[:, 2] * all_boxes[:, 3]
            categories = []
            for area in areas_norm:
                if area < self.size_thresholds['tiny']: cat = 'tiny'
                elif area < self.size_thresholds['small']: cat = 'small'
                elif area < self.size_thresholds['normal_max']: cat = 'normal'
                elif area < self.size_thresholds['very_large']: cat = 'large'
                else: cat = 'very_large'
                categories.append(cat)
                
            # --- 3. Compute Context Features for GRID ---
            context_map_list = [{} for _ in range(len(all_boxes))]
            pil_img = Image.fromarray(image_source)
            
            for scale in context_scales_grid:
                ctx_crops_scale = []
                for i in range(len(all_boxes)):
                    cx, cy = all_boxes[i, 0]*img_w, all_boxes[i, 1]*img_h
                    w, h = all_boxes[i, 2]*img_w, all_boxes[i, 3]*img_h
                    max_dim = max(w, h)
                    
                    # Target dimension
                    ctx_dim = max(max_dim * scale, 64) 
                    
                    # --- ROBUST CROP LOGIC START ---
                    # 1. Calculate ideal coordinates centered on object
                    half_size = ctx_dim / 2
                    x1 = cx - half_size
                    x2 = cx + half_size
                    y1 = cy - half_size
                    y2 = cy + half_size
                    
                    # 2. Shift window if it falls off the edge (try to keep size constant)
                    if x1 < 0:
                        x2 += abs(x1) # Shift right
                        x1 = 0
                    if x2 > img_w:
                        x1 -= (x2 - img_w) # Shift left
                        x2 = img_w
                        
                    if y1 < 0:
                        y2 += abs(y1) # Shift down
                        y1 = 0
                    if y2 > img_h:
                        y1 -= (y2 - img_h) # Shift up
                        y2 = img_h
                        
                    # 3. Hard Clamp (Handles case where context > image size)
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(img_w, int(x2))
                    y2 = min(img_h, int(y2))
                    
                    # 4. Final Sanity Check (Prevent 0-width crops)
                    if x2 <= x1: x2 = x1 + 1
                    if y2 <= y1: y2 = y1 + 1
                    # --- ROBUST CROP LOGIC END ---
                        
                    ctx_crops_scale.append(pil_img.crop((x1, y1, x2, y2)))
                
                # Bulk extract DINO
                feats_scale = self.extract_dino_features(ctx_crops_scale, force_cpu=False).cpu()
                
                # Store
                for i in range(len(all_boxes)):
                    context_map_list[i][scale] = feats_scale[i]

            # --- 4. Store in Cache ---
            for i in range(len(all_boxes)):
                cache.append({
                    'label': labels[i],
                    'category': categories[i],
                    'static_obj': feat_obj[i],
                    'static_geom': geom[i],
                    'static_tex': texture[i],
                    'context_map': context_map_list[i]
                })
            
            processed += len(all_boxes)
            if processed % 50 == 0:
                print(f"    [Cache] {processed}/{total_items} samples processed...")
                gc.collect()

        torch.cuda.empty_cache()
        print(f"    [Cache] Done. Cached {len(cache)} samples.")
        return cache

    def fine_tune_dino(self, verified_data, save_path="verifier/dino_finetune.pt",
                    n_epochs=80, batch_size=16, grad_accum_steps=2,
                    lora_rank=8, apply_from_block=12):
        """
        LoRA fine-tuning of DINOv2.

        Blocks 0..apply_from_block-1  — fully frozen, activations cached once (same trick as before).
        Blocks apply_from_block..23   — base weights frozen, LoRA A/B adapters on qkv + proj.
        Projection head               — small MLP, fully trainable.

        Trainable params: ~786K (LoRA) + ~135K (head) ≈ 0.85M total.
        Compare to the original last-block approach: ~4M params.
        """
        print("\n--> [LoRA] Preparing data...")

        # ------------------------------------------------------------------ #
        # 1. Collect crops                                                     #
        # ------------------------------------------------------------------ #
        all_crops, all_labels = [], []
        augmentor = Augmentor()

        for item in verified_data:
            if len(item['pos']) == 0 and len(item['neg']) == 0:
                continue
            image_source, _ = ImageProcessor.load_image(item['path'], self.STANDARD_IMAGE_SCALE)
            pos_crops, _ = self._get_dual_crops(image_source, item['pos'])
            neg_crops, _ = self._get_dual_crops(image_source, item['neg'])
            all_crops.extend(pos_crops);  all_labels.extend([1] * len(pos_crops))
            all_crops.extend(neg_crops);  all_labels.extend([0] * len(neg_crops))

        pos_idx = [i for i, l in enumerate(all_labels) if l == 1][:1000]
        neg_idx = [i for i, l in enumerate(all_labels) if l == 0][:1000]
        keep       = pos_idx + neg_idx
        all_crops  = [all_crops[i]  for i in keep]
        all_labels = [all_labels[i] for i in keep]

        ft_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        base_tensors = torch.stack([ft_transform(c)          for c in all_crops])
        aug_tensors  = torch.stack([aug_tf(ft_transform(c))  for c in all_crops * 2])
        base_labels  = torch.tensor(all_labels, dtype=torch.long)

        tensors = torch.cat([base_tensors, aug_tensors])
        labels  = torch.cat([base_labels,  base_labels.repeat(2)])

        n_pos = (labels == 1).sum().item()
        n_neg = (labels == 0).sum().item()
        print(f"    [LoRA] {n_pos} pos | {n_neg} neg | {len(tensors)} total samples")

        # ------------------------------------------------------------------ #
        # 2. Inject LoRA into blocks apply_from_block..23                     #
        # ------------------------------------------------------------------ #
        dino_model = self.model_manager.models['dino']
        dino_model.to(self.device)
        self.model_manager.current_key = 'dino'

        # Freeze everything first
        for param in dino_model.parameters():
            param.requires_grad = False

        lora_layer_map = {}   # key → LoRALinear, used for saving

        for idx in range(apply_from_block, len(dino_model.blocks)):
            attn = dino_model.blocks[idx].attn

            wrapped_qkv  = LoRALinear(attn.qkv,  rank=lora_rank, alpha=lora_rank * 2).to(self.device)
            wrapped_proj = LoRALinear(attn.proj, rank=lora_rank, alpha=lora_rank * 2).to(self.device)

            attn.qkv  = wrapped_qkv
            attn.proj = wrapped_proj

            lora_layer_map[f'block{idx}_qkv']  = wrapped_qkv
            lora_layer_map[f'block{idx}_proj'] = wrapped_proj

        proj_head = torch.nn.Sequential(
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        ).to(self.device)

        # ------------------------------------------------------------------ #
        # 3. Pre-compute frozen activations (blocks 0..apply_from_block-1)    #
        #    Identical hook trick — just the cutoff block index changes.       #
        # ------------------------------------------------------------------ #
        print(f"    [LoRA] Pre-computing frozen activations (blocks 0–{apply_from_block - 1})...")

        frozen_cache = []
        dino_model.eval()

        captured = {}
        def capture_input(module, inp, out):
            captured['x'] = inp[0].detach()

        hook = dino_model.blocks[apply_from_block].register_forward_hook(capture_input)

        CACHE_BATCH = 32
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for i in range(0, len(tensors), CACHE_BATCH):
                    batch = tensors[i:i + CACHE_BATCH].to(self.device)
                    dino_model(batch)
                    frozen_cache.append(captured['x'].float().cpu())
                    del batch

        hook.remove()
        del captured
        torch.cuda.empty_cache()

        frozen_acts = torch.cat(frozen_cache, dim=0)   # (N, 197, 1024) on CPU
        del frozen_cache
        print(f"    [LoRA] Cached {frozen_acts.shape[0]} activations at block {apply_from_block}. "
              f"Shape: {tuple(frozen_acts.shape)}")

        # ------------------------------------------------------------------ #
        # 4. Trainable params: LoRA A/B matrices + proj_head                  #
        # ------------------------------------------------------------------ #
        trainable_params = list(proj_head.parameters())
        for layer in lora_layer_map.values():
            trainable_params += [layer.lora_A, layer.lora_B]

        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"    [LoRA] Trainable: {n_trainable / 1e6:.2f}M params "
              f"(rank={lora_rank}, blocks {apply_from_block}–{len(dino_model.blocks) - 1})")
        print(f"--> [LoRA] Training {n_epochs} epochs...")

        # ------------------------------------------------------------------ #
        # 5. Training loop — identical to before except the forward pass      #
        #    now runs blocks apply_from_block..23 (with LoRA) + proj_head.    #
        # ------------------------------------------------------------------ #
        optimizer = optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)
        scaler    = GradScaler()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        def supcon_loss(features, labels, temperature=0.07):
            device   = features.device
            n        = features.shape[0]
            sim      = torch.matmul(features, features.T) / temperature
            self_mask = torch.eye(n, dtype=torch.bool, device=device)
            sim.masked_fill_(self_mask, float('-inf'))
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
            if pos_mask.sum() == 0:
                return torch.tensor(0.0, requires_grad=True, device=device)
            log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
            loss = -(log_prob.masked_fill(~pos_mask, 0.0)).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
            return loss.mean()

        best_loss = float('inf')

        for epoch in tqdm(range(n_epochs)):
            # Set LoRA layers + proj_head to train mode; frozen blocks stay eval
            for layer in lora_layer_map.values():
                layer.train()
            proj_head.train()

            perm         = torch.randperm(len(frozen_acts))
            epoch_acts   = frozen_acts[perm]
            epoch_labels = labels[perm]

            epoch_loss = 0.0
            n_batches  = 0
            optimizer.zero_grad()

            for i in range(0, len(epoch_acts), batch_size):
                batch_acts   = epoch_acts[i:i + batch_size].to(self.device)
                batch_labels = epoch_labels[i:i + batch_size].to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Run only the LoRA-adapted blocks — blocks 0..apply_from_block-1
                    # were already baked into batch_acts by the cache step above.
                    x = batch_acts
                    for blk_idx in range(apply_from_block, len(dino_model.blocks)):
                        x = dino_model.blocks[blk_idx](x)

                    cls  = x[:, 0, :]              # CLS token  (B, 1024)
                    proj = proj_head(cls)          # (B, 64)
                    proj = F.normalize(proj, dim=1)
                    loss = supcon_loss(proj, batch_labels) / grad_accum_steps

                scaler.scale(loss).backward()

                if (i // batch_size + 1) % grad_accum_steps == 0 or \
                   (i + batch_size) >= len(epoch_acts):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * grad_accum_steps
                n_batches  += 1
                del batch_acts, batch_labels, x, cls, proj, loss

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1:3d}/{n_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            if avg_loss < best_loss:
                best_loss = avg_loss

                # Save only the LoRA deltas + proj_head — not the full model
                lora_state = {
                    name: {'A': layer.lora_A.data.cpu(), 'B': layer.lora_B.data.cpu()}
                    for name, layer in lora_layer_map.items()
                }
                torch.save({
                    'type':              'lora',
                    'lora_state':        lora_state,
                    'proj_head':         proj_head.state_dict(),
                    'lora_rank':         lora_rank,
                    'apply_from_block':  apply_from_block,
                    'epoch':             epoch,
                    'loss':              best_loss,
                }, save_path)

        print(f"\n--> [LoRA] Done. Best loss: {best_loss:.4f} | Saved: {save_path}")

        # Freeze everything again — LoRA weights stay injected in the live model
        for param in dino_model.parameters():
            param.requires_grad = False

        del frozen_acts
        gc.collect()
        torch.cuda.empty_cache()

        return proj_head

    def load_finetuned_dino(self, save_path="verifier/dino_finetune.pt"):
        if not os.path.exists(save_path):
            print(f"--> [LoRA] No fine-tuned weights found at {save_path}")
            return False

        print(f"--> [LoRA] Loading fine-tuned DINOv2 weights from {save_path}...")
        checkpoint  = torch.load(save_path, map_location='cpu')
        dino_model  = self.model_manager.models['dino']

        # ---- New LoRA format ----
        if checkpoint.get('type') == 'lora':
            lora_rank        = checkpoint['lora_rank']
            apply_from_block = checkpoint['apply_from_block']
            lora_state       = checkpoint['lora_state']

            # Freeze base weights, then inject LoRALinear wrappers and load saved A/B
            for param in dino_model.parameters():
                param.requires_grad = False

            for name, ab in lora_state.items():
                # name format: 'block{idx}_qkv' or 'block{idx}_proj'
                parts    = name.split('_')         # ['block12', 'qkv']
                blk_idx  = int(parts[0].replace('block', ''))
                layer_id = parts[1]                # 'qkv' or 'proj'

                attn     = dino_model.blocks[blk_idx].attn
                original = getattr(attn, layer_id)  # the raw nn.Linear

                # Only wrap if not already wrapped (idempotent on repeated loads)
                if not isinstance(original, LoRALinear):
                    wrapped = LoRALinear(original, rank=lora_rank, alpha=lora_rank * 2)
                    setattr(attn, layer_id, wrapped)
                else:
                    wrapped = original

                wrapped.lora_A.data = ab['A']
                wrapped.lora_B.data = ab['B']

            print(f"    [LoRA] Restored LoRA adapters | "
                  f"rank={lora_rank} | blocks {apply_from_block}–{len(dino_model.blocks) - 1} | "
                  f"epoch {checkpoint['epoch']} | loss {checkpoint['loss']:.4f}")

        # ---- Legacy format (single-block full fine-tune) ----
        else:
            block_idx = checkpoint['block_idx']
            dino_model.blocks[block_idx].load_state_dict(checkpoint['block_state'])
            print(f"    [Legacy] Restored block {block_idx} | "
                  f"epoch {checkpoint['epoch']} | loss {checkpoint['loss']:.4f}")

        return True

    def tune_context_factors_optuna(self, verified_data, n_trials=50):
        """
        Optimizes context factors using cached features.
        Metric: Maximize separation = mean(pos_conf) - mean(neg_conf) - variance_penalty
        """
        # 1. Define the grid of scales to test (Must match what is in _precompute_tuning_cache)
        search_grid = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
        
        # 2. Build Cache (One-time heavy computation)
        cache = self._precompute_tuning_cache(verified_data, search_grid)
        if not cache:
            print("[Error] No cache generated.")
            return None

        # Pre-extract labels since they don't change
        y_true = np.array([item['label'] for item in cache])
        
        print(f"\n--> [Tuning] Running {n_trials} trials...")
        
        def objective(trial):
            # --- A. Suggest Parameters ---
            s_tiny = trial.suggest_categorical('context_tiny', search_grid)
            s_small = trial.suggest_categorical('context_small', search_grid)
            s_normal = trial.suggest_categorical('context_normal', search_grid)
            s_large = trial.suggest_categorical('context_large', search_grid)
            
            scale_lookup = {
                'tiny': s_tiny,
                'small': s_small,
                'normal': s_normal,
                'large': s_large,
                'very_large': s_large 
            }
            
            # --- B. Construct Feature Matrix X from Cache ---
            X_list = []
            for item in cache:
                cat = item['category']
                chosen_scale = scale_lookup[cat]
                
                # Get components
                f_obj = item['static_obj']
                f_geom = item['static_geom']
                f_tex = item['static_tex']
                f_ctx = item['context_map'].get(chosen_scale, list(item['context_map'].values())[0])

                # Concatenate [Obj, Ctx, Geom, Tex]
                feat = torch.cat([f_obj, f_ctx, f_geom, f_tex], dim=0)
                X_list.append(feat.numpy())
            
            X_train = np.stack(X_list)
            
            # --- C. Train & Predict ---
            # We train a fresh verifier on this specific feature configuration
            clf = EnsembleVerifier()
            clf.fit(X_train, y_true)
            
            # Get probabilities for the same data
            # (Since we care about "separability" of the features, training error is acceptable here)
            probs = clf.predict_proba(X_train)[:, 1]
            
            # --- D. Calculate Your Custom Metric ---
            pos_confidences = probs[y_true == 1]
            neg_confidences = probs[y_true == 0]
            
            if len(pos_confidences) == 0 or len(neg_confidences) == 0:
                return -1.0

            mean_pos = np.mean(pos_confidences)
            mean_neg = np.mean(neg_confidences)
            std_pos = np.std(pos_confidences)
            std_neg = np.std(neg_confidences)
            
            # Primary metric: separation
            separation = mean_pos - mean_neg
            
            # Secondary metric: tightness (penalize high variance)
            variance_penalty = 0.05 * (std_pos + std_neg)
            
            score = separation - variance_penalty
            
            # Log for analysis
            trial.set_user_attr('mean_pos', mean_pos)
            trial.set_user_attr('mean_neg', mean_neg)
            trial.set_user_attr('separation', separation)
            
            return score

        # Run Study
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Apply Best Parameters
        best = study.best_params
        self.context_factors['tiny'] = best['context_tiny']
        self.context_factors['small'] = best['context_small']
        self.context_factors['normal'] = best['context_normal']
        self.context_factors['large'] = best['context_large']
        self.context_factors['very_large'] = best['context_large'] * 0.8
        
        print(f"\n    [Optimized Context Factors] (Score: {study.best_value:.4f})")
        print(f"      Tiny:   {self.context_factors['tiny']}x")
        print(f"      Small:  {self.context_factors['small']}x")
        print(f"      Normal: {self.context_factors['normal']}x")
        print(f"      Large:  {self.context_factors['large']}x")
        
        return study
    
    def _resize_to_stride_cv2(self, image_arr, target_size=None):
        """ 
        OpenCV version of resize_to_stride. 
        Input: numpy array (H, W, 3)
        """
        h, w, _ = image_arr.shape
        if target_size is not None:
            scale = target_size / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = w, h
        
        # Ensure stride of 32
        new_w = max(32, (new_w // 32) * 32)
        new_h = max(32, (new_h // 32) * 32)
        
        return cv2.resize(image_arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def _filter_small_area(self, boxes, logits, min_area=150):
        """
        Removes detections that are physically too small to be meaningful objects.
        min_area: Minimum area in pixels (w * h)
        """
        if len(boxes) == 0: return boxes, logits
        
        # boxes is (N, 4) in format x1, y1, x2, y2
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        areas = ws * hs
        
        keep = areas >= min_area
        
        n_dropped = len(boxes) - keep.sum()
        if n_dropped > 0:
            print(f"    [Noise] Dropped {n_dropped} tiny artifacts (< {min_area}px area)")
            
        return boxes[keep], logits[keep]

    def _filter_contained_boxes(self, boxes_xyxy, logits, threshold=0.85, confidence_margin=0.10):
        """
        Removes boxes that are inside other boxes, unless they are SIGNIFICANTLY more confident.
        threshold: Intersection/Area ratio to consider 'contained' (0.85 = 85% overlap)
        confidence_margin: How much better the small box must be to kill the large box.
        """
        if len(boxes_xyxy) == 0: return boxes_xyxy, logits
        n = len(boxes_xyxy)
        keep = np.ones(n, dtype=bool)
        
        # Calculate area of every box
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

        for i in range(n):
            if not keep[i]: continue
            for j in range(n):
                if i == j or not keep[j]: continue
                
                # Calculate Intersection
                xx1 = max(boxes_xyxy[i, 0], boxes_xyxy[j, 0])
                yy1 = max(boxes_xyxy[i, 1], boxes_xyxy[j, 1])
                xx2 = min(boxes_xyxy[i, 2], boxes_xyxy[j, 2])
                yy2 = min(boxes_xyxy[i, 3], boxes_xyxy[j, 3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter_area = w * h
                
                # Check if box I is inside box J (I is the "contained/small" one)
                if inter_area / (areas[i] + 1e-6) > threshold:
                    
                    if logits[i] > (logits[j] + confidence_margin):
                        keep[j] = False 
                    else:
                        keep[i] = False
                        break

        return boxes_xyxy[keep], logits[keep]
    
    def extract_dino_features(self, images, force_cpu=False):
        """
        Extracts DINOv2 features using Mixed Precision (AMP) for speed.
        """ 
        if not images: 
            return torch.empty(0)

        chunk_size = 32
        embeddings_list = []

        if force_cpu:
            dino_model = self.model_manager.models['dino']
            dino_model.to(self.cpu)
            self.model_manager.invalidate('dino')
            device = self.cpu
            context = torch.no_grad() # No AMP on CPU usually
        else:
            self.model_manager.switch_to('dino')
            dino_model = self.model_manager.models['dino']
            device = self.device
            # Enable Mixed Precision for GPU
            context = torch.autocast(device_type=self.device, dtype=torch.float16)

        with torch.no_grad():
            with context:
                for i in range(0, len(images), chunk_size):
                    batch_pil = images[i:i+chunk_size]
                    
                    # Standardize: PIL -> Tensor (ImageNet Norm) -> Device
                    batch_tensors = torch.stack([self.dino_transform(img) for img in batch_pil]).to(device)

                    emb = dino_model(batch_tensors)
                    embeddings_list.append(emb.detach().float().cpu()) 
                    del batch_tensors, emb

        if not embeddings_list: 
            return torch.empty(0)
            
        visual_feats = torch.cat(embeddings_list)
        visual_feats = F.normalize(visual_feats, dim=1, p=2)

        return visual_feats
    
    def extract_clip_features(self, images, force_cpu=False):
        if not images: 
            return torch.empty(0)

        embeddings_list = []
        chunk_size = 32
        
        if force_cpu:
            self.model_manager.models['clip'].to(self.cpu)
            self.model_manager.invalidate('clip')
            device = self.cpu
            context = torch.no_grad()
        else:
            self.model_manager.switch_to('clip')
            device = self.device
            context = torch.autocast(device_type=self.device, dtype=torch.float16)
            
        model = self.model_manager.models['clip']
        
        with torch.no_grad():
            with context:
                for i in range(0, len(images), chunk_size):
                    batch_pil = images[i:i+chunk_size]
                    batch_tensors = torch.stack([self.clip_transform(img) for img in batch_pil]).to(device)
                    
                    emb = model.get_image_features(pixel_values=batch_tensors)
                    embeddings_list.append(emb.detach().float().cpu())
                    del batch_tensors, emb

        if not embeddings_list: 
            return torch.empty(0)
            
        visual_feats = torch.cat(embeddings_list)
        visual_feats = F.normalize(visual_feats, dim=1, p=2)

        return visual_feats
    def extract_roi_patch_features(self, image_source, boxes, force_cpu=False):
        """
        Runs ONE DINOv2 forward pass on the full image, then does ROI pooling per box.
        Produces object, context, and contrast features with no context scale parameter.
        
        Multi-layer:
          - blocks[5]  → early (texture, low-level patterns)
          - blocks[11] → mid   (structural)
          - final      → late  (semantic)
        
        Per-box output:
          obj_early  (1024) — what the object looks like texturally
          obj_late   (1024) — what the object is semantically
          ctx_late   (1024) — what surrounds it
          contrast   (1024) — obj_late - ctx_late  (the key discriminator)
          mid_global (1024) — full-scene structural context
        
        Total: (N, 5120). No crop, no scale factor, no magic number.
        Cached: repeated calls with the same image_source hit the cache.
        """
        if len(boxes) == 0:
            return torch.empty(0)

        boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else np.array(boxes)

        # --- Device setup (mirrors existing extract_dino_features pattern) ---
        if force_cpu:
            dino_model = self.model_manager.models['dino']
            dino_model.to(self.cpu)
            self.model_manager.invalidate('dino')
            device = self.cpu
        else:
            self.model_manager.switch_to('dino')
            dino_model = self.model_manager.models['dino']
            device = self.device

        # --- Single forward pass, cached by image identity ---
        img_id = hash(image_source.tobytes())
        cache_valid = (
            self._dino_cache.get('img_id') == img_id and
            self._dino_cache.get('device') == device
        )

        if not cache_valid:
            pil_img = Image.fromarray(image_source)
            img_tensor = self.dino_transform(pil_img).unsqueeze(0).to(device)
            # dino_transform resizes to (336, 336) → patch grid is 336/14 = 24x24

            intermediate = {}

            def make_hook(key):
                def hook(module, inp, out):
                    # out: (B, num_patches+1, dim) — index 0 is CLS, rest are patches
                    intermediate[key] = out[:, 1:, :].detach()
                return hook

            h_early = dino_model.blocks[5].register_forward_hook(make_hook('early'))
            h_mid   = dino_model.blocks[11].register_forward_hook(make_hook('mid'))

            try:
                with torch.no_grad():
                    if device == self.device:
                        with torch.autocast(device_type=self.device, dtype=torch.float16):
                            out = dino_model.forward_features(img_tensor)
                    else:
                        out = dino_model.forward_features(img_tensor)
            finally:
                h_early.remove()
                h_mid.remove()
            GRID = 336 // 14   # = 24
            DIM  = 1024        # ViT-L hidden dim

            # Move everything to CPU immediately to free VRAM
            self._dino_cache = {
                'img_id': img_id,
                'device': device,
                'early': intermediate['early'].float().cpu()[0].reshape(GRID, GRID, DIM),  # (24,24,1024)
                'mid':   intermediate['mid'].float().cpu()[0].reshape(GRID, GRID, DIM),
                'late':  out['x_norm_patchtokens'].float().cpu()[0].reshape(GRID, GRID, DIM),
                'grid':  GRID,
                'dim':   DIM,
            }

            del img_tensor, intermediate, out

        grid = self._dino_cache['grid']
        dim  = self._dino_cache['dim']
        feat_early = self._dino_cache['early']   # (24, 24, 1024) on CPU
        feat_mid   = self._dino_cache['mid']
        feat_late  = self._dino_cache['late']

        # Precompute mid global once (same for all boxes in this image)
        mid_global = F.normalize(feat_mid.reshape(-1, dim).mean(0, keepdim=True), dim=1).squeeze(0)

        # --- Per-box ROI pooling ---
        all_features = []

        for box in boxes_np:
            cx, cy, bw, bh = box

            # Map normalized cxcywh → patch grid integer bounds
            x1 = max(0,    int((cx - bw / 2) * grid))
            y1 = max(0,    int((cy - bh / 2) * grid))
            x2 = min(grid, int((cx + bw / 2) * grid) + 1)
            y2 = min(grid, int((cy + bh / 2) * grid) + 1)

            # Guarantee at least one patch
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)

            # Inner ROI (object patches)
            obj_early = feat_early[y1:y2, x1:x2].reshape(-1, dim).mean(0)
            obj_late  = feat_late[y1:y2, x1:x2].reshape(-1, dim).mean(0)

            # Outer ROI (all patches outside the box = true context, no scale param)
            inner_mask             = torch.zeros(grid, grid, dtype=torch.bool)
            inner_mask[y1:y2, x1:x2] = True
            ctx_late = feat_late[~inner_mask].mean(0)

            # Contrast: what makes this box different from its surroundings
            contrast = obj_late - ctx_late

            # Normalize each component independently before concatenating
            def n(t): return F.normalize(t.unsqueeze(0), dim=1, p=2).squeeze(0)

            feat = torch.cat([
                n(obj_early),   # (1024) — texture signal
                n(obj_late),    # (1024) — semantic object identity
                n(ctx_late),    # (1024) — semantic surroundings
                n(contrast),    # (1024) — object vs background gap
                mid_global,     # (1024) — already normalized above
            ])   # → (5120,)

            all_features.append(feat)

        return torch.stack(all_features).float()   # (N, 5120)

    def _extract_geometric_features(self, boxes, img_shape):
        """
        Geometric features - Pre-normalized to ~0-1 range.
        This makes them comparable in scale to DINO features (unit vectors).
        """
        h_img, w_img = img_shape[:2]
        
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes).float()
        
        w_box = boxes[:, 2]
        h_box = boxes[:, 3]
        area = w_box * h_box
        
        # Aspect ratio - log scale to handle wide range (0.1 to 10)
        aspect = w_box / (h_box + 1e-6)
        aspect_log = torch.log(aspect + 1e-6)  # -2.3 to 2.3
        aspect_norm = (aspect_log + 3.0) / 6.0  # Normalize to ~0-1
        
        # Position (already 0-1)
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        
        # Distance from center (0 to ~0.7)
        dist_from_center = torch.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
        dist_norm = dist_from_center / 0.707  # Normalize to 0-1 (max dist = sqrt(0.5))
        
        # Scale features
        relative_size = torch.sqrt(area)  # Already 0-1
        
        # Log area: typical range -9 (0.01% of image) to 0 (100% of image)
        log_area = torch.log(area + 1e-6)
        log_area_norm = (log_area + 10.0) / 10.0  # Normalize to 0-1
        
        # Compactness (0-1, where 1 = circle)
        perimeter = 2 * (w_box + h_box)
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        
        # All in 0-1 range now
        return torch.stack([
            w_box,           # 0-1
            h_box,           # 0-1
            area,            # 0-1
            aspect_norm,     # 0-1
            cx,              # 0-1
            cy,              # 0-1
            dist_norm,       # 0-1
            relative_size,   # 0-1
            log_area_norm,   # 0-1
            compactness      # 0-1
        ], dim=1)

    def _extract_texture_features(self, crops):
        """
        Texture features - already in 0-1 range.
        """
        if not crops:
            return torch.empty(0, 2)
        
        features = []
        for crop in crops:
            img = np.array(crop.convert('L'))
            
            # Edge density (0-1)
            edges = cv2.Canny(img, 50, 150)
            edge_density = edges.sum() / (img.shape[0] * img.shape[1] + 1e-6)
            
            # Variance (0-1)
            variance = np.var(img) / 255.0
            
            features.append([edge_density, variance])
        
        return torch.tensor(features, dtype=torch.float32)

    def _extract_features(self, image_source, boxes, obj_crops, ctx_crops, force_cpu=False):
        """
        Extract rich feature set combining visual + geometric + texture.
        
        Args:
            image_source: The original image (numpy array)
            boxes: Normalized boxes (N, 4) in cxcywh format
            obj_crops: List of PIL Image crops (object-focused)
            ctx_crops: List of PIL Image crops (context-focused)
            force_cpu: Whether to force CPU for feature extraction

        !! NOTE:
            Visual features now come from ROI patch pooling (single DINOv2 pass).
            obj_crops kept for texture only. ctx_crops no longer used.
            Signature unchanged — all callers work without modification.
        
        Returns:
            Combined feature tensor (N, feature_dim)
        """
        roi_feats = self.extract_roi_patch_features(image_source, boxes, force_cpu=force_cpu) # (N, 5120)
        geom      = self._extract_geometric_features(boxes, image_source.shape).float() # (N, 10)
        texture   = self._extract_texture_features(obj_crops).float() # (N, 2)

        return torch.cat([roi_feats, geom, texture], dim=1)   # (N, 5132)
    # def _extract_context_features(self, images, force_cpu=False):
    #     """
    #     Returns [DINO_Object (1024) | CLIP_Context (512)]
    #     """
    #     if not images: return torch.empty(0)
        
    #     feat_dino = self.extract_dino_features(images, force_cpu=force_cpu)
    #     # feat_clip = self.extract_clip_features(images, force_cpu=force_cpu)

    #     # return torch.cat([feat_dino, feat_clip], dim=1)
    #     return feat_dino

    def _get_dual_crops(self, image_source, boxes, min_crop_size=64):
        """
        Adaptive dual-crop strategy using calibrated size thresholds.
        Context size is determined by the object's size category.
        
        Strategy:
        - Object crop: Tight bounding box with small padding
        - Context crop: Size based on calibrated thresholds (no magic numbers)
        """
        if len(boxes) == 0: return [], []
        
        img_h, img_w, _ = image_source.shape
        pil_img = Image.fromarray(image_source)
        
        # Handle Tensor vs Numpy
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes
        
        # Standardize Coordinates (Absolute xywh)
        if boxes_np.max() <= 1.0:
            cx = boxes_np[:, 0] * img_w
            cy = boxes_np[:, 1] * img_h
            w = boxes_np[:, 2] * img_w
            h = boxes_np[:, 3] * img_h
        else:
            cx = boxes_np[:, 0]
            cy = boxes_np[:, 1]
            w = boxes_np[:, 2]
            h = boxes_np[:, 3]
        
        areas_norm = (w / img_w) * (h / img_h)
        
        context_factors = np.zeros(len(boxes_np))
        
        tiny_mask = areas_norm < self.size_thresholds['tiny']
        small_mask = (areas_norm >= self.size_thresholds['tiny']) & (areas_norm < self.size_thresholds['small'])
        normal_mask = (areas_norm >= self.size_thresholds['small']) & (areas_norm < self.size_thresholds['normal_max'])
        large_mask = (areas_norm >= self.size_thresholds['normal_max']) & (areas_norm < self.size_thresholds['very_large'])
        very_large_mask = areas_norm >= self.size_thresholds['very_large']
        
        context_factors[tiny_mask] = self.context_factors['tiny']
        context_factors[small_mask] = self.context_factors['small']
        context_factors[normal_mask] = self.context_factors['normal']
        context_factors[large_mask] = self.context_factors['large']
        context_factors[very_large_mask] = self.context_factors['very_large']
        
        obj_crops = []
        ctx_crops_pil = []
        
        for i in range(len(w)):
            obj_padding = 1.1
            o_dim_w = np.maximum(w[i] * obj_padding, min_crop_size)
            o_dim_h = np.maximum(h[i] * obj_padding, min_crop_size)
            
            o_x1 = int(max(0, cx[i] - o_dim_w/2))
            o_y1 = int(max(0, cy[i] - o_dim_h/2))
            o_x2 = int(min(img_w, cx[i] + o_dim_w/2))
            o_y2 = int(min(img_h, cy[i] + o_dim_h/2))
            
            obj_crops.append(pil_img.crop((o_x1, o_y1, o_x2, o_y2)))
            
            max_dim = max(w[i], h[i])
            
            context_factor = context_factors[i]
            
            ctx_dim = max_dim * context_factor
            ctx_dim = max(ctx_dim, 2 * min_crop_size)
            
            c_x1 = int(max(0, cx[i] - ctx_dim/2))
            c_y1 = int(max(0, cy[i] - ctx_dim/2))
            c_x2 = int(min(img_w, cx[i] + ctx_dim/2))
            c_y2 = int(min(img_h, cy[i] + ctx_dim/2))
            
            # Handle edge cases
            actual_w = c_x2 - c_x1
            actual_h = c_y2 - c_y1
            
            if actual_w < ctx_dim and c_x1 == 0:
                c_x2 = int(min(img_w, ctx_dim))
            elif actual_w < ctx_dim and c_x2 == img_w:
                c_x1 = int(max(0, img_w - ctx_dim))
            
            if actual_h < ctx_dim and c_y1 == 0:
                c_y2 = int(min(img_h, ctx_dim))
            elif actual_h < ctx_dim and c_y2 == img_h:
                c_y1 = int(max(0, img_h - ctx_dim))
            
            ctx_crops_pil.append(pil_img.crop((c_x1, c_y1, c_x2, c_y2)))
        
        return obj_crops, ctx_crops_pil
    
    def _save_verifier(self, model, file_path="weights/verifier.pkl"):
        """Saves the trained SVM model to disk."""
        if file_path is None: return
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if 'size_thresholds' not in model:
            model['size_thresholds'] = self.size_thresholds
        if 'threshold_adjustments' not in model:
            model['threshold_adjustments'] = self.threshold_adjustments
        if 'context_factors' not in model:
            model['context_factors'] = self.context_factors
        
        joblib.dump(model, file_path)
        print(f"--> [System] Verifier model + size thresholds saved to {file_path}")

    def load_verifier(self, file_path="weights/verifier.pkl"):
        """Loads a pre-trained SVM model from disk."""
        if not os.path.exists(file_path):
            print(f"--> [System] No model found at {file_path}")
            return None
        
        print(f"--> [System] Loading verifier model from {file_path}...")
        package = joblib.load(file_path)
        
        if 'size_thresholds' in package:
            self.size_thresholds = package['size_thresholds']
            print("    [System] Restored calibrated size thresholds from model.")
        
        # if 'threshold_adjustments' in package:
        #     self.threshold_adjustments = package['threshold_adjustments']
        #     print("    [System] Restored calibrated threshold adjustements from model.")

        if 'context_factors' in package:
            self.context_factors = package['context_factors']
            print("    [System] Restored context factors thresholds from model.")
        
        return package
    
    def _process_verifier_batch(self, image_source, boxes_norm, label_val, augmentor, X_feats_list, y_labels_list, meta_list, image_path, force_cpu, n_augment, use_augmentation=False):
        """
        Helper method to process a single batch.
        """
        if len(boxes_norm) == 0: return 0
        
        if isinstance(boxes_norm, np.ndarray):
            boxes_t = torch.from_numpy(boxes_norm).float()
        else:
            boxes_t = boxes_norm.float().detach().cpu()

        h, w, _ = image_source.shape
        chunk_size = 32
        total_processed = 0

        for i in range(0, len(boxes_t), chunk_size):
            batch_boxes = boxes_t[i:i+chunk_size]
            
            # 1. Get Originals (Base)
            obj_crops, ctx_crops = self._get_dual_crops(image_source, batch_boxes)
            if len(obj_crops) == 0: continue
            
            # 2. Create SEPARATE lists for accumulation
            final_obj = list(obj_crops)
            final_ctx = list(ctx_crops)
            
            multiplier = 1 
            
            if use_augmentation and n_augment > 0:
                with torch.no_grad():
                    for _ in range(n_augment):
                        obj_aug = augmentor.augment(obj_crops)
                        ctx_aug = augmentor.augment(ctx_crops)
                        
                        final_obj.extend(obj_aug)
                        final_ctx.extend(ctx_aug)
                
                multiplier = 1 + n_augment

            # ===== CHANGED: Use enhanced feature extraction =====
            # We need to repeat the boxes for augmented versions
            repeated_boxes = batch_boxes.repeat(multiplier, 1)
            
            feats = self._extract_features(
                image_source, 
                repeated_boxes,
                final_obj, 
                final_ctx, 
                force_cpu=force_cpu
            )

            X_feats_list.append(feats.detach().cpu())
            y_labels_list.append(torch.full((len(feats),), label_val))
            
            total_processed += len(feats)

            b = batch_boxes.numpy()
            x1 = (b[:, 0] - b[:, 2]/2) * w
            y1 = (b[:, 1] - b[:, 3]/2) * h
            x2 = (b[:, 0] + b[:, 2]/2) * w
            y2 = (b[:, 1] + b[:, 3]/2) * h
            abs_boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)
            
            batch_meta = []
            for box in abs_boxes:
                batch_meta.append((image_path, box))
            
            for _ in range(multiplier):
                meta_list.extend(batch_meta)

            del final_obj, final_ctx, feats, obj_crops, ctx_crops

        return total_processed

    def train_verifier(self, verified_data, save_path="weights/verifier.pkl", fast_train=False):
        """
        Phase 2 of Calibration: Extracts CONTEXT-AWARE features, trains SVM.
        Refactored to enforce CPU-heavy preprocessing and strict VRAM cleanup.
        """
        print(f"\n--> [Training] Processing {len(verified_data)} verified images...")
        self.model_manager.switch_to('dino')
        augmentor = Augmentor()
        
        X_feats = []
        y_labels = []
        train_metadata = []
        
        total_pos = 0
        total_neg = 0
        
        for i, item in enumerate(verified_data):
            if len(item['pos']) == 0 and len(item['neg']) == 0: continue
            
            image_source, _ = ImageProcessor.load_image(item['path'], self.STANDARD_IMAGE_SCALE)
            n_augment_pos = 1 if fast_train else 5
            total_pos += self._process_verifier_batch(
                image_source, item['pos'], 1, 
                augmentor, X_feats, y_labels, train_metadata, item['path'],
                force_cpu=False,
                use_augmentation=not fast_train,
                n_augment = n_augment_pos
            )

            n_augment_neg = 1 if fast_train else 2
            total_neg += self._process_verifier_batch(
                image_source, item['neg'], 0, 
                augmentor, X_feats, y_labels, train_metadata, item['path'],
                force_cpu=False,
                use_augmentation=not fast_train,
                n_augment = n_augment_neg
            )

            print(f"    [Embedder] {i+1}/{len(verified_data)} processed.")

            del image_source 
            torch.cuda.empty_cache()
            gc.collect()

        if total_pos == 0 or total_neg == 0:
            print(f"[Error] Training failed. Insufficient data.")
            return None, None, None, None

        print(f"--> [Training] Fitting SVM on {total_pos} Positives vs {total_neg} Negatives...")
        
        if len(X_feats) > 0:
            X = torch.cat(X_feats).numpy()
            y = torch.cat(y_labels).numpy()
            
            clf = EnsembleVerifier()
            clf.fit(X, y)
            
            package = {
                'model': clf, 
                'X': X, 
                'y': y,
                'meta': train_metadata
            }
            self._save_verifier(package, save_path)
            
            del X_feats, y_labels
            gc.collect()
            
            return clf, X, y, train_metadata
        else:
            return None, None, None, None

    def detect_objects(self, image_path: str, prompt: str, box_threshold=0.35, text_threshold=0.25, batch_size=8):
        """
        Uses ModelManager to keep GD loaded across calls unless forced to switch.
        """
        if not prompt.endswith("."): prompt += "."

        print(f"--> [Detect] Image: {image_path} Prompt: '{prompt}' (Batch Size: {batch_size})") 
        
        self.model_manager.switch_to('gd')
        gd_model = self.model_manager.models['gd']

        image_source, pil_image = ImageProcessor.load_image(image_path, self.STANDARD_IMAGE_SCALE)
        img_h, img_w, _ = image_source.shape
        
        SCALES = [320, None] 
        SLICE_MIN_SIZE = 512
        OVERLAP_RATIO = 0.5
        
        all_boxes_list = []
        all_logits_list = []
        transform = self._get_gd_transform() 

        scale_batches = [] 
        for scale in SCALES:
            if scale is None or (img_w <= scale and img_h <= scale):
                crops = [(0, 0, img_w, img_h)]
            else:
                stride = int(scale * (1 - OVERLAP_RATIO))
                x_steps = sorted(list(set(list(range(0, img_w - scale, stride)) + [img_w - scale])))
                y_steps = sorted(list(set(list(range(0, img_h - scale, stride)) + [img_h - scale])))
                crops = [(x, y, scale, scale) for y in y_steps for x in x_steps]

            target_sz = max(scale, SLICE_MIN_SIZE) if scale else 800
            
            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i:i+batch_size]
                tensors = []
                meta = []
                for (sx, sy, sw, sh) in batch_crops:
                    slice_img = image_source[sy:sy+sh, sx:sx+sw]
                    resized_slice = self._resize_to_stride_cv2(slice_img, target_size=target_sz)
                    slice_tensor, _ = transform(Image.fromarray(resized_slice), None)
                    tensors.append(slice_tensor)
                    h_slice, w_slice, _ = slice_img.shape
                    meta.append((sx, sy, sw, sh, w_slice, h_slice))
                
                if tensors:
                    scale_batches.append((torch.stack(tensors), meta))

        for batch_input, batch_meta in scale_batches:
            batch_input = batch_input.to(self.device)
            batch_prompts = [prompt] * len(batch_input)

            # AMP for speed on 4GB VRAM
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    outputs = gd_model(batch_input, captions=batch_prompts)
            
            pred_logits = outputs["pred_logits"].sigmoid().cpu()
            pred_boxes = outputs["pred_boxes"].cpu()
            
            for b_idx in range(len(batch_meta)):
                sx, sy, sw, sh, w_slice_orig, h_slice_orig = batch_meta[b_idx]
                logits_raw = pred_logits[b_idx]
                boxes = pred_boxes[b_idx]

                if logits_raw.shape[-1] > 1: scores, _ = logits_raw.max(dim=1)
                else: scores = logits_raw.view(-1)
                
                mask = scores > box_threshold
                boxes = boxes[mask]
                valid_scores = scores[mask]

                if len(boxes) == 0: continue

                cx, cy = boxes[:, 0] * w_slice_orig, boxes[:, 1] * h_slice_orig
                w, h = boxes[:, 2] * w_slice_orig, boxes[:, 3] * h_slice_orig
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2

                margin = 2
                is_l, is_t = (sx <= margin), (sy <= margin)
                is_r, is_b = (sx + sw >= img_w-margin), (sy + sh >= img_h-margin)

                discard = (
                    ((x1 < margin) & (not is_l)) |
                    ((y1 < margin) & (not is_t)) |
                    ((x2 > w_slice_orig - margin) & (not is_r)) |
                    ((y2 > h_slice_orig - margin) & (not is_b))
                )
                
                keep = ~discard
                if not keep.any(): continue

                res_boxes = torch.stack([x1[keep] + sx, y1[keep] + sy, x2[keep] + sx, y2[keep] + sy], dim=1)
                all_boxes_list.append(res_boxes)
                all_logits_list.append(valid_scores[keep])

        if not all_boxes_list:
            return torch.empty((0, 4)), torch.empty(0), [], image_source

        # Global NMS
        global_boxes = torch.cat(all_boxes_list)
        global_logits = torch.cat(all_logits_list)

        g_boxes_dev = global_boxes.to(self.device)
        g_scores_dev = global_logits.to(self.device)
        keep_idxs = torchvision.ops.nms(g_boxes_dev, g_scores_dev, iou_threshold=0.5)
        
        global_boxes = global_boxes[keep_idxs.cpu()]
        global_logits = global_logits[keep_idxs.cpu()]
        del g_boxes_dev, g_scores_dev # cleanup

        if len(global_boxes) > 0:
            final_boxes_np = global_boxes.numpy()
            final_logits_np = global_logits.numpy()
            
            final_boxes_np, final_logits_np = self._filter_contained_boxes(final_boxes_np, final_logits_np, threshold=0.95, confidence_margin=0.1)
            final_boxes_np, final_logits_np = self._filter_small_area(final_boxes_np, final_logits_np, min_area=150) # // 0.02 % area of the image

            # Convert to Norm CXCYWH for return
            if len(final_boxes_np) > 0:
                fw = final_boxes_np[:, 2] - final_boxes_np[:, 0]
                fh = final_boxes_np[:, 3] - final_boxes_np[:, 1]
                fcx = final_boxes_np[:, 0] + fw/2
                fcy = final_boxes_np[:, 1] + fh/2
                final_boxes = torch.tensor(np.stack([fcx/img_w, fcy/img_h, fw/img_w, fh/img_h], axis=1))
                final_logits = torch.tensor(final_logits_np)
            else:
                final_boxes = torch.empty((0, 4))
                final_logits = torch.empty(0)
        else:
            final_boxes = torch.empty((0, 4))
            final_logits = torch.empty(0)

        return final_boxes, final_logits, ["object"] * len(final_boxes), image_source
    
    def filter_candidates(self, image_source, boxes, classifier, confidence_threshold=0.5, force_cpu=False, use_tta=True):
        if len(boxes) == 0 or classifier is None: return boxes, None
        
        if isinstance(boxes, np.ndarray):
            boxes_t = torch.from_numpy(boxes).float()
        else:
            boxes_t = boxes.clone()
            
        n_candidates = len(boxes_t)
        final_probs = np.zeros(n_candidates, dtype=np.float32)
        
        BATCH_SIZE = 64
        augmentor = Augmentor()
        
        print(f"--> [Filter] Processing {n_candidates} candidates in batches of {BATCH_SIZE}...")

        for i in range(0, n_candidates, BATCH_SIZE):
            batch_boxes = boxes_t[i : i + BATCH_SIZE]
            
            obj_crops, _ = self._get_dual_crops(image_source, batch_boxes)
            if not obj_crops: continue
            
            base_feats = self._extract_features(image_source, batch_boxes, obj_crops, [], force_cpu=force_cpu).cpu().numpy()
            batch_probs = classifier.predict_proba(base_feats)[:, 1]

            # if use_tta:
            #     flip_tf = augmentor.transforms.transforms[1]
            #     jit_tf  = augmentor.transforms.transforms[0]
            #
            #     obj_flip  = [flip_tf(img) for img in obj_crops]
            #     feats_flip = self._extract_features(image_source, batch_boxes, obj_flip, [], force_cpu=force_cpu).cpu().numpy()
            #     probs_flip = classifier.predict_proba(feats_flip)[:, 1]
            #
            #     obj_jit  = [jit_tf(img) for img in obj_crops]
            #     feats_jit = self._extract_features(image_source, batch_boxes, obj_jit, [], force_cpu=force_cpu).cpu().numpy()
            #     probs_jit = classifier.predict_proba(feats_jit)[:, 1]
            #
            #     batch_probs = (batch_probs * 0.50) + (probs_flip * 0.25) + (probs_jit * 0.25)
            #
            #     del obj_flip, feats_flip, obj_jit, feats_jit
            
            final_probs[i : i + BATCH_SIZE] = batch_probs
            del obj_crops, base_feats

            if i % (BATCH_SIZE * 5) == 0:
                gc.collect()

        # Single clean threshold — trust the model
        keep_mask    = final_probs > confidence_threshold
        keep_boxes   = boxes_t[keep_mask]
        keep_scores  = torch.tensor(final_probs[keep_mask])
        reject_boxes = boxes_t[~keep_mask]
        reject_scores = torch.tensor(final_probs[~keep_mask])

        print(f"    [Filter] Rejected {(~keep_mask).sum()} / {n_candidates} candidates.")

        gc.collect()
        torch.cuda.empty_cache()

        return keep_boxes, keep_scores, reject_boxes, reject_scores

    def _pad_for_visualization(self, image, boxes_xyxy, masks, padding=60):
        h, w, c = image.shape
        new_h = h + 2 * padding
        new_w = w + 2 * padding
        padded_image = np.zeros((new_h, new_w, c), dtype=np.uint8)
        padded_image[padding:padding+h, padding:padding+w] = image
        adj_boxes = boxes_xyxy + padding
        n_masks = len(masks)
        adj_masks = np.zeros((n_masks, new_h, new_w), dtype=masks.dtype)
        adj_masks[:, padding:padding+h, padding:padding+w] = masks
        return padded_image, adj_boxes, adj_masks

    def visualize_debug(self, image_source, boxes_xyxy, masks, logits, use_padding=True):
        if masks is None or len(masks) == 0:
            return

        if use_padding:
            vis_image, vis_boxes, vis_masks = self._pad_for_visualization(image_source, boxes_xyxy, masks)
        else:
            vis_image, vis_boxes, vis_masks = image_source.copy(), boxes_xyxy, masks

        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()

        WINDOW_NAME = "Debug View"
        print(f"[DEBUG] Visualizing {len(vis_masks)} items. SPACE for next, ESC to quit.")
        
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        if vis_masks.ndim == 4: vis_masks = vis_masks.squeeze(1)

        for i in range(len(vis_masks)):            
            single_detection = sv.Detections(
                xyxy=vis_boxes[i:i+1],
                mask=vis_masks[i:i+1].astype(bool),
                confidence=logits[i:i+1],
                class_id=np.array([0]) 
            )
            
            annotated_overlay = vis_image.copy()
            annotated_overlay = mask_annotator.annotate(scene=annotated_overlay, detections=single_detection)
            annotated_overlay = box_annotator.annotate(scene=annotated_overlay, detections=single_detection)
            
            # Show raw mask on left
            single_mask_vis = (vis_masks[i] > 0.0).astype(np.uint8) * 255
            single_mask_vis = cv2.cvtColor(single_mask_vis, cv2.COLOR_GRAY2BGR)

            combined_view = np.hstack((single_mask_vis, annotated_overlay))
            
            cv2.imshow(WINDOW_NAME + f"{len(vis_masks)}", combined_view)
            key = cv2.waitKey(0) 
            if key == 27: break
        cv2.destroyAllWindows()

    def generate_masks(self, image_source, boxes, logits=None, debug=False):
        if len(boxes) == 0: return None, None
        self.model_manager.switch_to('sam')

        print("--> [Step 3] Segmenting with SAM 2...")
        h, w, _ = image_source.shape
        
        boxes_cxcywh = boxes.clone()
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * w
        boxes_xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * h
        boxes_xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * w
        boxes_xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * h

        self.model_manager.sam_predictor.reset_predictor()
        gc.collect()
        torch.cuda.empty_cache()

        self.model_manager.sam_predictor.set_image(image_source)

        # Batch predict — mask decoder memory scales with N boxes
        SAM_BATCH = 16
        all_masks = []
        boxes_np = boxes_xyxy.numpy()

        for i in range(0, len(boxes_np), SAM_BATCH):
            batch = boxes_np[i : i + SAM_BATCH]
            masks_batch, _, _ = self.model_manager.sam_predictor.predict(
                point_coords=None, point_labels=None,
                box=batch, multimask_output=False
            )
            if masks_batch.ndim == 4:
                masks_batch = masks_batch.squeeze(1)
            all_masks.append(masks_batch)

        # Release image features immediately after all batches done
        self.model_manager.sam_predictor.reset_predictor()
        gc.collect()
        torch.cuda.empty_cache()

        masks = np.concatenate(all_masks, axis=0)

        if debug:
            debug_logits = logits if logits is not None else torch.ones(len(boxes))
            self.visualize_debug(image_source, boxes_np, masks, debug_logits, use_padding=True)

        return boxes_np, masks

    def save_result(self, image_source, boxes_xyxy, masks, logits, file_name="result.jpg", use_padding=False, show_boxes = False):
        if use_padding:
            vis_image, vis_boxes, vis_masks = self._pad_for_visualization(image_source, boxes_xyxy, masks)
        else:
            vis_image, vis_boxes, vis_masks = image_source.copy(), boxes_xyxy, masks

        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()

        detections = sv.Detections(
            xyxy=vis_boxes,
            mask=vis_masks.astype(bool),
            class_id=np.zeros(len(vis_boxes)).astype(int),
            confidence=logits
        )

        labels = [f"{conf:.2f}" for conf in logits]

        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()

        annotated_image = vis_image.copy()
        annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
        
        if(show_boxes):
            try:
                label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            except AttributeError:
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        out_path = f"output/{file_name}"
        cv2.imwrite(out_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"--> [Saved] {out_path}")
