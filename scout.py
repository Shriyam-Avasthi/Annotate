import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import os
import shutil
from transformers import CLIPProcessor, CLIPModel
from core_engine import AnnotateEngine
from utils import ImageProcessor
import torch.nn.functional as F

class Scout:
    def __init__(self, engine : AnnotateEngine):
        self.engine = engine
        self.device = engine.device
        
        # State
        self.filtered_paths = None
        self.dataset_scores = None          # CLIP Relevance Scores
        self.dataset_complexities = None    # DINO Outlier Scores
        self.dataset_embeddings = None      # CLIP Embeddings (For De-duplication)
        
        self.clip_model = None
        self.clip_processor = None
        
        print(f"[Scout] Ready. Pipeline: CLIP (Gate + De-Dup) -> DINO (Miner).")

    def _load_clip(self):
        if self.clip_model is None:
            print("[Scout] Loading CLIP (ViT-B/32)...")
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", 
                use_safetensors=True
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()

    def semantic_sieve(self, image_dir, prompt_text="pothole", keep_top_k=500, batch_size=64):
        """
        Stage 1: Filters by relevance AND stores embeddings for de-duplication later.
        """
        self._load_clip()
        
        image_files = glob.glob(os.path.join(image_dir, "*.*"))
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]
        
        if not image_files: return []

        print(f"[Scout] Sieving {len(image_files)} images for: '{prompt_text}'...")
        
        # Text Embeds
        inputs = self.clip_processor(text=[prompt_text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.clip_model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        scores = []
        embeddings = [] # Store these!
        
        for i in tqdm(range(0, len(image_files), batch_size), desc="CLIP Filtering"):
            batch_paths = image_files[i:i+batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except: continue
            
            if not images: continue
            
            inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                image_embeds = self.clip_model.get_image_features(**inputs)
                image_embeds = F.normalize(image_embeds, p=2, dim=-1)
                
                # Similarity
                sims = torch.matmul(image_embeds, text_embeds.T).flatten()
                
                # Store
                scores.extend(sims.cpu().numpy().tolist())
                embeddings.append(image_embeds.cpu().numpy())

        # Consolidate
        scores = np.array(scores)
        embeddings = np.vstack(embeddings)
        
        # Top-K Filter
        top_indices = np.argsort(scores)[::-1][:keep_top_k]
        
        self.filtered_paths = np.array(image_files)[top_indices]
        self.dataset_scores = scores[top_indices]
        self.dataset_embeddings = embeddings[top_indices] # Keep aligned
        
        # Cleanup
        del self.clip_model
        del self.clip_processor
        self.clip_model = None
        torch.cuda.empty_cache()
        
        print(f"[Scout] Sieve complete. Kept {len(self.filtered_paths)} candidates.")
        return self.filtered_paths

    def mine_complexity(self, batch_size=32):
        """
        Stage 2: Calculates 'Foreground Mass' instead of raw variance.
        This counts distinct objects/patches rather than just measuring contrast.
        """
        if self.filtered_paths is None:
            print("[Scout] Error: Run semantic_sieve() first.")
            return

        print(f"[Scout] Mining complexity (Foreground Mass) on survivors...")
        
        self.engine.model_manager.switch_to('dino')
        dino_model = self.engine.model_manager.models['dino']
        
        complexities = []
        
        for i in tqdm(range(0, len(self.filtered_paths), batch_size), desc="DINO Mining"):
            batch_paths = self.filtered_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    img, _ = ImageProcessor.load_image(path)
                    pil_img = Image.fromarray(img)
                    t_img = self.engine.dino_transform(pil_img)
                    batch_tensors.append(t_img)
                except: continue
            
            if not batch_tensors: continue

            input_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                features = dino_model.forward_features(input_tensor)
                patch_tokens = features["x_norm_patchtokens"] # (B, N, Dim)
                
                # === NEW METRIC: Foreground Mass ===
                # 1. Calculate the "Mean Patch" for each image (The Background)
                mean_patch = torch.mean(patch_tokens, dim=1, keepdim=True) # (B, 1, Dim)
                
                # 2. Calculate distance of every patch from the mean
                # (B, N, Dim) - (B, 1, Dim) -> (B, N)
                dists = torch.norm(patch_tokens - mean_patch, dim=2, p=2)
                
                # 3. Count "Significant" Patches (Outliers)
                # A patch is "foreground" if it deviates more than 1 std_dev from the mean
                thresholds = torch.mean(dists, dim=1, keepdim=True) + torch.std(dists, dim=1, keepdim=True)
                
                # We sum the MAGNITUDE of the deviation for the outliers
                # This rewards having *many* potholes or *large* potholes
                mask = dists > thresholds
                foreground_energy = torch.sum(dists * mask.float(), dim=1)
                
                complexities.extend(foreground_energy.cpu().numpy().tolist())

        self.dataset_complexities = np.array(complexities)
        print("[Scout] Mining complete.")

    def export_candidates(self, top_k=50, export_dir="assets/calibration_pool", dedupe_threshold=0.95):
        """
        Exports candidates with strictly enforced DE-DUPLICATION and RELEVANCE GATING.
        dedupe_threshold: 0.95 means if images are 95% similar, drop the second one.
        """
        if self.dataset_complexities is None: return []
        
        print(f"[Scout] De-duping and Exporting top {top_k}...")
        
        # 1. Gated Scoring
        # We multiply Complexity by Relevance. 
        # If Relevance is low (it's a forest), Complexity (messy leaves) is ignored.
        
        # Normalize both to 0-1
        s_norm = (self.dataset_scores - self.dataset_scores.min()) / (np.ptp(self.dataset_scores) + 1e-6)
        c_norm = (self.dataset_complexities - self.dataset_complexities.min()) / (np.ptp(self.dataset_complexities) + 1e-6)
        
        # Gated Score: Relevance is the Gatekeeper
        final_score = s_norm * (1.0 + c_norm) 
        
        # Sort desc
        sorted_indices = np.argsort(final_score)[::-1]
        
        # 2. Greedy De-Duplication
        selected_indices = []
        selected_vectors = []
        
        # We need the embeddings we saved in Step 1
        all_vecs = self.dataset_embeddings
        
        for idx in sorted_indices:
            if len(selected_indices) >= top_k: break
            
            candidate_vec = all_vecs[idx]
            
            # Check against all previously selected
            is_duplicate = False
            if selected_vectors:
                # Batch Dot Product
                # (M, 512) @ (512,) -> (M,)
                sims = np.dot(selected_vectors, candidate_vec)
                if np.any(sims > dedupe_threshold):
                    is_duplicate = True
            
            if not is_duplicate:
                selected_indices.append(idx)
                selected_vectors.append(candidate_vec)
        
        print(f"[Scout] Selected {len(selected_indices)} unique images.")

        # 3. Export
        os.makedirs(export_dir, exist_ok=True)
        exported_paths = []
        
        for idx in selected_indices:
            path = self.filtered_paths[idx]
            rel = self.dataset_scores[idx]
            comp = self.dataset_complexities[idx]
            
            # Filename for debug
            fname = f"Score{final_score[idx]:.2f}_R{rel:.2f}_C{comp:.2f}_" + os.path.basename(path)
            dst = os.path.join(export_dir, fname)
            shutil.copy2(path, dst)
            exported_paths.append(dst)
            
        return exported_paths