import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import os
import shutil
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from core_engine import AnnotateEngine
from utils import ImageProcessor
import torch.nn.functional as F

class Scout:
    """
    ADVERSARIAL ENTROPY SCOUT:
    1. Visual Entropy: Measures 'Messiness' via DINO Patch Self-Similarity.
    2. Semantic Confusion: Measures 'Ambiguity' via CLIP Hard Negatives.
    3. Clustering: Ensures scenario diversity.
    """
    def __init__(self, engine : AnnotateEngine):
        self.engine = engine
        self.device = engine.device
        
        # State
        self.candidates = [] # List of dicts
        
        # We define "Confusers" - things that look like potholes but aren't (or coexist)
        self.confusers = [
            "shadows", "tree shade", "manhole cover", "wet asphalt patch", 
            "water puddle", "cracks", "gravel", "oil stain", "leaf litter"
        ]

        self.clip_model = None
        self.clip_processor = None

    def _load_clip(self):
        if self.clip_model is None:
            print("[Scout] Loading CLIP (ViT-B/32)...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()

    def _compute_entropy(self, patch_tokens):
        """
        Calculates Visual Entropy from DINO patches.
        High Entropy = High Structural Complexity (Messy scene).
        Low Entropy = High Repetition (Clean road).
        """
        # patch_tokens: (1, N_patches, Dim)
        # Normalize
        p = F.normalize(patch_tokens, dim=2, p=2).squeeze(0) # (N, Dim)
        
        # Self-Similarity Matrix (N, N)
        # How similar is every patch to every other patch?
        sim_matrix = torch.matmul(p, p.T)
        
        # Convert to Probability Distribution (Softmax over rows)
        # "If I am Patch i, what is the probability I am semantically related to Patch j?"
        probs = F.softmax(sim_matrix / 0.1, dim=1) # Temp 0.1 sharpens peaks
        
        # Shannon Entropy per patch: -sum(p * log(p))
        # If a patch is unique (only similar to itself), entropy is LOW (0).
        # If a patch matches everything (blank wall), entropy is HIGH? 
        # Actually, let's invert the logic for "Information Content":
        # We want images where patches are DIVERSE (Low correlation off-diagonal).
        
        # Let's use a simpler proxy: Mean Off-Diagonal Correlation.
        # Clean Road -> High correlation everywhere.
        # Complex Scene -> Low correlation (patches are distinct).
        
        n = probs.shape[0]
        mask = torch.eye(n, device=self.device).bool()
        off_diag = sim_matrix.masked_select(~mask)
        
        # Variance of off-diagonal correlations
        # High Variance = Some things match, some things don't (Structure)
        # Low Variance = Everything matches (Blank) or Nothing matches (Noise)
        
        # We actually want 'Visual Information Content'. 
        # Let's use the singular value spectrum (Nuclear Norm) of the patch matrix.
        # High Rank = High Complexity.
        _, S, _ = torch.svd(p)
        complexity = torch.sum(torch.log(S + 1e-6)) # Log-Energy of spectrum
        
        return complexity.item()

    def mine_hard_examples(self, image_dir, prompt="pothole", top_k=50, batch_size=32):
        """
        The Master Pipeline.
        """
        image_files = glob.glob(os.path.join(image_dir, "*.*"))
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]
        
        if not image_files: return []

        print(f"[Scout] Analyzing {len(image_files)} images for Hardness & Complexity...")

        # --- STAGE 1: SEMANTIC SCORING (CLIP) ---
        self._load_clip()
        
        # Pre-compute text embeds for Target AND Confusers
        all_prompts = [prompt] + self.confusers
        text_inputs = self.clip_processor(text=all_prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.clip_model.get_text_features(**text_inputs) # (N_prompts, Dim)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            target_vec = text_embeds[0]
            confuser_vecs = text_embeds[1:]

        candidates = []

        for i in tqdm(range(0, len(image_files), batch_size), desc="Semantic Scan"):
            batch_paths = image_files[i:i+batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except: continue
            
            if not images: continue
            
            # CLIP Inference
            inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                img_embeds = self.clip_model.get_image_features(**inputs)
                img_embeds = F.normalize(img_embeds, p=2, dim=-1)
            
            # 1. Target Relevance
            relevance = torch.matmul(img_embeds, target_vec).cpu().numpy()
            
            # 2. Confusion Score (Max similarity to any confuser)
            # (B, Dim) @ (Dim, N_conf) -> (B, N_conf)
            confusions = torch.matmul(img_embeds, confuser_vecs.T)
            max_confusion, _ = torch.max(confusions, dim=1)
            max_confusion = max_confusion.cpu().numpy()
            
            # Store prelim results
            for j, path in enumerate(batch_paths):
                # Filter irrelevance immediately
                if relevance[j] > 0.15: 
                    candidates.append({
                        'path': path,
                        'relevance': relevance[j],
                        'confusion': max_confusion[j],
                        'clip_vec': img_embeds[j].cpu().numpy() # Keep for de-dupe
                    })

        # --- STAGE 2: VISUAL ENTROPY (DINO) ---
        # Only run on survivors
        print(f"[Scout] Computing Visual Entropy for {len(candidates)} survivors...")
        
        self.engine.model_manager.switch_to('dino')
        dino_model = self.engine.model_manager.models['dino']
        
        # Prepare for clustering
        dino_cls_vecs = []
        
        # We can't batch easily here due to variable image sizes logic in DINO transform
        # usually, but we assume resized batching.
        
        batch_cands = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
        
        for batch in tqdm(batch_cands, desc="Entropy Mining"):
            batch_tensors = []
            for c in batch:
                img, _ = ImageProcessor.load_image(c['path'])
                t = self.engine.dino_transform(Image.fromarray(img))
                batch_tensors.append(t)
                
            input_tensor = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                features = dino_model.forward_features(input_tensor)
                
                # [CLS] for clustering
                cls = features["x_norm_clstoken"]
                cls = F.normalize(cls, dim=1, p=2).cpu().numpy()
                dino_cls_vecs.extend(cls)
                
                # Patch Entropy for Complexity
                patch_tokens = features["x_norm_patchtokens"] # (B, N, Dim)
                
                for k in range(len(batch)):
                    # Compute entropy per image in batch
                    entropy = self._compute_entropy(patch_tokens[k:k+1])
                    batch[k]['entropy'] = entropy

        # --- STAGE 3: CLUSTER & SELECT ---
        print(f"[Scout] Clustering & Selecting Hardest Cases...")
        
        # Convert to arrays for sklearn
        dino_cls_vecs = np.array(dino_cls_vecs)
        
        # Cluster into K groups (Scenario Diversity)
        # We want 15-20 images, so maybe 15 clusters?
        n_clusters = min(15, len(candidates))
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(dino_cls_vecs)
        
        final_selection = []
        
        for c_id in range(n_clusters):
            # Get members of this cluster
            indices = np.where(labels == c_id)[0]
            cluster_cands = [candidates[i] for i in indices]
            
            # Finding the "Hardest" image in this cluster
            # Hardness = (High Relevance) + (High Confusion) + (High Entropy)
            # We want relevant potholes that are confusing and visually messy.
            
            # Normalize local scores for fair weighting
            rels = np.array([c['relevance'] for c in cluster_cands])
            # confs = np.array([c['confusion'] for c in cluster_cands])
            ents = np.array([c['entropy'] for c in cluster_cands])
            
            # Safe norm
            rels = (rels - rels.min()) / (np.ptp(rels) + 1e-6)
            # confs = (confs - confs.min()) / (np.ptp(confs) + 1e-6)
            ents = (ents - ents.min()) / (np.ptp(ents) + 1e-6)
            
            # Weighted Score:
            # 40% Entropy (Messiness)
            # 40% Confusion (Semantic Hardness)
            # 20% Relevance (Just to ensure it's not total junk)
            scores = (0.5 * rels) + (0.5 * ents)
            
            best_local_idx = np.argmax(scores)
            best_cand = cluster_cands[best_local_idx]
            
            # Add metadata for debug filename
            best_cand['final_score'] = scores[best_local_idx]
            final_selection.append(best_cand)

        return final_selection

    def export(self, selection, export_dir="assets/calibration_pool"):
        os.makedirs(export_dir, exist_ok=True)
        print(f"[Scout] Exporting {len(selection)} hard-mined images...")
        
        for c in selection:
            # E=Entropy, C=Confusion, R=Relevance
            fname = f"E{c['entropy']:.1f}_C{c['confusion']:.2f}_R{c['relevance']:.2f}_" + os.path.basename(c['path'])
            dst = os.path.join(export_dir, fname)
            shutil.copy2(c['path'], dst)