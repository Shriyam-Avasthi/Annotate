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

# Grounding DINO Imports
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict

# SAM 2 Imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# For DinoV2
from torchvision import transforms
from torchvision.ops import box_iou

from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

class ModelManager:
    """
    Manages VRAM resources. Ensures only one model is on the GPU at a time,
    but prevents unnecessary unloading/reloading if the same model is requested twice.
    """
    def __init__(self, gd_model, sam_model, sam_predictor, dino_model, device="cuda"):
        self.device = device
        self.cpu = "cpu"
        
        self.models = {
            "gd": gd_model,
            "sam": sam_model,
            "dino": dino_model
        }
        self.sam_predictor = sam_predictor
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
            return
        
        if self.current_key is not None:
            print(f"    [VRAM] Offloading {self.current_key}...")
            self.models[self.current_key].to(self.cpu)
            
            # Specific cleanup for SAM 2 to prevent memory leaks
            if self.current_key == "sam":
                self.sam_predictor.reset_predictor()
            
            self.current_key = None
            self._flush_vram()

        print(f"    [VRAM] Loading {key}...")
        self.models[key].to(self.device)
        
        if key == "sam":
            self.sam_predictor.model = self.models[key]
            
        self.current_key = key

class Augmentor:
    """
    Generates 4 views of every crop to make the model robust to 
    lighting and orientation changes.
    """
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ])

    def augment(self, crop_tensors):
        """
        Input: Tensor (N, 3, 224, 224)
        Output: Tensor (N, 3, 224, 224) with variations
        """
        if len(crop_tensors) == 0: return crop_tensors
        return self.transforms(crop_tensors)

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

        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dino_model.to(self.cpu) # Keep on CPU until needed
        dino_model.eval()

        self.model_manager = ModelManager(gd_model, sam_model, sam_predictor, dino_model, device)
        self.STANDARD_IMAGE_SCALE = 1280

        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[Init] Ready.")

    def _get_gd_transform(self):
        # Standard transform (Resize handled manually in slicer)
        return T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
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

    def _filter_contained_boxes(self, boxes_xyxy, logits, threshold=0.85):
        if len(boxes_xyxy) == 0: return boxes_xyxy, logits
        n = len(boxes_xyxy)
        keep = np.ones(n, dtype=bool)
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

        for i in range(n):
            if not keep[i]: continue
            for j in range(n):
                if i == j or not keep[j]: continue
                xx1 = max(boxes_xyxy[i, 0], boxes_xyxy[j, 0])
                yy1 = max(boxes_xyxy[i, 1], boxes_xyxy[j, 1])
                xx2 = min(boxes_xyxy[i, 2], boxes_xyxy[j, 2])
                yy2 = min(boxes_xyxy[i, 3], boxes_xyxy[j, 3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter_area = w * h
                
                # Check if I is inside J
                if inter_area / (areas[i] + 1e-6) > threshold:
                    # If small box (I) is inside large box (J),
                    # usually the small box is the "better" detection.
                    # But if the large box has MUCH higher confidence, we might doubt the small one.
                    if logits[i] >= logits[j]:
                        keep[j] = False 
                    else:
                        keep[i] = False
                        break 
        return boxes_xyxy[keep], logits[keep]
    
    def extract_dino_features(self, crop_tensor_batch, boxes=None, force_cpu=False):
        """
        Extracts DINOv2 features with batch chunking to prevent OOM on 4GB VRAM.
        
        Args:
            crop_tensor_batch: Tensor of image crops (N, C, H, W)
            force_cpu: If True, runs on CPU to avoid disturbing the GPU (used during background detection).
        """ 
        if len(crop_tensor_batch) == 0: 
            return torch.empty(0)

        chunk_size = 32
        embeddings_list = []

        if force_cpu:
            dino_model = self.model_manager.models['dino']
            dino_model.to("cpu")

            if self.model_manager.current_key == 'dino':
                self.model_manager.current_key = None
            
            with torch.no_grad():
                for i in range(0, len(crop_tensor_batch), chunk_size):
                    batch = crop_tensor_batch[i:i+chunk_size].to("cpu")
                    emb = dino_model(batch)
                    embeddings_list.append(emb)

        else:
            self.model_manager.switch_to('dino')
            
            with torch.no_grad():
                for i in range(0, len(crop_tensor_batch), chunk_size):
                    batch = crop_tensor_batch[i:i+chunk_size].to(self.device)
                    emb = self.model_manager.models['dino'](batch)
                    embeddings_list.append(emb.cpu())

        if not embeddings_list: 
            return torch.empty(0)
            
        visual_feats = torch.cat(embeddings_list)
        visual_feats = F.normalize(visual_feats, dim=1, p=2)

        # if boxes is not None:
        #     if not isinstance(boxes, torch.Tensor):
        #         boxes = torch.from_numpy(boxes).float()
            
        #     boxes = boxes.to(visual_feats.device)

        #     areas = (boxes[:, 2] * boxes[:, 3]).unsqueeze(1) / (self.STANDARD_IMAGE_SCALE * self.STANDARD_IMAGE_SCALE)
        #     ratios = (boxes[:, 2] / (boxes[:, 3] + 1e-6)).unsqueeze(1)

        #     geom_feats = torch.cat([torch.sqrt(areas), torch.log(ratios + 1)], dim=1)

        #     # Concatenate [Visual(384) + Geom(2)] -> 386 dim
        #     return torch.cat([visual_feats, geom_feats], dim=1)

        return visual_feats

    def _get_crops_from_boxes(self, image_source, boxes, min_pad_px=10, padding_factor=0.10):
        """
        Added 'padding_factor' to give context to small objects.
        """
        if len(boxes) == 0: return []
        
        img_h, img_w, _ = image_source.shape
        pil_img = Image.fromarray(image_source)
        crops = []

        if not isinstance(boxes, torch.Tensor):
            boxes = torch.from_numpy(boxes).float()
        
        if boxes.numel() > 0 and boxes.max() <= 1.0:
            boxes_abs = boxes.clone()
            boxes_abs[:, 0] *= img_w
            boxes_abs[:, 1] *= img_h
            boxes_abs[:, 2] *= img_w
            boxes_abs[:, 3] *= img_h
        else:
            boxes_abs = boxes
        
        for box in boxes_abs:
            cx, cy, w, h = box.tolist()
            
            pad_w = max(min_pad_px, w * padding_factor)
            pad_h = max(min_pad_px, h * padding_factor)
            
            x1 = int((cx - w/2) - pad_w)
            y1 = int((cy - h/2) - pad_h)
            x2 = int((cx + w/2) + pad_w)
            y2 = int((cy + h/2) + pad_h)
            
            # Clamp to image boundaries
            crop = pil_img.crop((max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))
            crops.append(self.dino_transform(crop))
            
        return torch.stack(crops) if crops else torch.empty(0)
    
    # def _visualize_calibration_debug(self, image_source, pos_boxes, neg_boxes):
    #     """
    #     Visualizes positive (Green) and negative (Red) samples for calibration.
    #     Expects normalized cxcywh boxes.
    #     """
    #     h, w, _ = image_source.shape
    #     vis_image = image_source.copy()

    #     # Helper to convert Normalized cxcywh -> Absolute xyxy
    #     def convert_boxes(boxes):
    #         if len(boxes) == 0: return np.empty((0, 4))
    #         xyxy = boxes.clone()
    #         xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    #         xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    #         xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    #         xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    #         return xyxy.numpy()

        # pos_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
        # neg_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)

        # if len(neg_boxes) > 0:
        #     neg_xyxy = convert_boxes(neg_boxes)
        #     neg_detections = sv.Detections(xyxy=neg_xyxy, class_id=np.array([0] * len(neg_xyxy)))
        #     vis_image = neg_annotator.annotate(scene=vis_image, detections=neg_detections)

        # if len(pos_boxes) > 0:
        #     pos_xyxy = convert_boxes(pos_boxes)
        #     pos_detections = sv.Detections(xyxy=pos_xyxy, class_id=np.array([1] * len(pos_xyxy)))
        #     vis_image = pos_annotator.annotate(scene=vis_image, detections=pos_detections)

        # cv2.imshow("Calibration Debug (Green=Pos, Red=Neg) - Press Key", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def _save_verifier(self, model, file_path="weights/verifier.pkl"):
        """Saves the trained SVM model to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        print(f"--> [System] verifier model saved to {file_path}")

    def load_verifier(self, file_path="weights/verifier.pkl"):
        """Loads a pre-trained SVM model from disk."""
        if not os.path.exists(file_path):
            print(f"--> [System] No model found at {file_path}")
            return None
        
        print(f"--> [System] Loading verifier model from {file_path}...")
        return joblib.load(file_path)
    
    def train_verifier(self, verified_data, save_path="weights/verifier.pkl"):
        """
        Phase 2 of Calibration: Extracts features, trains SVM, and SAVES METADATA for debugging.
        """
        print(f"\n--> [Training] Processing {len(verified_data)} verified images...")
        self.model_manager.switch_to('dino')
        augmentor = Augmentor()
        
        X_feats = []
        y_labels = []
        train_metadata = [] 
        X_geom = []
        
        total_pos = 0
        total_neg = 0
        
        for i, item in enumerate(verified_data):
            if len(item['pos']) == 0 and len(item['neg']) == 0: continue
            
            image_source, _ = ImageProcessor.load_image(item['path'], self.STANDARD_IMAGE_SCALE)
            h, w, _ = image_source.shape
            
            # Helper to process boxes
            def process_boxes(boxes_norm, label_val):
                if len(boxes_norm) == 0: return 0
                if isinstance(boxes_norm, np.ndarray):
                    boxes_t = torch.from_numpy(boxes_norm).float()
                else:
                    boxes_t = boxes_norm.float()

                base_crops = self._get_crops_from_boxes(image_source, boxes_t, min_pad_px=10, padding_factor=0.2)
                if len(base_crops) == 0: return 0
                
                # Augmentation Loop
                crops_to_process = [base_crops]
                for _ in range(3):
                    crops_to_process.append(augmentor.augment(base_crops))
                all_crops = torch.cat(crops_to_process)

                all_boxes = boxes_t.repeat(4, 1)

                feats = self.extract_dino_features(all_crops, boxes=all_boxes)

                X_feats.append(feats)
                y_labels.append(torch.full((len(feats),), label_val))
                
                b = boxes_norm.clone().numpy() if hasattr(boxes_norm, 'clone') else boxes_norm.copy()
                x1 = (b[:, 0] - b[:, 2]/2) * w
                y1 = (b[:, 1] - b[:, 3]/2) * h
                x2 = (b[:, 0] + b[:, 2]/2) * w
                y2 = (b[:, 1] + b[:, 3]/2) * h
                abs_boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)
                
                batch_meta = []
                for box in abs_boxes:
                    batch_meta.append((item['path'], box))
                
                train_metadata.extend(batch_meta * 4)
                return len(feats)

            total_pos += process_boxes(item['pos'], 1)
            total_neg += process_boxes(item['neg'], 0)
                    
            print(f"    [Embedder] {i+1}/{len(verified_data)} processed.")

        if total_pos == 0 or total_neg == 0:
            print(f"[Error] Training failed. Insufficient data.")
            return None, None, None, None

        print(f"--> [Training] Fitting SVM on {total_pos} Positives vs {total_neg} Negatives...")
        X = torch.cat(X_feats).cpu().numpy()
        y = torch.cat(y_labels).numpy()
        
        # svm = LinearSVC(class_weight='balanced', dual="auto", max_iter=2000)
        # clf = CalibratedClassifierCV(svm)
        # clf.fit(X, y)

        # clf = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, class_weight='balanced')
        # clf.fit(X, y)

        clf = EnsembleVerifier()
        clf.fit(X, y)
        
        package = {
            'model': clf, 
            'X': X, 
            'y': y,
            'meta': train_metadata
        }
        self._save_verifier(package, save_path)
        
        return clf, X, y, train_metadata

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
        
        SCALES = [160, 320, 640, None] 
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
                is_l, is_t = (sx == 0), (sy == 0)
                is_r, is_b = (sx + sw >= img_w), (sy + sh >= img_h)

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
            
            final_boxes_np, final_logits_np = self._filter_contained_boxes(final_boxes_np, final_logits_np, threshold=0.95)
            final_boxes_np, final_logits_np = self._filter_small_area(final_boxes_np, final_logits_np, min_area=100)

            # Debug Visualization (Before Conversion)
            # if debug and len(final_boxes_np) > 0:
            #     vis_image = image_source.copy()
            #     vis_annotator = sv.BoxAnnotator()
            #     detections = sv.Detections(xyxy=final_boxes_np, class_id=np.zeros(len(final_boxes_np)).astype(int))
            #     vis_image = vis_annotator.annotate(scene=vis_image, detections=detections)
            #     cv2.imshow("Detection Debug (Press Key)", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

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
    
    def filter_candidates(self, image_source, boxes, classifier, confidence_threshold = 0.5, force_cpu = False):
        if len(boxes) == 0 or classifier is None: return boxes, None
        
        candidate_crops = self._get_crops_from_boxes(image_source, boxes)
        features_tensor = self.extract_dino_features(candidate_crops, boxes=boxes, force_cpu=force_cpu)
        candidate_feats = features_tensor.cpu().numpy() 
        
        probs = classifier.predict_proba(candidate_feats)[:, 1] 
        keep_mask = probs > confidence_threshold
        
        keep_boxes = boxes[keep_mask]
        keep_scores = torch.tensor(probs[keep_mask])

        reject_boxes = boxes[~keep_mask]
        reject_scores = torch.tensor(probs[~keep_mask])
        print(f"    [Filter] SVM rejected {len(boxes) - len(keep_boxes)} false positives.")
        return keep_boxes, keep_scores , reject_boxes, reject_scores

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
        
        print(f"--> [Step 3] Segmenting with SAM 2...")
        h, w, _ = image_source.shape
        boxes_cxcywh = boxes.clone() 
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * w
        boxes_xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * h
        boxes_xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * w
        boxes_xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * h

        self.model_manager.sam_predictor.set_image(image_source)
        
        masks, scores, _ = self.model_manager.sam_predictor.predict(
            point_coords=None, point_labels=None,
            box=boxes_xyxy.numpy(), multimask_output=False
        )
        if masks.ndim == 4: masks = masks.squeeze(1)

        if debug:
            if logits is None: debug_logits = torch.ones(len(boxes))
            else: debug_logits = logits
            self.visualize_debug(image_source, boxes_xyxy.numpy(), masks, debug_logits, use_padding=True)
        
        return boxes_xyxy.numpy(), masks

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