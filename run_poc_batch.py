from core_engine import AnnotateEngine
from utils import VisualDebugger
from hitl_engine import ActiveCalibrationSession
import os
import glob
import numpy as np
import json
import torch
from scout import Scout
import gc

TARGET_DIR = "assets/test_images/"  # Directory containing images to process
OUTPUT_DIR = "outputs/"             # Directory to save the results
ANCHOR_DIR = "assets/anchors/Extra/"
TEXT_PROMPT = "A pothole."
FINETUNE_PATH = "verifier/dino_finetune.pt"

MODEL_PKG_PATH = "verifier/pothole_verifier_v17.pkl"
DATASET_PATH = None

def save_dataset(verified_data, output_path):
    """Saves the verification data (paths and tensors) to a JSON file."""
    serializable_data = []
    for item in verified_data:
        pos = item['pos']
        neg = item['neg']

        def to_list(data):
            if isinstance(data, torch.Tensor):
                return data.tolist()
            elif isinstance(data, np.ndarray):
                return data.tolist()
            return data

        serializable_data.append({
            "path": item['path'],
            "pos": to_list(pos),
            "neg": to_list(neg)
        })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    print(f"--> [System] Training dataset saved to {output_path}")

def load_dataset(input_path):
    """Loads the JSON dataset and converts lists back to PyTorch Tensors."""
    if input_path is None: return None
    if not os.path.exists(input_path):
        return None
        
    print(f"--> [System] Loading saved training dataset from {input_path}")
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    
    verified_data = []
    for item in raw_data:
        verified_data.append({
            "path": item['path'],
            "pos": torch.tensor(item['pos'], dtype=torch.float32),
            "neg": torch.tensor(item['neg'], dtype=torch.float32)
        })
    return verified_data

def main():
    engine = AnnotateEngine()
    engine.load_finetuned_dino(FINETUNE_PATH)

    verifier_pkg = engine.load_verifier(MODEL_PKG_PATH)
    saved_dataset = load_dataset(DATASET_PATH)
    
    train_metadata = None

    if saved_dataset is not None:
        print(f"--> [Setup] Found saved dataset ({len(saved_dataset)} images). Skipping manual calibration.")
        engine.fine_tune_dino(saved_dataset, save_path=FINETUNE_PATH, n_epochs=80, apply_from_block=20)
        engine.calibrate_size_thresholds(saved_dataset)

        verifier_model, X_train, y_train, train_metadata = engine.train_verifier(
            saved_dataset, save_path=MODEL_PKG_PATH, fast_train=False
        )
        
        verifier_pkg = {
            'model': verifier_model,
            'X': X_train,
            'y': y_train,
            'meta': train_metadata
        }

    elif verifier_pkg is None:
        engine.load_finetuned_dino(FINETUNE_PATH)
        print("--> [Calibrate] No saved model or dataset found. Starting calibration...")
        anchor_files = glob.glob(os.path.join(ANCHOR_DIR, "*.*"))
        
        if not anchor_files:
            print("[Error] Please add reference images to assets/anchors/Extra/")
            return
        
        session = ActiveCalibrationSession(
            engine, 
            image_paths=anchor_files, 
            prompt=TEXT_PROMPT, 
            model_save_path=MODEL_PKG_PATH, 
            initial_batch_size=20
        )
    
        verified_data = session.start()

        if verified_data and len(verified_data) > 0:
            engine.calibrate_size_thresholds(verified_data)
            save_dataset(verified_data, DATASET_PATH)
            verifier_pkg = engine.load_verifier(MODEL_PKG_PATH)
        else:
            print("[Abort] Calibration cancelled or no data collected.")
            return

    else:
        engine.load_finetuned_dino(FINETUNE_PATH)
        print("--> [Setup] Loaded cached model directly (No retraining).")

    if verifier_pkg is None:
        print("[Error] Failed to load or train a verifier.")
        return
   
    engine.load_finetuned_dino(FINETUNE_PATH)

    verifier_model = verifier_pkg['model']
    
    # --- BATCH PROCESSING LOGIC ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Grab all typical image formats from the target directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    target_images = []
    for ext in image_extensions:
        target_images.extend(glob.glob(os.path.join(TARGET_DIR, ext)))
        
    if not target_images:
        print(f"[Warning] No images found in directory: {TARGET_DIR}")
        return
        
    print(f"\n--> Starting batch processing for {len(target_images)} images...")

    for img_path in target_images:
        base_name = os.path.basename(img_path).split('.')[0]
        print(f"\n--> Processing Target: {img_path}")
        
        try:
            boxes, logits, _, image_source = engine.detect_objects(
                img_path, 
                TEXT_PROMPT,
                box_threshold=0.05,
                text_threshold=0.20,
                batch_size=4
            )

            if len(boxes) > 0:
                filtered_boxes, svm_scores, reject_boxes, reject_scores = engine.filter_candidates(
                    image_source, boxes, verifier_model, confidence_threshold=0.5, force_cpu=False
                )

                # Visualize rejected boxes
                rej_boxes_xyxy, rej_masks = engine.generate_masks(image_source, reject_boxes, logits=reject_scores, debug=False)

                if rej_boxes_xyxy is not None:
                    engine.save_result(
                        image_source, rej_boxes_xyxy, rej_masks, reject_scores,
                        file_name=os.path.join(OUTPUT_DIR, f"{base_name}_rejected.jpg"), 
                        use_padding=True, show_boxes=True
                    )
                    
                del rej_boxes_xyxy, rej_masks, reject_boxes, reject_scores
                
                # Visualize accepted boxes
                boxes_xyxy, masks = engine.generate_masks(
                    image_source, filtered_boxes, logits=svm_scores, debug=False
                )

                if boxes_xyxy is not None:
                    engine.save_result(
                        image_source, boxes_xyxy, masks, svm_scores, 
                        file_name=os.path.join(OUTPUT_DIR, f"{base_name}_calibrated.jpg"), 
                        use_padding=True, show_boxes=False
                    )
                else:
                    print(f"    No accepted candidates for {base_name}.")
            else:
                print(f"    No candidates initially detected for {base_name}.")
                
        except Exception as e:
            print(f"[Error] Failed processing {img_path}: {e}")

        finally:
            # Explicit cleanup per iteration to prevent memory leaks
            gc.collect()
            torch.cuda.empty_cache()

    print("\n--> Batch processing complete!")

if __name__ == "__main__":
    main()
