from core_engine import AnnotateEngine
from utils import VisualDebugger
from hitl_engine import ActiveCalibrationSession
import os
import glob
import numpy as np
import json
import torch
from scout import Scout

TARGET_IMAGE = "assets/test-2.jpg"
ANCHOR_DIR = "assets/anchors/Extra/"
TEXT_PROMPT = "A pothole."

MODEL_PKG_PATH = "verifier/pothole_verifier_v15.pkl"
# This is the new file where we will save the boxes/paths
DATASET_PATH = "verifier/pothole_training_data_2.json"
# DATASET_PATH = None

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
    # scout = Scout(engine)
    # debugger = VisualDebugger()

    verifier_pkg = engine.load_verifier(MODEL_PKG_PATH)
    saved_dataset = load_dataset(DATASET_PATH)
    
    train_metadata = None

    if saved_dataset is not None:
        print(f"--> [Setup] Found saved dataset ({len(saved_dataset)} images). Skipping manual calibration.")
        
        engine.calibrate_size_thresholds(saved_dataset)

        # n_trials = 50
        # study = engine.tune_context_factors_optuna(saved_dataset, n_trials=n_trials)

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
        print("--> [Setup] Loaded cached model directly (No retraining).")

    if verifier_pkg is None:
        print("[Error] Failed to load or train a verifier.")
        return

    verifier_model = verifier_pkg['model']
    X_train = verifier_pkg['X']
    y_train = verifier_pkg['y']
    train_metadata = verifier_pkg.get('meta', None)

    print(f"\n--> Processing Target: {TARGET_IMAGE}")
    
    boxes, logits, _, image_source = engine.detect_objects(
        TARGET_IMAGE, 
        TEXT_PROMPT,
        box_threshold=0.05,
        text_threshold=0.20,
        batch_size=4
    )

    if len(boxes) > 0:
        filtered_boxes, svm_scores, reject_boxes, reject_scores = engine.filter_candidates(image_source, boxes, verifier_model, confidence_threshold=0.5, force_cpu=False)

        # Visualize rejected boxes
        rej_boxes_xyxy, rej_masks = engine.generate_masks(image_source, reject_boxes, logits=reject_scores, debug=False)

        if rej_boxes_xyxy is not None:
            engine.save_result(
                image_source, rej_boxes_xyxy, rej_masks, reject_scores,
                file_name="rejected_by_svm.jpg", use_padding=True, show_boxes=True
            )
            
        boxes_xyxy, masks = engine.generate_masks(
            image_source, filtered_boxes, logits=svm_scores, debug=False
        )

        if boxes_xyxy is not None:
            engine.save_result(
                image_source, boxes_xyxy, masks, svm_scores, 
                file_name="calibrated_result.jpg", use_padding=True, show_boxes=False
            )
        else:
            print("No candidates detected.")

if __name__ == "__main__":
    main()
