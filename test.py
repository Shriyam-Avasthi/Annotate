from core_engine_archived import AnnotateEngine
from utils import VisualDebugger
from hitl_engine import ActiveCalibrationSession
import os
import glob
import numpy as np
from scout2 import Scout

TARGET_IMAGE = "assets/test-3.jpg"
ANCHOR_DIR = "assets/anchors/full_images/" 
TEXT_PROMPT = "Complex images of roads with potholes, difficult for AI models to detect with high accuracy."

MODEL_PKG_PATH = "verifier/pothole_verifier_v5.pkl"

def main():
    engine = AnnotateEngine()
    scout = Scout(engine)
    debugger = VisualDebugger()

    verifier_pkg = engine.load_verifier(MODEL_PKG_PATH)
    train_metadata = None

    # =======================================================================
    # scout.semantic_sieve(
    #     image_dir="assets/raw_pool/", 
    #     prompt_text="potholes, road damage, cracks in asphalt", 
    #     keep_top_k=500
    # )
    
    # # 2. Complexity Mining (Slow Detail)
    # # Only runs DINO on those 500 images
    # scout.mine_complexity()
    
    # # 3. Export Balanced Samples
    # # Finds images that are BOTH "relevant" (potholes) AND "complex" (hard for the model)
    # scout.export_candidates(top_k=50)
    # ===================================================

    # # 1. Sieve (CLIP) -> Get Top 500 relevant
    # scout.semantic_sieve("assets/raw_pool/", prompt_text=TEXT_PROMPT, keep_top_k=500)
    
    # # 2. Analyze (DINO) -> Get Features for those 500
    # scout.compute_dino_features()
    
    # # 3. Select (Hybrid) -> 25 High Rel + 25 High Complexity Clusters
    # scout.export_hybrid(total_k=50)

    # =======================================================================

    scout = Scout(engine)
    
    # Run the full pipeline
    # "Mine Hard Examples" does Sieve -> Entropy -> Cluster -> Select
    hardest_images = scout.mine_hard_examples(
        "assets/raw_pool/", 
        prompt="pothole", 
        top_k=20
    )
    
    scout.export(hardest_images)

    # =======================================================================

    # scout.analyze(
    #     image_dir="assets/raw_pool/", 
    #     prompt="potholes, road damage, cracks in asphalt"
    # )
    # scout.export_stratified(top_k=50)
    # scout.calibrate("assets/raw_pool/", sample_size=1000)
    
    # has_anchors = scout.load_anchors("assets/anchors/full_images/")
    # scout.index_dataset("assets/raw_pool/")
    
    # # if has_anchors:
    # calibration_images = scout.mine_targeted(top_k=20, min_conf=0.25)
    # else:
    #     print("[System] No anchors found. Falling back to generic diversity.")
    #     calibration_images = scout.sample_diversity(n_clusters=20)
    

    # if verifier_pkg is None:
    #     print("--> [Calibrate] No saved model found. Starting calibration...")
    #     anchor_files = glob.glob(os.path.join(ANCHOR_DIR, "*.*"))
        
    #     if not anchor_files:
    #         print("[Error] Please add reference images to assets/anchors/full_images/")
    #         return
        
    #     session = ActiveCalibrationSession(engine, image_paths= calibration_images, prompt= TEXT_PROMPT, model_save_path=MODEL_PKG_PATH)
    #     verified_data = session.start()

    #     verifier_pkg = engine.load_verifier(MODEL_PKG_PATH)

    #     # if verified_data is None:
    #     #     print("[Abort] Calibration cancelled.")
    #     #     return

    #     # verifier_model, X_train, y_train, train_metadata = engine.train_verifier(verified_data, save_path=MODEL_PKG_PATH)
    # else:
    #     print("--> [Setup] Loaded cached model & training data.")

    # verifier_model = verifier_pkg['model']
    # X_train = verifier_pkg['X']
    # y_train = verifier_pkg['y']
    # train_metadata = verifier_pkg.get('meta', None)
    # if train_metadata is None:
    #     print("[Warning] Cached model has no debug metadata. Training point clicks won't work.")
        
    # print(f"\n--> Processing Target: {TARGET_IMAGE}")
    
    # boxes, logits, _, image_source = engine.detect_objects(
    #     TARGET_IMAGE, 
    #     TEXT_PROMPT,
    #     box_threshold=0.05,
    #     text_threshold=0.20
    # )

    # if len(boxes) > 0:
    #     filtered_boxes, svm_scores, _, _ = engine.filter_candidates(
    #         image_source, boxes, verifier_model, confidence_threshold=0.3
    #     )
        
    #     candidate_crops_t = engine._get_crops_from_boxes(image_source, boxes, padding_factor=0.2)
    #     candidate_feats = engine._extract_dino_features(candidate_crops_t, boxes=boxes).cpu().numpy()
        
    #     h, w, _ = image_source.shape
    #     boxes_abs = boxes.clone().numpy()
    #     boxes_abs[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * w
    #     boxes_abs[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * h
    #     boxes_abs[:, 2] = (boxes[:, 0] + boxes[:, 2]/2) * w
    #     boxes_abs[:, 3] = (boxes[:, 1] + boxes[:, 3]/2) * h

    # else:
    #     candidate_feats = None
    #     boxes_abs = None
    
    # if candidate_feats is not None:
    #     debugger.plot_svm_decision_space(
    #         X_train, 
    #         y_train, 
    #         train_metadata=train_metadata,
    #         X_candidates=candidate_feats,
    #         candidate_boxes=boxes_abs,
    #         candidate_image=image_source
    #     )
        
    #     debugger.plot_confidence_distribution(verifier_model, X_train, y_train, candidate_feats)

    # boxes_xyxy, masks = engine.generate_masks(
    #     image_source, filtered_boxes, logits=svm_scores, debug=False
    # )
    
    # if boxes_xyxy is not None:
    #     engine.save_result(
    #         image_source, boxes_xyxy, masks, svm_scores, 
    #         file_name="calibrated_result.jpg", use_padding=True, show_boxes=True
    #     )

if __name__ == "__main__":
    main()