import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import cv2
from PIL import Image

class ImageProcessor:
    @staticmethod
    def load_image(image_path, target_size=1280):
        """
        Loads an image and standardizes it to a fixed size with padding.
        Returns:
            image_source (np.array): HxWx3 array (RGB) for model processing
            pil_image (PIL.Image): Original PIL object (padded)
            scale_info (dict): Metadata about original size and padding
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_image = Image.open(image_path).convert("RGB")
        w, h = pil_image.size
        target_w, target_h = (target_size, target_size)

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_image = pil_image.resize((new_w, new_h), resample=Image.LANCZOS)

        final_image = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        final_image.paste(resized_image, (paste_x, paste_y))

        image_source = np.asarray(final_image)
        image_source = np.ascontiguousarray(image_source)   # Critical for SAM2

        return image_source, final_image

class VisualDebugger:
    def __init__(self):
        self.pca = PCA(n_components=2)
        
    def _load_and_crop(self, img_path, box_xyxy):
        """Helper to load an image from disk and crop it safely."""
        if not os.path.exists(img_path): return None, "File Not Found"
        # Read image (Opencv reads as BGR)
        img, _ = ImageProcessor.load_image(img_path)
        
        h, w, _ = img.shape
        x1, y1, x2, y2 = box_xyxy
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return None, "Empty Crop"
        return crop, f"{os.path.basename(img_path)}\nBox: [{x1},{y1},{x2},{y2}]"

    def plot_svm_decision_space(self, X_train, y_train, train_metadata=None, 
                                X_candidates=None, candidate_boxes=None, candidate_image=None):
        """
        Fully Interactive Plot. Click ANY point (Green, Red, or Blue) to see its source crop.
        """
        print("--> [Debug] Generating Interactive Feature Space Plot...")
        # scaler = StandardScaler()
        
        # X_train = scaler.fit_transform(X_train)
        X_train_2d = self.pca.fit_transform(X_train)
        
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 3)
        ax_plot = fig.add_subplot(gs[0, 0:2])
        ax_viewer = fig.add_subplot(gs[0, 2])
        
        ax_viewer.set_title("Click any point to inspect source")
        ax_viewer.axis("off")

        pos_mask = y_train == 1
        neg_mask = y_train == 0
        
        neg_collection = ax_plot.scatter(X_train_2d[neg_mask, 0], X_train_2d[neg_mask, 1], 
                        c='red', alpha=0.3, label='Negatives (Train)', s=30, picker=5)
        pos_collection = ax_plot.scatter(X_train_2d[pos_mask, 0], X_train_2d[pos_mask, 1], 
                        c='green', alpha=0.6, label='Positives (Train)', s=50, edgecolors='black', picker=5)

        cand_collection = None
        if X_candidates is not None and len(X_candidates) > 0:
            # X_candidates = scaler.transform(X_candidates)
            X_cand_2d = self.pca.transform(X_candidates)
            cand_collection = ax_plot.scatter(X_cand_2d[:, 0], X_cand_2d[:, 1], 
                                              c='blue', marker='*', s=200, 
                                              label='Candidates (Target)', edgecolors='white', picker=5)

        ax_plot.set_title("SVM Feature Space [Interactive]")
        ax_plot.set_xlabel("PC1")
        ax_plot.set_ylabel("PC2")
        ax_plot.legend()
        ax_plot.grid(True, alpha=0.3)

        def on_pick(event):
            crop = None
            title = ""
            ind = event.ind[0]

            if event.artist == cand_collection:
                if candidate_boxes is not None and candidate_image is not None:
                    box = candidate_boxes[ind].astype(int)
                    h_img, w_img, _ = candidate_image.shape
                    x1, y1 = max(0, box[0]), max(0, box[1])
                    x2, y2 = min(w_img, box[2]), min(h_img, box[3])
                    crop = candidate_image[y1:y2, x1:x2]
                    title = f"Candidate #{ind} (Target Image)"

            elif train_metadata is not None and (event.artist == pos_collection or event.artist == neg_collection):
                if event.artist == pos_collection:
                    global_idx = np.where(pos_mask)[0][ind]
                    prefix = "POS"
                else:
                    global_idx = np.where(neg_mask)[0][ind]
                    prefix = "NEG"
                
                # print("[Debug] Selected Index:", global_idx)
                img_path, box_abs = train_metadata[global_idx]
                crop, meta_txt = self._load_and_crop(img_path, box_abs)
                title = f"{prefix} Train Point #{global_idx}\n{meta_txt}"

            if crop is not None and crop.size > 0:
                ax_viewer.clear()
                ax_viewer.imshow(crop)
                ax_viewer.set_title(title, fontsize=9)
                ax_viewer.axis("off")
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', on_pick)
        print("    [Interactive] Window open. Click ANY point to see crops.")
        plt.savefig("output/debug_feature_space.png")
        print("    [Saved] output/debug_feature_space.png")
        plt.show()

    def plot_confidence_distribution(self, clf, X_train, y_train, X_candidates=None):
        """
        Visualizes the probability scores. 
        Are the positives clustered near 1.0? Are the candidates stuck near 0.4?
        """
        print("--> [Debug] Generating Score Distribution Plot...")
        
        train_probs = clf.predict_proba(X_train)[:, 1]
        
        plt.figure(figsize=(10, 6))
        
        sns.histplot(train_probs[y_train==1], color='green', label='Anchor Positives', kde=True, bins=20, alpha=0.5)
        sns.histplot(train_probs[y_train==0], color='red', label='Anchor Negatives', kde=True, bins=20, alpha=0.5)
        
        if X_candidates is not None:
            cand_probs = clf.predict_proba(X_candidates)[:, 1]
            sns.histplot(cand_probs, color='blue', label='Target Candidates', kde=True, bins=20, alpha=0.6)
            
            plt.axvline(0.5, color='black', linestyle='--', label='Default Cutoff (0.5)')
            plt.axvline(cand_probs.mean(), color='blue', linestyle=':', label='Avg Candidate Score')

        plt.title("SVM Confidence Distribution")
        plt.xlabel("Pothole Probability (0.0 = Not Pothole, 1.0 = Definitely Pothole)")
        plt.ylabel("Count")
        plt.legend()
        
        plt.savefig("output/debug_score_dist.png")
        print("    [Saved] output/debug_score_dist.png")
        plt.show()