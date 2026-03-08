import cv2
import numpy as np
import threading
import queue
import time
from core_engine import AnnotateEngine
import torch

# TODO: Improve the edit tracking system from basic incrementation to actual change comparisons.
class InteractiveAnnotator:
    """
    Manages the OpenCV interactions for a single image.
    Tracks edits to calculate volatility.
    """
    def __init__(self, image, boxes, logits, window_name, conf_threshold=0.3, pre_labels=None):
        self.image = image.copy()
        self.vis_image = image.copy()
        self.h, self.w, _ = image.shape
        
        if len(boxes) > 0:
            self.boxes = self._denormalize(boxes)
            self.scores = logits.numpy() if hasattr(logits, 'numpy') else np.array(logits)
        else:
            self.boxes = np.empty((0, 4))
            self.scores = np.array([])

        if pre_labels is not None:
            self.labels = pre_labels
        elif len(self.scores) > 0:
            self.labels = (self.scores >= conf_threshold).astype(int)
        else:
            self.labels = np.array([], dtype=int)
        
        self.edit_count = 0
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.window_name = window_name

    def _denormalize(self, boxes_norm):
        if len(boxes_norm) == 0: return np.empty((0, 4))
        b = boxes_norm.clone().numpy() if hasattr(boxes_norm, 'clone') else boxes_norm.copy()
        b[:, 0] = (boxes_norm[:, 0] - boxes_norm[:, 2]/2) * self.w
        b[:, 1] = (boxes_norm[:, 1] - boxes_norm[:, 3]/2) * self.h
        b[:, 2] = (boxes_norm[:, 0] + boxes_norm[:, 2]/2) * self.w
        b[:, 3] = (boxes_norm[:, 1] + boxes_norm[:, 3]/2) * self.h
        return b

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self._toggle_closest_box(x, y)
            self._redraw()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._delete_closest(x, y)
            self._redraw()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            temp = self.vis_image.copy()
            cv2.rectangle(temp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self._add_box(self.ix, self.iy, x, y)
            self._redraw()

    def _get_closest_idx(self, px, py, threshold=50):
        if len(self.boxes) == 0: return None
        centers_x = (self.boxes[:, 0] + self.boxes[:, 2]) / 2
        centers_y = (self.boxes[:, 1] + self.boxes[:, 3]) / 2
        dists = np.sqrt((centers_x - px)**2 + (centers_y - py)**2)
        idx = np.argmin(dists)
        return idx if dists[idx] < threshold else None

    def _toggle_closest_box(self, px, py):
        idx = self._get_closest_idx(px, py)
        if idx is not None: 
            self.labels[idx] = 1 - self.labels[idx]
            self.edit_count += 1

    def _delete_closest(self, px, py):
        idx = self._get_closest_idx(px, py)
        if idx is not None:
            self.boxes = np.delete(self.boxes, idx, axis=0)
            self.labels = np.delete(self.labels, idx, axis=0)
            self.scores = np.delete(self.scores, idx, axis=0)
            self.edit_count += 1

    def _add_box(self, x1, y1, x2, y2):
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        if (x_max - x_min) * (y_max - y_min) < 50: return
        self.edit_count += 1    # Only increment the edit count if the box is actually added.
        new_box = np.array([x_min, y_min, x_max, y_max])
        self.boxes = np.vstack([self.boxes, new_box]) if len(self.boxes) > 0 else np.array([new_box])
        self.labels = np.append(self.labels, 1) # Default Pos
        self.scores = np.append(self.scores, 1.0) # Manual confidence

    def _redraw(self):
        self.vis_image = self.image.copy()
        
        header = f"Edits: {self.edit_count} | Drag: Add | Right-Click: Flip | Space: Done"
        cv2.rectangle(self.vis_image, (0, 0), (self.w, 40), (0, 0, 0), -1)
        cv2.putText(self.vis_image, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 255, 0) if self.labels[i] == 1 else (0, 0, 255)
            cv2.rectangle(self.vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.vis_image, f"{self.scores[i]:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        cv2.imshow(self.window_name, cv2.cvtColor(self.vis_image, cv2.COLOR_RGB2BGR))

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self._redraw()
        
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 32: break # SPACE
            if k == 27: return None, None, -1 # ESC
        
        pos_mask = self.labels == 1
        neg_mask = self.labels == 0
        
        return self._normalize(self.boxes[pos_mask]), self._normalize(self.boxes[neg_mask]), self.edit_count

    def _normalize(self, boxes_abs):
        if len(boxes_abs) == 0: return np.empty((0, 4))
        w = boxes_abs[:, 2] - boxes_abs[:, 0]
        h = boxes_abs[:, 3] - boxes_abs[:, 1]
        cx = boxes_abs[:, 0] + w/2
        cy = boxes_abs[:, 1] + h/2
        return np.stack([cx/self.w, cy/self.h, w/self.w, h/self.h], axis=1)

# TODO: The multi-step verifier training is broken. If we make the embedder run on cpu, it takes too long and if the embedder waits for gpu lock, its still quite slow. 
class ActiveCalibrationSession:
    """
    Orchestrates the Active Learning Loop.
    - Thread 1 (Producer): Pre-fetches Stage 1 detections.
    - Thread 2 (Consumer): HITL annotation + JIT Filtering + Batch Training.
    """
    def __init__(self, engine, image_paths, prompt, initial_batch_size=5, model_save_path="weights/judiciary.pkl"):
        self.engine = engine
        self.image_paths = image_paths
        self.prompt = prompt
        self.batch_size = initial_batch_size
        self.save_path = model_save_path
        
        self.queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.pause_producer_event = threading.Event() # Used during training
        self.gpu_lock = threading.Lock()
        
        self.verified_data = [] 
        self.current_model = None 
        
        self.window_name = "Calibration Window"
        self.window_size = (1024, 768)

    def _producer(self):
        print(f"--> [Background] Producer starting on {len(self.image_paths)} images...")
        
        for path in self.image_paths:
            if self.stop_event.is_set(): break
            
            # Pause check: If Consumer is training, we should pause to release VRAM/locks
            while self.pause_producer_event.is_set():
                time.sleep(0.1)

            with self.gpu_lock:
                try:
                    boxes, logits, _, image = self.engine.detect_objects(path, self.prompt, box_threshold=0.05, batch_size=4)
                except Exception as e:
                    print(f"\n[Producer Error] Failed on {path}: {e}")
                    continue

            package = {
                "path": path,
                "image": image,
                "boxes": boxes, # Raw Candidates
                "logits": logits
            }
            
            self.queue.put(package)
        
        self.queue.put(None) # Sentinel
        print("--> [Background] Producer finished.")

    def start(self):
        t = threading.Thread(target=self._producer, daemon=True)
        t.start()
        
        print(f"--> [UI] Session started. Batch Size: {self.batch_size}")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])

        batch_counter = 0
        batch_edits = 0
        images_processed_in_batch = 0
        
        try:
            while True:
                try:
                    self._show_waiting_screen()
                    data = self.queue.get(timeout=0.2) 
                except queue.Empty:
                    if not t.is_alive() and self.queue.empty():
                        break
                    cv2.waitKey(50)
                    continue

                if data is None: break # Sentinel received

                image = data['image']
                boxes = data['boxes']
                logits = data['logits']
                
                if self.current_model is not None and len(boxes) > 0:
                    print("[HITL] Waiting for acquiring GPU lock")
                    with self.gpu_lock:
                        print("[HITL] GPU lock acquired.")
                        keep_boxes, keep_logits, reject_boxes, reject_logits = self.engine.filter_candidates(
                            image, boxes, self.current_model, confidence_threshold=0.25, force_cpu=False
                        )
                    all_boxes = torch.cat([keep_boxes, reject_boxes]) if len(reject_boxes) > 0 else keep_boxes
                    all_logits = torch.cat([keep_logits, reject_logits]) if len(reject_boxes) > 0 else keep_logits
                    
                    pre_labels = np.concatenate([
                        np.ones(len(keep_boxes), dtype=int),
                        np.zeros(len(reject_boxes), dtype=int)
                    ]) if len(reject_boxes) > 0 else np.ones(len(keep_boxes), dtype=int)
                    
                    annotator = InteractiveAnnotator(image, all_boxes, all_logits, self.window_name, 
                                                    pre_labels=pre_labels)
                else:
                    # No model yet, use confidence threshold for initial labeling
                    annotator = InteractiveAnnotator(image, boxes, logits, self.window_name)
                
                pos_boxes, neg_boxes, edits = annotator.run()
                
                if pos_boxes is None: # ESC pressed
                    self.stop_event.set()
                    break

                if isinstance(pos_boxes, torch.Tensor):
                    pos_boxes = pos_boxes.detach().cpu()
                if isinstance(neg_boxes, torch.Tensor):
                    neg_boxes = neg_boxes.detach().cpu()
                
                self.verified_data.append({
                    "path": data['path'],
                    "pos": pos_boxes,
                    "neg": neg_boxes
                })
                
                batch_edits += edits
                images_processed_in_batch += 1
                is_last_batch = (images_processed_in_batch == len(self.image_paths) - batch_counter * self.batch_size)

                if (images_processed_in_batch >= self.batch_size) or (images_processed_in_batch == len(self.image_paths) - batch_counter * self.batch_size): 
                    batch_counter += 1
                    avg_volatility = batch_edits / self.batch_size
                    is_volatility_satisfactory = avg_volatility < 0.5 and len(self.verified_data) >= 15
                    print(f"\n[Batch {batch_counter}] Completed. Volatility: {avg_volatility:.2f}")

                    self.pause_producer_event.set()
                    
                    print(f"--> [Training] Retraining verifier on {len(self.verified_data)} images...")
                    self._show_waiting_screen()
                    print("[HITL] Waiting for acquiring GPU lock")
                    with self.gpu_lock:
                        print("[HITL] GPU lock acquired.")
                        self.current_model, _, _, _ = self.engine.train_verifier(
                            self.verified_data, save_path=self.save_path, 
                            fast_train = not (is_last_batch or is_volatility_satisfactory)
                        )
                    
                    self.pause_producer_event.clear()
                    
                    batch_edits = 0
                    images_processed_in_batch = 0
                    
                    if is_volatility_satisfactory:
                        print("--> [Stopping] Volatility is low. Model is stable.")
                        self.stop_event.set()
                        break

        except KeyboardInterrupt:
            self.stop_event.set()
            print("\n[System] Interrupted.")
        
        cv2.destroyAllWindows()
        t.join(timeout=2.0)
        
        return self.verified_data

    def _show_waiting_screen(self):
        blank = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        text = "Waiting for the image to arrive..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        text_x = (1024 - text_size[0]) // 2
        text_y = (768 + text_size[1]) // 2
        cv2.putText(blank, text, (text_x, text_y), font, 1.0, (200, 200, 200), 2)        
        cv2.imshow(self.window_name, blank)
        cv2.waitKey(1)