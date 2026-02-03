import time
import torch
import numpy as np
from ultralytics import YOLO
from transformers import Sam3Model, Sam3Processor
from PIL import Image

class PrecisionJSONPipeline:
    def __init__(self, yolo_path, sam_id="facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = YOLO(yolo_path)
        self.sam_processor = Sam3Processor.from_pretrained(sam_id)
        self.sam_model = Sam3Model.from_pretrained(sam_id).to(self.device)

    def _select_best(self, detections):
        if not detections: return None
        if len(detections) == 1: return detections[0] # Return full dict to keep conf
        filtered = [d for d in detections if d['conf'] >= 0.51]
        if not filtered: return None
        return sorted(filtered, key=lambda x: x['conf'], reverse=True)[0]

    def get_yolo_detections(self, image_path):
        results = self.yolo_model.predict(source=image_path, conf=0.25, verbose=False)[0]
        stamps, signatures = [], []
        if hasattr(results, 'obb') and results.obb is not None:
            for i in range(len(results.obb)):
                cls, conf = int(results.obb.cls[i]), float(results.obb.conf[i])
                corners = results.obb.xyxyxyxy[i].cpu().numpy()
                box = [np.min(corners[:,0]), np.min(corners[:,1]), np.max(corners[:,0]), np.max(corners[:,1])]
                det = {'box': np.array(box).astype(int), 'conf': conf}
                if cls == 1: stamps.append(det)
                elif cls == 0: signatures.append(det)
        return {'stamp': self._select_best(stamps), 'sig': self._select_best(signatures)}

    def run_sam_on_crop(self, full_img, buffer_box, text, threshold):
        x1, y1, x2, y2 = buffer_box
        h_orig, w_orig = full_img.shape[:2]
        crop = full_img[y1:y2, x1:x2].copy()
        if crop.size == 0: return None, 0.0

        inputs = self.sam_processor(images=crop, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        res = self.sam_processor.post_process_instance_segmentation(outputs, threshold=threshold, target_sizes=[crop.shape[:2]])[0]

        masks = res.get("masks", [])
        scores = res.get("scores", [])
        if len(masks) == 0: return None, 0.0

        best_idx = torch.argmax(scores).item()
        conf_score = scores[best_idx].item()
        best_mask = (masks[best_idx].cpu().numpy() > 0).astype(np.uint8)

        full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = best_mask
        return full_mask, conf_score

    def mask_to_bbox(self, mask):
        if mask is None: return None
        coords = np.column_stack(np.where(mask > 0))
        if coords.shape[0] > 0:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            return [int(xmin), int(ymin), int(xmax), int(ymax)]
        return None

    def process_to_json(self, path):
        # 1. Start Timing
        start_time = time.time()

        img = np.array(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]

        det = self.get_yolo_detections(path)
        is_from_yolo = det['stamp'] is not None

        # 2. Buffer Logic
        if is_from_yolo:
            bx1, by1, bx2, by2 = det['stamp']['box']
            bw, bh = bx2 - bx1, by2 - by1
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            buf = [max(0, int(cx - (bw*1.75)/2)), max(0, int(cy - (bh*1.75)/2)),
                   min(w, int(cx + (bw*1.75)/2)), min(h, int(cy + (bh*1.75)/2))]
        else:
            buf = [int(2*w/3), int(2*h/3), w, h]

        # 3. SAM Inference (Modified to return conf)
        s_mask, s_conf = self.run_sam_on_crop(img, buf, "stamp", 0.4)
        sig_mask, sig_conf = self.run_sam_on_crop(img, buf, "signature", 0.2)

        # 4. Final BBox Calculation & Recovery (with Confidence Logic)
        final_stamp_bbox = self.mask_to_bbox(s_mask)
        stamp_final_conf = s_conf
        
        if final_stamp_bbox is None and is_from_yolo:
            final_stamp_bbox = det['stamp']['box'].tolist()
            stamp_final_conf = det['stamp']['conf'] # Use YOLO conf if fallback triggered

        final_sig_bbox = self.mask_to_bbox(sig_mask)
        sig_final_conf = sig_conf

        # 5. Pipeline Confidence Calculation
        pipeline_confidence = round((0.5 * stamp_final_conf) + (0.5 * sig_final_conf), 4)

        # 6. Calculate Latency in Seconds
        end_time = time.time()
        latency_seconds = round(end_time - start_time, 3)

        output_data = {
            "signature": {
                "present": final_sig_bbox is not None,
                "bbox": final_sig_bbox if final_sig_bbox else []
            },
            "stamp": {
                "present": final_stamp_bbox is not None,
                "bbox": final_stamp_bbox if final_stamp_bbox else []
            },
            "confidence_score": pipeline_confidence,
            "inference_latency_seconds": latency_seconds
        }

        return output_data