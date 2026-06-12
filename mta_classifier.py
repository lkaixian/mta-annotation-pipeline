"""
MTA Classifier Module
=====================
Runs the Malaysian Trash Annotation (MTA) YOLO segmentation model
to classify each SAM2-generated polygon region.

Uses the trained YOLOv11 model to predict class names and confidence
scores for trash categories.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from config import (
    MTA_MODEL_PATH, CLASS_NAMES, CLASS_COLORS, DEVICE, IMGSZ,
    CONFIDENCE_THRESHOLD
)


class MTAClassifier:
    """
    MTA YOLO model wrapper for classifying segmented trash regions.

    Workflow:
      1. Receive polygon region (from SAM2)
      2. Run MTA YOLO model on the region
      3. Return class prediction with confidence scores
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the MTA classifier.

        Args:
            model_path: Path to the MTA YOLO .pt model. Defaults to config value.
        """
        from ultralytics import YOLO

        path = Path(model_path) if model_path else MTA_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"MTA model not found at {path}. "
                f"Available models in {path.parent}: "
                f"{[f.name for f in path.parent.glob('*.pt')]}"
            )

        print(f"[MTA] Loading model: {path.name} on {DEVICE}...")
        self.model = YOLO(str(path))
        self.model.to(DEVICE)
        self.model_path = path
        print(f"[MTA] Model loaded successfully.")

    def classify_region(
        self,
        image: np.ndarray,
        polygon: List[Tuple[int, int]],
        conf: float = 0.10,
    ) -> Dict:
        """
        Classify a polygon region of an image.

        Strategy: Run inference on the full image but only consider
        detections that overlap with the given polygon (by IoU).

        Args:
            image: Full image (BGR numpy array).
            polygon: List of (x, y) polygon points defining the region.
            conf: Minimum confidence threshold for inference.

        Returns:
            Dict with:
              - 'predicted_class': top predicted class name
              - 'predicted_class_id': class index
              - 'confidence': top class confidence
              - 'all_scores': list of {class_name, class_id, confidence} sorted desc
              - 'is_uncertain': True if confidence < CONFIDENCE_THRESHOLD
        """
        # Get bbox from polygon
        pts = np.array(polygon)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        target_bbox = [int(x1), int(y1), int(x2), int(y2)]

        # Run inference on full image at low confidence to get all candidates
        results = self.model.predict(
            image, conf=conf, imgsz=IMGSZ, verbose=False,
            retina_masks=True, device=DEVICE
        )

        # Match detections to target polygon by IoU
        class_scores = {}  # cls_id -> max confidence

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    det_xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    iou = self._compute_iou(target_bbox, det_xyxy.tolist())
                    if iou > 0.3:
                        cid = int(boxes.cls[i].item())
                        c = float(boxes.conf[i].item())
                        if cid not in class_scores or c > class_scores[cid]:
                            class_scores[cid] = c

        # Check if ANY MTA detection matched this polygon
        matched = len(class_scores) > 0

        # Build sorted scores for all classes
        all_scores = []
        for cid, name in enumerate(CLASS_NAMES):
            all_scores.append({
                "class_name": name,
                "class_id": cid,
                "confidence": class_scores.get(cid, 0.0),
            })
        all_scores.sort(key=lambda x: x["confidence"], reverse=True)

        # Top prediction
        top = all_scores[0] if all_scores else {
            "class_name": "unknown", "class_id": -1, "confidence": 0.0
        }

        # Check uncertainty
        is_uncertain = top["confidence"] < CONFIDENCE_THRESHOLD
        margin = 0.0
        if len(all_scores) >= 2:
            margin = all_scores[0]["confidence"] - all_scores[1]["confidence"]

        return {
            "predicted_class": top["class_name"],
            "predicted_class_id": top["class_id"],
            "confidence": top["confidence"],
            "all_scores": all_scores,
            "is_uncertain": is_uncertain,
            "matched": matched,
            "margin": margin,
            "bbox": target_bbox,
        }

    def classify_all(
        self,
        image: np.ndarray,
        polygons: List[List[Tuple[int, int]]],
    ) -> List[Dict]:
        """
        Classify all polygon regions in a single image.

        Runs inference once on the full image, then matches each polygon
        to the best overlapping detection.

        Args:
            image: Full image (BGR).
            polygons: List of polygon point lists.

        Returns:
            List of classification dicts (same format as classify_region).
        """
        # Run inference once on full image
        results = self.model.predict(
            image, conf=0.05, imgsz=IMGSZ, verbose=False,
            retina_masks=True, device=DEVICE
        )

        # Extract all detections
        all_detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    all_detections.append({
                        "xyxy": boxes.xyxy[i].cpu().numpy().astype(int).tolist(),
                        "cls_id": int(boxes.cls[i].item()),
                        "conf": float(boxes.conf[i].item()),
                    })

        # Match each polygon to detections
        classifications = []
        for polygon in polygons:
            pts = np.array(polygon)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            target_bbox = [int(x1), int(y1), int(x2), int(y2)]

            class_scores = {}
            for det in all_detections:
                iou = self._compute_iou(target_bbox, det["xyxy"])
                if iou > 0.3:
                    cid = det["cls_id"]
                    c = det["conf"]
                    if cid not in class_scores or c > class_scores[cid]:
                        class_scores[cid] = c

            # Check if ANY MTA detection matched this polygon
            matched = len(class_scores) > 0

            # Build all-class scores
            all_scores = []
            for cid, name in enumerate(CLASS_NAMES):
                all_scores.append({
                    "class_name": name,
                    "class_id": cid,
                    "confidence": class_scores.get(cid, 0.0),
                })
            all_scores.sort(key=lambda x: x["confidence"], reverse=True)

            top = all_scores[0]
            is_uncertain = top["confidence"] < CONFIDENCE_THRESHOLD
            margin = 0.0
            if len(all_scores) >= 2:
                margin = all_scores[0]["confidence"] - all_scores[1]["confidence"]

            classifications.append({
                "predicted_class": top["class_name"],
                "predicted_class_id": top["class_id"],
                "confidence": top["confidence"],
                "all_scores": all_scores,
                "is_uncertain": is_uncertain,
                "matched": matched,
                "margin": margin,
                "bbox": target_bbox,
            })

        return classifications

    def run_full_inference(
        self, image: np.ndarray, conf: float = 0.25
    ) -> List[Dict]:
        """
        Run standard MTA inference on the full image (no SAM2 polygons).
        Returns detections with bboxes, masks, and class predictions.

        Args:
            image: Input image (BGR).
            conf: Confidence threshold.

        Returns:
            List of detection dicts with xyxy, class, conf, mask, polygon.
        """
        results = self.model.predict(
            image, conf=conf, imgsz=IMGSZ, verbose=False,
            retina_masks=True, device=DEVICE
        )

        detections = []
        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            masks = r.masks

            if boxes is not None:
                for i in range(len(boxes)):
                    cid = int(boxes.cls[i].item())
                    confidence = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
                    color = CLASS_COLORS[cid % len(CLASS_COLORS)]

                    det = {
                        "class_id": cid,
                        "class_name": name,
                        "confidence": confidence,
                        "xyxy": xyxy,
                        "color": color,
                        "mask": None,
                        "polygon": None,
                    }

                    # Extract mask and polygon if available
                    if masks is not None and i < len(masks):
                        mask = masks.data[i].cpu().numpy()
                        h, w = image.shape[:2]
                        if mask.shape != (h, w):
                            mask = cv2.resize(
                                mask, (w, h), interpolation=cv2.INTER_NEAREST
                            )
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        det["mask"] = binary_mask

                        # Convert to polygon
                        from sam2_segmentor import SAM2Segmentor
                        polygon = SAM2Segmentor.mask_to_polygon(binary_mask)
                        det["polygon"] = polygon

                    detections.append(det)

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    @staticmethod
    def _compute_iou(box1: List[int], box2: List[int]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / max(union, 1e-6)
