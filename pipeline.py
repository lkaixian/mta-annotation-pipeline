"""
MTA Pipeline — Core Orchestration
===================================
Ties together SAM2 segmentation, MTA classification, active learning,
Roboflow upload, and X-AnyLabeling export into a unified pipeline.

Usage:
    from pipeline import MTAPipeline
    pipe = MTAPipeline()
    results = pipe.process_image("path/to/image.jpg")
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional

from config import (
    SAM2_MODEL, MTA_MODEL_PATH, CLASS_NAMES, CLASS_COLORS,
    AUTO_APPROVE_THRESHOLD, DEVICE
)


class MTAPipeline:
    """
    End-to-end annotation pipeline:
      1. SAM2 generates polygon masks
      2. MTA YOLO classifies each polygon
      3. Active learning records predictions
      4. Human reviews via UI
      5. Approved annotations → Roboflow + X-AnyLabeling export
    """

    def __init__(
        self,
        sam2_model: Optional[str] = None,
        mta_model: Optional[str] = None,
        use_sam2: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            sam2_model: SAM2 checkpoint name. Defaults to config.
            mta_model: Path to MTA YOLO model. Defaults to config.
            use_sam2: If True, use SAM2 for polygon generation.
                      If False, use MTA model's own segmentation masks.
        """
        self.use_sam2 = use_sam2
        self._sam2 = None
        self._mta = None
        self._mta_attention = None
        self._al = None
        self._roboflow = None
        self._exporter = None
        self._sam2_model_name = sam2_model or SAM2_MODEL
        self._mta_model_path = mta_model or str(MTA_MODEL_PATH)

    @property
    def sam2(self):
        """Lazy-load SAM2 segmentor."""
        if self._sam2 is None:
            from sam2_segmentor import SAM2Segmentor
            self._sam2 = SAM2Segmentor(self._sam2_model_name)
        return self._sam2

    @property
    def mta(self):
        """Lazy-load MTA classifier."""
        if self._mta is None:
            from mta_classifier import MTAClassifier
            self._mta = MTAClassifier(self._mta_model_path)
        return self._mta

    @property
    def mta_attention(self):
        """Lazy-load MTA attention model."""
        if self._mta_attention is None:
            from mta_classifier import MTAClassifier
            from config import MTA_ATTENTION_PATH
            self._mta_attention = MTAClassifier(str(MTA_ATTENTION_PATH))
        return self._mta_attention

    @property
    def active_learning(self):
        """Lazy-load active learning manager."""
        if self._al is None:
            from active_learning import ActiveLearningManager
            self._al = ActiveLearningManager()
        return self._al

    @property
    def roboflow(self):
        """Lazy-load Roboflow uploader."""
        if self._roboflow is None:
            from roboflow_uploader import RoboflowUploader
            self._roboflow = RoboflowUploader()
        return self._roboflow

    @property
    def exporter(self):
        """Lazy-load X-AnyLabeling exporter."""
        if self._exporter is None:
            from xanylabeling_export import XAnyLabelingExporter
            self._exporter = XAnyLabelingExporter()
        return self._exporter

    def process_image(
        self,
        image_path: str,
        use_sam2_override: Optional[bool] = None,
        record_predictions: bool = True,
    ) -> Dict:
        """
        Process a single image through the full pipeline.

        Steps:
          1. Load image
          2. Generate polygons (SAM2 or MTA model masks)
          3. Classify each polygon with MTA model
          4. Record predictions in active learning DB
          5. Return structured results for UI review

        Args:
            image_path: Path to the input image.
            use_sam2_override: Override the default SAM2 setting.
            record_predictions: If True, record predictions in the DB.

        Returns:
            Dict with:
              - 'image_path': source image path
              - 'image': loaded image (BGR numpy array)
              - 'segments': list of segment dicts with polygons + classifications
              - 'prediction_ids': list of DB prediction IDs
              - 'annotated_image': image with drawn polygons
              - 'processing_time': time taken to process the frame in seconds
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        return self.process_frame(
            image, str(image_path),
            use_sam2_override=use_sam2_override,
            record_predictions=record_predictions,
        )

    def process_frame(
        self,
        image: np.ndarray,
        image_path: str = "frame",
        use_sam2_override: Optional[bool] = None,
        pipeline_mode: str = "Attention First",
        record_predictions: bool = True,
    ) -> Dict:
        """
        Process a single frame (image array) through the pipeline.

        Args:
            image: Input image (BGR numpy array).
            image_path: Path/name for tracking purposes.
            use_sam2_override: Override default SAM2 usage.
            record_predictions: Record predictions in DB.

        Returns:
            Pipeline result dict.
        """
        start_time = time.time()
        h, w = image.shape[:2]
        use_sam2 = use_sam2_override if use_sam2_override is not None else self.use_sam2

        # Step 1: Generate polygon segments
        if use_sam2:
            segments = self._generate_sam2_polygons(image, pipeline_mode=pipeline_mode)
        else:
            segments = self._generate_mta_polygons(image)

        # Step 2: Classify each polygon with MTA model
        polygons = [seg["polygon"] for seg in segments]
        if polygons:
            classifications = self.mta.classify_all(image, polygons)
        else:
            classifications = []

        # Step 3: Merge segments with classifications and filter unmatched
        merged = []
        prediction_ids = []
        discarded = 0

        for i, (seg, cls) in enumerate(zip(segments, classifications)):
            # Skip segments where MTA found no matching detection
            # and the fallback prediction is just "trash" (often background/sky/ground),
            # UNLESS the attention model detected it (attention_confidence > 0).
            is_unmatched_trash = (
                not cls.get("matched", True) and 
                cls.get("predicted_class") == "trash" and 
                cls.get("confidence", 0.0) <= 0.0 and
                seg.get("attention_confidence", 0.0) <= 0.0
            )
            if is_unmatched_trash:
                discarded += 1
                continue

            item = {
                **seg,
                **cls,
                "attention_confidence": seg.get("attention_confidence", 0.0),
                "color": CLASS_COLORS[cls["predicted_class_id"] % len(CLASS_COLORS)]
                         if cls["predicted_class_id"] >= 0 else (128, 128, 128),
            }
            merged.append(item)

            # Step 4: Record prediction in active learning DB
            if record_predictions:
                pred_id = self.active_learning.record_prediction(
                    image_path=image_path,
                    segment_id=i,
                    polygon=seg["polygon"],
                    predicted_class=cls["predicted_class"],
                    predicted_class_id=cls["predicted_class_id"],
                    confidence=cls["confidence"],
                    margin=cls.get("margin", 0.0),
                )
                item["prediction_id"] = pred_id
                prediction_ids.append(pred_id)

        if discarded > 0:
            print(f"[Pipeline] Filtered out {discarded} unmatched segments "
                  f"(no MTA detection). Keeping {len(merged)}.")

        # Step 5: Create annotated visualization
        annotated = self._draw_annotations(image, merged)

        processing_time = time.time() - start_time
        return {
            "image_path": str(image_path),
            "image": image,
            "image_height": h,
            "image_width": w,
            "segments": merged,
            "prediction_ids": prediction_ids,
            "annotated_image": annotated,
            "processing_time": processing_time,
        }

    def _generate_sam2_polygons(self, image: np.ndarray, pipeline_mode: str = "Attention First") -> List[Dict]:
        """Generate polygon segments using SAM2 prompted by MTA bounding boxes."""
        from mta_classifier import MTAClassifier

        from config import MTA_CONFIDENCE_THRESHOLD
        
        if pipeline_mode == "Attention First":
            print("[Pipeline] Running MTA Attention to get bounding box prompts...")
            # Run fast inference to find candidate regions
            detections = self.mta_attention.run_full_inference(image, conf=MTA_CONFIDENCE_THRESHOLD)
            
        elif pipeline_mode == "Classifier Only":
            print("[Pipeline] Running MTA Classifier to get bounding box prompts...")
            detections = self.mta.run_full_inference(image, conf=MTA_CONFIDENCE_THRESHOLD)
            
        elif pipeline_mode == "Ensemble (U then N)":
            from config import IOU_THRESHOLD_ENSEMBLE
            print("[Pipeline] Running Ensemble mode: MTA Classifier + Attention...")
            mta_dets = self.mta.run_full_inference(image, conf=MTA_CONFIDENCE_THRESHOLD)
            attn_dets = self.mta_attention.run_full_inference(image, conf=MTA_CONFIDENCE_THRESHOLD)
            
            # Combine logic: keep all MTA detections, only keep attention detections that don't overlap
            detections = list(mta_dets)
            for ad in attn_dets:
                overlap = False
                for md in mta_dets:
                    iou = MTAClassifier._compute_iou(ad["xyxy"], md["xyxy"])
                    if iou > IOU_THRESHOLD_ENSEMBLE:
                        overlap = True
                        break
                if not overlap:
                    detections.append(ad)
            
            print(f"[Pipeline] Ensemble combined: {len(mta_dets)} MTA + {len(detections) - len(mta_dets)} non-overlapping Attention detections.")
            
        elif pipeline_mode == "Boolean Mask Fusion / Consensus Segmentation":
            from config import IOU_THRESHOLD_INTERSECT, MIN_INTERSECT_AREA, IOU_THRESHOLD_ENSEMBLE
            print("[Pipeline] Running Boolean Mask Fusion / Consensus Segmentation mode...")
            mta_dets = self.mta.run_full_inference(image, conf=MTA_CONFIDENCE_THRESHOLD)
            attn_dets = self.mta_attention.run_full_inference(image, conf=MTA_CONFIDENCE_THRESHOLD)
            
            mta_bboxes = [d["xyxy"] for d in mta_dets]
            attn_bboxes = [d["xyxy"] for d in attn_dets]
            
            mta_segments = self.sam2.segment_with_bbox_prompts(image, mta_bboxes) if mta_bboxes else []
            attn_segments = self.sam2.segment_with_bbox_prompts(image, attn_bboxes) if attn_bboxes else []

            # Retrieve attention confidence for attn_segments
            for seg in attn_segments:
                best_iou = 0
                best_conf = 0.0
                for det in attn_dets:
                    iou = MTAClassifier._compute_iou(seg["bbox"], det["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_conf = det["confidence"]
                seg["attention_confidence"] = best_conf if best_iou > IOU_THRESHOLD_ENSEMBLE else 0.0

            from sam2_segmentor import SAM2Segmentor
            final_segments = []
            mta_used = [False] * len(mta_segments)
            attn_used = [False] * len(attn_segments)

            for a_idx, a_seg in enumerate(attn_segments):
                for m_idx, m_seg in enumerate(mta_segments):
                    iou = MTAClassifier._compute_iou(a_seg["bbox"], m_seg["bbox"])
                    if iou > IOU_THRESHOLD_INTERSECT: # Significant overlap
                        intersect_mask = cv2.bitwise_and(a_seg["mask"], m_seg["mask"])
                        area = int(np.sum(intersect_mask))
                        if area < MIN_INTERSECT_AREA:
                            continue
                            
                        new_polygon = SAM2Segmentor.mask_to_polygon(intersect_mask)
                        if new_polygon is None or len(new_polygon) < 3:
                            continue
                            
                        ys, xs = np.where(intersect_mask > 0)
                        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

                        merged_seg = {
                            "mask": intersect_mask,
                            "polygon": new_polygon,
                            "bbox": bbox,
                            "area": area,
                            "confidence": max(m_seg.get("confidence", 0), a_seg.get("confidence", 0)),
                            "attention_confidence": a_seg.get("attention_confidence", 0.0),
                            "segment_id": len(final_segments),
                        }
                        final_segments.append(merged_seg)
                        mta_used[m_idx] = True
                        attn_used[a_idx] = True

            # Add unused MTA segments
            for m_idx, m_seg in enumerate(mta_segments):
                if not mta_used[m_idx]:
                    final_segments.append(m_seg)
                    
            # Add unused Attention segments
            for a_idx, a_seg in enumerate(attn_segments):
                if not attn_used[a_idx]:
                    final_segments.append(a_seg)
            
            return final_segments
            
        else:
            detections = self.mta_attention.run_full_inference(image, conf=MTA_CONFIDENCE_THRESHOLD)
            
        bboxes = [d["xyxy"] for d in detections]
        
        if not bboxes:
            print("[Pipeline] No objects detected. Skipping SAM2.")
            return []

        print(f"[Pipeline] Running SAM2 with {len(bboxes)} bounding box prompts...")
        segments = self.sam2.segment_with_bbox_prompts(image, bboxes)
        print(f"[Pipeline] SAM2 returned {len(segments)} refined segments")

        # Match segments back to detections by IoU of bboxes to retrieve attention confidence
        for seg in segments:
            seg_bbox = seg["bbox"]
            best_iou = 0
            best_conf = 0.0
            for det in detections:
                iou = MTAClassifier._compute_iou(seg_bbox, det["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_conf = det["confidence"]
            # Fallback for dynamic mode matching where config wasn't imported
            try:
                from config import IOU_THRESHOLD_ENSEMBLE
                thresh = IOU_THRESHOLD_ENSEMBLE
            except ImportError:
                thresh = 0.3
            seg["attention_confidence"] = best_conf if best_iou > thresh else 0.0

        return segments

    def preview_segment_by_points(
        self, image: np.ndarray, points: List[List[int]], labels: List[int],
        bbox: Optional[List[int]] = None
    ) -> Optional[Dict]:
        """Generate a single interactive SAM2 polygon preview."""
        pts = np.array([points], dtype=np.float32) if points else None
        lbls = np.array([labels], dtype=np.int32) if labels else None
        bboxes = np.array([bbox], dtype=np.float32) if bbox else None
        
        segments = self.sam2.segment_image(image, points=pts, labels=lbls, bboxes=bboxes)
        if not segments:
            return None
            
        # Return the most confident segment
        return segments[0]

    def commit_preview_segment(self, image: np.ndarray, image_path: str, seg: Dict) -> Dict:
        """
        Classifies and commits a finalized interactive segment to the DB.
        """
        polygon = seg["polygon"]

        # Classify with MTA
        classifications = self.mta.classify_all(image, [polygon])
        cls = classifications[0] if classifications else {
            "predicted_class": "trash",
            "predicted_class_id": 0,
            "confidence": 0.0,
        }

        from config import CLASS_COLORS
        item = {
            **seg,
            **cls,
            "color": CLASS_COLORS[cls.get("predicted_class_id", 0) % len(CLASS_COLORS)],
        }

        # Record prediction
        pred_id = self.active_learning.record_prediction(
            image_path=image_path,
            segment_id=999,
            polygon=polygon,
            predicted_class=item.get("predicted_class", "trash"),
            predicted_class_id=item.get("predicted_class_id", 0),
            confidence=item.get("confidence", 0.0),
            margin=item.get("margin", 0.0),
        )
        item["prediction_id"] = pred_id

        return item

    def _generate_mta_polygons(self, image: np.ndarray) -> List[Dict]:
        """
        Generate polygon segments using MTA model's own masks.
        Fallback when SAM2 is not available.
        """
        print("[Pipeline] Using MTA model masks for polygons...")
        detections = self.mta.run_full_inference(image, conf=0.15)

        segments = []
        for i, det in enumerate(detections):
            if det.get("polygon") and det.get("mask") is not None:
                ys, xs = np.where(det["mask"] > 0)
                if len(ys) == 0:
                    continue
                segments.append({
                    "mask": det["mask"],
                    "polygon": det["polygon"],
                    "bbox": det["xyxy"],
                    "area": int(np.sum(det["mask"])),
                    "confidence": det["confidence"],
                    "segment_id": i,
                })

        print(f"[Pipeline] MTA model found {len(segments)} segments")
        return segments

    def _draw_annotations(
        self, image: np.ndarray, segments: List[Dict]
    ) -> np.ndarray:
        """Draw polygon annotations on the image."""
        annotated = image.copy()

        for seg in segments:
            polygon = seg.get("polygon", [])
            color = seg.get("color", (128, 128, 128))
            class_name = seg.get("predicted_class", "?")
            confidence = seg.get("confidence", 0.0)

            if not polygon:
                continue

            pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)

            # Draw filled polygon with transparency
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

            # Draw polygon outline
            cv2.polylines(annotated, [pts], True, color, 2)

            # Draw label
            bbox = seg.get("bbox", [0, 0, 0, 0])
            attn_conf = seg.get("attention_confidence", None)
            if attn_conf is not None and attn_conf > 0:
                label = f"{class_name} {confidence:.0%} (Attn: {attn_conf:.0%})"
            else:
                label = f"{class_name} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_x = bbox[0]
            label_y = bbox[1]
            cv2.rectangle(
                annotated,
                (label_x, label_y - th - 6),
                (label_x + tw + 6, label_y),
                color, -1
            )
            cv2.putText(
                annotated, label,
                (label_x + 3, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )

        return annotated

    def approve_segment(
        self,
        prediction_id: int,
        polygon_correct: bool = True,
        class_correct: bool = True,
        corrected_class: Optional[str] = None,
        corrected_class_id: Optional[int] = None,
    ) -> Dict:
        """
        Approve or reject a segment's polygon and/or class prediction.

        Args:
            prediction_id: DB prediction ID.
            polygon_correct: Whether the polygon shape is acceptable.
            class_correct: Whether the class prediction is correct.
            corrected_class: Corrected class name (if class_correct=False).
            corrected_class_id: Corrected class ID.

        Returns:
            Dict with reward values.
        """
        poly_reward = self.active_learning.record_polygon_feedback(
            prediction_id, polygon_correct
        )

        cls_reward = self.active_learning.record_class_feedback(
            prediction_id, class_correct,
            corrected_class=corrected_class,
            corrected_class_id=corrected_class_id,
        )

        total = poly_reward + cls_reward
        return {
            "polygon_reward": poly_reward,
            "class_reward": cls_reward,
            "total_reward": total,
            "should_retrain": self.active_learning.should_retrain(),
        }

    def auto_approve_high_confidence(
        self, segments: List[Dict], threshold: float = AUTO_APPROVE_THRESHOLD
    ) -> Dict:
        """
        Auto-approve segments with confidence above threshold.

        Args:
            segments: List of segment dicts from process_image.
            threshold: Confidence threshold for auto-approval.

        Returns:
            Dict with counts of auto-approved and pending items.
        """
        auto_approved = 0
        pending = 0

        for seg in segments:
            if seg.get("confidence", 0) >= threshold and "prediction_id" in seg:
                self.approve_segment(
                    prediction_id=seg["prediction_id"],
                    polygon_correct=True,
                    class_correct=True,
                )
                auto_approved += 1
            else:
                pending += 1

        return {"auto_approved": auto_approved, "pending": pending}

    def export_approved(
        self,
        image_path: str,
        segments: List[Dict],
        to_roboflow: bool = True,
        to_xanylabeling: bool = True,
    ) -> Dict:
        """
        Export approved annotations to Roboflow and/or X-AnyLabeling.

        Args:
            image_path: Source image path.
            segments: Approved segment dicts.
            to_roboflow: Upload to Roboflow.
            to_xanylabeling: Export X-AnyLabeling JSON.

        Returns:
            Dict with export results.
        """
        # Build export data
        polygons_with_classes = []
        for seg in segments:
            polygons_with_classes.append({
                "polygon": seg.get("polygon", []),
                "class_name": seg.get("predicted_class", "unknown"),
                "class_id": seg.get("predicted_class_id", -1),
                "confidence": seg.get("confidence", 0.0),
            })

        results = {}

        # Export to X-AnyLabeling
        if to_xanylabeling:
            img = cv2.imread(image_path) if isinstance(image_path, str) else None
            h = segments[0].get("image_height", 640) if segments else 640
            w = segments[0].get("image_width", 640) if segments else 640
            if img is not None:
                h, w = img.shape[:2]

            json_path = self.exporter.export_annotation(
                image_path=image_path,
                polygons_with_classes=polygons_with_classes,
                image_height=h,
                image_width=w,
            )
            results["xanylabeling_path"] = json_path

        # Upload to Roboflow
        if to_roboflow:
            from roboflow_uploader import RoboflowUploader
            if RoboflowUploader.is_configured():
                success = self.roboflow.upload_annotation(
                    image_path=image_path,
                    polygons_with_classes=polygons_with_classes,
                    is_prediction=False,
                )
                results["roboflow_uploaded"] = success
            else:
                results["roboflow_uploaded"] = False
                results["roboflow_error"] = "Not configured"

        return results

    def process_batch(
        self,
        image_dir: str,
        extensions: List[str] = None,
        auto_approve: bool = False,
    ) -> List[Dict]:
        """
        Process all images in a directory.

        Args:
            image_dir: Directory containing images.
            extensions: File extensions to process.
            auto_approve: Auto-approve high-confidence predictions.

        Returns:
            List of pipeline result dicts.
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in extensions
        ]

        print(f"[Pipeline] Processing {len(image_files)} images from {image_dir}...")
        results = []
        batch_start_time = time.time()

        for i, img_path in enumerate(image_files):
            print(f"[Pipeline] [{i+1}/{len(image_files)}] {img_path.name}")
            try:
                result = self.process_image(str(img_path))
                results.append(result)

                if auto_approve:
                    self.auto_approve_high_confidence(result["segments"])

            except Exception as e:
                print(f"[Pipeline] Error processing {img_path.name}: {e}")
                results.append({
                    "image_path": str(img_path),
                    "error": str(e),
                    "segments": [],
                })

        batch_end_time = time.time()
        total_time = batch_end_time - batch_start_time
        avg_time = total_time / max(1, len(results))
        print(f"[Pipeline] Batch complete: {len(results)} images processed in {total_time:.2f}s (avg {avg_time:.2f}s/image)")
        return results
