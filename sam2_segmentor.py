"""
SAM2 Segmentor Module
=====================
Wraps Meta's Segment Anything Model 2 (SAM2) for polygon generation.
Uses Ultralytics integration for simplified API on Windows.

Converts binary masks → polygon coordinates for annotation.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from config import (
    SAM2_MODEL, DEVICE, SAM2_MIN_MASK_AREA, POLYGON_SIMPLIFY_EPSILON
)


class SAM2Segmentor:
    """
    SAM2-based image segmentor that generates polygon annotations.

    Pipeline:
      1. Load SAM2 model via Ultralytics
      2. Run "segment everything" on input image
      3. Convert binary masks → simplified polygon coordinates
    """

    def __init__(self, model_name: str = SAM2_MODEL):
        """
        Initialize SAM2 model.

        Args:
            model_name: SAM2 checkpoint name (e.g., 'sam2.1_l.pt').
                        Will be downloaded automatically on first use.
        """
        from ultralytics import SAM

        print(f"[SAM2] Loading model: {model_name} on {DEVICE}...")
        self.model = SAM(model_name)
        self.model_name = model_name
        print(f"[SAM2] Model loaded successfully.")

    def segment_image(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        bboxes: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Segment an image using SAM2.

        Args:
            image: Input image as BGR numpy array.
            points: Optional (N, 2) array of point prompts.
            labels: Optional (N,) array of point labels (1=foreground, 0=background).
            bboxes: Optional (M, 4) array of bounding box prompts [x1, y1, x2, y2].

        Returns:
            List of dicts, each with:
              - 'mask': binary mask (H, W) numpy array
              - 'polygon': list of (x, y) points
              - 'bbox': [x1, y1, x2, y2]
              - 'area': mask area in pixels
              - 'confidence': predicted IoU score
        """
        # Build prediction kwargs
        kwargs = {"verbose": False, "device": DEVICE}
        if bboxes is not None:
            kwargs["bboxes"] = bboxes
        if points is not None:
            kwargs["points"] = points
            if labels is not None:
                kwargs["labels"] = labels

        results = self.model(image, **kwargs)

        return self._process_results(results, image.shape[:2])

    def segment_everything(self, image: np.ndarray) -> List[Dict]:
        """
        Run SAM2 in 'everything' mode — automatically segments all objects.

        Args:
            image: Input image as BGR numpy array.

        Returns:
            List of segment dicts (same format as segment_image).
        """
        results = self.model(image, verbose=False, device=DEVICE)
        return self._process_results(results, image.shape[:2])

    def segment_with_bbox_prompts(
        self, image: np.ndarray, bboxes: List[List[int]]
    ) -> List[Dict]:
        """
        Segment specific regions using bounding box prompts.
        Useful when MTA YOLO provides bboxes and we want precise polygon masks.

        Args:
            image: Input image as BGR numpy array.
            bboxes: List of [x1, y1, x2, y2] bounding boxes.

        Returns:
            List of segment dicts.
        """
        if not bboxes:
            return []

        bbox_array = np.array(bboxes, dtype=np.float32)
        return self.segment_image(image, bboxes=bbox_array)

    def _process_results(
        self, results, image_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Process SAM2 results into structured segment data.

        Args:
            results: Ultralytics SAM2 results object.
            image_shape: (height, width) of original image.

        Returns:
            List of segment dicts with masks, polygons, bboxes, areas.
        """
        segments = []
        h, w = image_shape

        if not results or len(results) == 0:
            return segments

        r = results[0]
        masks = r.masks

        if masks is None or len(masks) == 0:
            return segments

        for i in range(len(masks)):
            # Extract binary mask
            mask_data = masks.data[i].cpu().numpy()
            # Resize to original image size if needed
            if mask_data.shape != (h, w):
                mask_data = cv2.resize(
                    mask_data, (w, h), interpolation=cv2.INTER_NEAREST
                )
            binary_mask = (mask_data > 0.5).astype(np.uint8)

            # Calculate area
            area = int(np.sum(binary_mask))
            if area < SAM2_MIN_MASK_AREA:
                continue

            import config
            # Convert mask to polygon
            polygon = self.mask_to_polygon(binary_mask, epsilon=config.POLYGON_SIMPLIFY_EPSILON)
            if polygon is None or len(polygon) < 3:
                continue

            # Calculate bounding box from mask
            ys, xs = np.where(binary_mask > 0)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

            # Get confidence if available
            confidence = 1.0
            if hasattr(r, 'boxes') and r.boxes is not None and i < len(r.boxes):
                confidence = float(r.boxes.conf[i].item())

            segments.append({
                "mask": binary_mask,
                "polygon": polygon,
                "bbox": bbox,
                "area": area,
                "confidence": confidence,
                "segment_id": i,
            })

        # Sort by area descending (largest segments first)
        segments.sort(key=lambda s: s["area"], reverse=True)
        return segments

    @staticmethod
    def mask_to_polygon(
        binary_mask: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Convert a binary mask to a simplified polygon using contour detection.

        Args:
            binary_mask: (H, W) uint8 array with values 0 or 1.
            epsilon: Approximation accuracy for cv2.approxPolyDP.
                     Higher = fewer points = simpler polygon.

        Returns:
            List of (x, y) integer tuples, or None if no valid contour found.
        """
        if epsilon is None:
            import config
            epsilon = config.POLYGON_SIMPLIFY_EPSILON
            
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Take the largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < SAM2_MIN_MASK_AREA:
            return None

        # Simplify polygon
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        # Convert to list of (x, y) tuples
        polygon = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        return polygon

    @staticmethod
    def polygon_to_mask(
        polygon: List[Tuple[int, int]], height: int, width: int
    ) -> np.ndarray:
        """
        Convert polygon points back to a binary mask.

        Args:
            polygon: List of (x, y) integer tuples.
            height: Image height.
            width: Image width.

        Returns:
            (H, W) uint8 binary mask.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    @staticmethod
    def crop_polygon_region(
        image: np.ndarray, polygon: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Crop the bounding region of a polygon from the image,
        with pixels outside the polygon set to black.

        Args:
            image: Source image (BGR).
            polygon: List of (x, y) points.

        Returns:
            Cropped image region.
        """
        h, w = image.shape[:2]
        mask = SAM2Segmentor.polygon_to_mask(polygon, h, w)

        # Apply mask
        masked = cv2.bitwise_and(image, image, mask=mask)

        # Crop to bounding box
        pts = np.array(polygon)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        return masked[y1:y2, x1:x2]
