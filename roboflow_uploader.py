"""
Roboflow Uploader Module
========================
Handles uploading approved annotations to Roboflow for dataset management.
Supports both single image uploads and batch uploads with active learning flags.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from config import (
    ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT,
    CLASS_NAMES, ANNOTATIONS_DIR
)


class RoboflowUploader:
    """
    Uploads approved annotations to Roboflow.

    Supports:
      - Single image + annotation upload
      - Batch upload of approved items
      - Active learning mode (is_prediction=True for human review)
      - COCO JSON and YOLO format conversion
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
    ):
        """
        Initialize Roboflow uploader.

        Args:
            api_key: Roboflow API key. Falls back to env var / config.
            workspace: Roboflow workspace name.
            project: Roboflow project name.
        """
        self.api_key = api_key or ROBOFLOW_API_KEY
        self.workspace = workspace or ROBOFLOW_WORKSPACE
        self.project_name = project or ROBOFLOW_PROJECT
        self._rf = None
        self._project = None

    def _ensure_connection(self):
        """Lazily initialize Roboflow connection."""
        if self._project is not None:
            return

        if not self.api_key:
            raise ValueError(
                "Roboflow API key not set. Set ROBOFLOW_API_KEY environment "
                "variable or pass api_key to RoboflowUploader().\n"
                "Get your key at: https://app.roboflow.com/settings/api"
            )

        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError(
                "roboflow package not installed. Install with:\n"
                "  pip install roboflow"
            )

        print(f"[Roboflow] Connecting to workspace: {self.workspace}...")
        self._rf = Roboflow(api_key=self.api_key)
        self._project = self._rf.workspace(self.workspace).project(self.project_name)
        print(f"[Roboflow] Connected to project: {self.project_name}")

    def upload_annotation(
        self,
        image_path: str,
        polygons_with_classes: List[Dict],
        is_prediction: bool = False,
        batch_name: Optional[str] = None,
    ) -> bool:
        """
        Upload a single image with polygon annotations to Roboflow.

        Args:
            image_path: Path to the image file.
            polygons_with_classes: List of dicts with:
                - 'polygon': list of (x, y) tuples
                - 'class_name': class label
                - 'class_id': class index
            is_prediction: If True, upload as model prediction for active
                          learning review. If False, upload as verified annotation.
            batch_name: Optional batch grouping name.

        Returns:
            True if upload succeeded, False otherwise.
        """
        self._ensure_connection()

        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[Roboflow] Error: Image not found: {image_path}")
            return False

        # Create YOLO-format annotation file
        annotation_path = self._create_yolo_annotation(
            image_path, polygons_with_classes
        )

        try:
            upload_kwargs = {
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "annotation_labelmap": str(ANNOTATIONS_DIR / "classes.txt"),
                "is_prediction": is_prediction,
            }
            if batch_name:
                upload_kwargs["batch_name"] = batch_name

            self._project.single_upload(**upload_kwargs)

            status = "prediction" if is_prediction else "verified"
            print(f"[Roboflow] ✅ Uploaded {image_path.name} as {status}")
            return True

        except Exception as e:
            print(f"[Roboflow] ❌ Upload failed for {image_path.name}: {e}")
            return False
        finally:
            # Clean up temp annotation file
            if annotation_path.exists():
                annotation_path.unlink()

    def upload_batch(
        self,
        approved_items: List[Dict],
        is_prediction: bool = False,
        batch_name: Optional[str] = None,
    ) -> Dict:
        """
        Batch upload multiple approved annotations.

        Args:
            approved_items: List of dicts with:
                - 'image_path': path to image
                - 'polygons': list of polygon+class dicts
            is_prediction: Upload as predictions for review.
            batch_name: Batch grouping name.

        Returns:
            Dict with 'success_count', 'fail_count', 'total'.
        """
        self._ensure_connection()

        success = 0
        fail = 0

        for i, item in enumerate(approved_items):
            print(f"[Roboflow] Uploading {i+1}/{len(approved_items)}...")
            ok = self.upload_annotation(
                image_path=item["image_path"],
                polygons_with_classes=item["polygons"],
                is_prediction=is_prediction,
                batch_name=batch_name,
            )
            if ok:
                success += 1
            else:
                fail += 1

        result = {
            "success_count": success,
            "fail_count": fail,
            "total": len(approved_items),
        }
        print(f"[Roboflow] Batch complete: {success} uploaded, {fail} failed")
        return result

    def _create_yolo_annotation(
        self,
        image_path: Path,
        polygons_with_classes: List[Dict],
    ) -> Path:
        """
        Create a temporary YOLO-format annotation file for upload.

        YOLO polygon format:
          class_id x1 y1 x2 y2 ... xN yN  (normalized 0-1)

        Returns:
            Path to the temporary annotation file.
        """
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]

        # Write YOLO annotation
        ann_path = ANNOTATIONS_DIR / (image_path.stem + ".txt")

        with open(ann_path, "w") as f:
            for item in polygons_with_classes:
                cls_id = item.get("class_id", 0)
                polygon = item.get("polygon", [])
                if not polygon:
                    continue

                # Normalize coordinates
                normalized = []
                for px, py in polygon:
                    normalized.append(f"{px / w:.6f}")
                    normalized.append(f"{py / h:.6f}")

                f.write(f"{cls_id} {' '.join(normalized)}\n")

        # Also write classes.txt in the same directory so Roboflow parses IDs to names
        classes_path = ANNOTATIONS_DIR / "classes.txt"
        with open(classes_path, "w") as f:
            f.write("\n".join(CLASS_NAMES) + "\n")

        return ann_path

    def check_connection(self) -> bool:
        """Test if Roboflow connection is working."""
        try:
            self._ensure_connection()
            return True
        except Exception as e:
            print(f"[Roboflow] Connection failed: {e}")
            return False

    @staticmethod
    def is_configured() -> bool:
        """Check if Roboflow credentials are configured."""
        return bool(ROBOFLOW_API_KEY and ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT)
