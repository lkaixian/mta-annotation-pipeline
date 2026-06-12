"""
X-AnyLabeling Export Module
===========================
Generates X-AnyLabeling compatible annotation files (.json) and model
configuration (config.yaml) so annotations can be loaded and refined
in the X-AnyLabeling desktop application.

Format reference: https://github.com/CVHub520/X-AnyLabeling
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from config import (
    CLASS_NAMES, MTA_ONNX_PATH, ANNOTATIONS_DIR
)


class XAnyLabelingExporter:
    """
    Exports annotations in X-AnyLabeling JSON format and generates
    model configuration for loading the MTA model in X-AnyLabeling.

    X-AnyLabeling JSON format:
    {
      "version": "0.4.0",
      "flags": {},
      "shapes": [
        {
          "label": "class_name",
          "text": "",
          "points": [[x1,y1], [x2,y2], ...],
          "group_id": null,
          "shape_type": "polygon",
          "flags": {}
        }
      ],
      "imagePath": "image.jpg",
      "imageData": null,
      "imageHeight": 480,
      "imageWidth": 640
    }
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the exporter.

        Args:
            output_dir: Directory to save annotation files. Defaults to config.
        """
        self.output_dir = Path(output_dir) if output_dir else ANNOTATIONS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_annotation(
        self,
        image_path: str,
        polygons_with_classes: List[Dict],
        image_height: int,
        image_width: int,
        confidence_in_label: bool = True,
    ) -> str:
        """
        Export polygon annotations for an image in X-AnyLabeling JSON format.

        Args:
            image_path: Path to the source image.
            polygons_with_classes: List of dicts with:
                - 'polygon': list of (x, y) tuples
                - 'class_name': class label
                - 'confidence': prediction confidence (optional)
                - 'class_id': class index (optional)
            image_height: Image height in pixels.
            image_width: Image width in pixels.
            confidence_in_label: If True, append confidence to label text.

        Returns:
            Path to the saved JSON annotation file.
        """
        image_path = Path(image_path)

        shapes = []
        for item in polygons_with_classes:
            polygon = item.get("polygon", [])
            class_name = item.get("class_name", "unknown")
            confidence = item.get("confidence", 0.0)

            if not polygon:
                continue

            # Format points as [[x1,y1], [x2,y2], ...]
            points = [[float(x), float(y)] for x, y in polygon]

            # Build label text
            label = class_name
            description = ""
            if confidence_in_label and confidence > 0:
                description = f"{confidence:.1%}"

            shape = {
                "label": label,
                "text": description,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "description": description,
                "difficult": False,
                "attributes": {},
            }
            shapes.append(shape)

        # Build annotation document
        annotation = {
            "version": "0.4.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }

        # Save JSON file (same name as image, but .json)
        json_filename = image_path.stem + ".json"
        json_path = self.output_dir / json_filename

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        print(f"[X-AnyLabeling] Exported: {json_path.name} "
              f"({len(shapes)} shapes)")
        return str(json_path)

    def export_batch(
        self,
        items: List[Dict],
        confidence_in_label: bool = True,
    ) -> List[str]:
        """
        Export annotations for multiple images.

        Args:
            items: List of dicts with:
                - 'image_path': path to image
                - 'polygons': list of polygon+class dicts
                - 'image_height': image height
                - 'image_width': image width
            confidence_in_label: Append confidence to labels.

        Returns:
            List of exported JSON file paths.
        """
        exported = []
        for item in items:
            path = self.export_annotation(
                image_path=item["image_path"],
                polygons_with_classes=item["polygons"],
                image_height=item["image_height"],
                image_width=item["image_width"],
                confidence_in_label=confidence_in_label,
            )
            exported.append(path)

        print(f"[X-AnyLabeling] Batch export: {len(exported)} files")
        return exported

    @staticmethod
    def generate_model_config(
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate an X-AnyLabeling model configuration YAML file
        so the MTA model can be loaded directly in the application.

        Args:
            model_path: Path to the ONNX model. Defaults to config.
            output_path: Where to save config.yaml. Defaults to same dir as model.

        Returns:
            Path to the generated config.yaml.
        """
        import yaml

        model_path = Path(model_path) if model_path else MTA_ONNX_PATH

        config = {
            "type": "yolov8_seg",
            "name": "mta-trash-annotation",
            "display_name": "Malaysian Trash Annotation (MTA) - YOLOv11s",
            "model_path": model_path.name,
            "engine": "ort",
            "input_width": 640,
            "input_height": 640,
            "score_threshold": 0.25,
            "nms_threshold": 0.45,
            "confidence_threshold": 0.25,
            "classes": CLASS_NAMES,
        }

        if output_path:
            config_path = Path(output_path)
        else:
            config_path = model_path.parent / "xanylabeling_config.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"[X-AnyLabeling] Model config saved: {config_path}")
        return str(config_path)

    @staticmethod
    def load_annotation(json_path: str) -> Dict:
        """
        Load an X-AnyLabeling JSON annotation file.

        Args:
            json_path: Path to the .json annotation file.

        Returns:
            Parsed annotation dict with shapes, image info, etc.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert shapes to our internal format
        polygons = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue

            points = shape.get("points", [])
            polygon = [(int(p[0]), int(p[1])) for p in points]

            label = shape.get("label", "unknown")
            class_id = -1
            if label in CLASS_NAMES:
                class_id = CLASS_NAMES.index(label)

            polygons.append({
                "polygon": polygon,
                "class_name": label,
                "class_id": class_id,
                "description": shape.get("description", ""),
            })

        return {
            "image_path": data.get("imagePath", ""),
            "image_height": data.get("imageHeight", 0),
            "image_width": data.get("imageWidth", 0),
            "polygons": polygons,
        }
