"""
Custom COCO Export Module (Experimental)
========================================
Exports annotations to a standard COCO format extended with custom attributes
(contamination, material, object) parsed dynamically from the class names.
"""

import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional

from config import (
    CLASS_NAMES, ANNOTATIONS_DIR,
    MATERIAL_CLASSES, OBJECT_CLASSES, CONTAMINATION_CLASSES
)

class CustomCocoExporter:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else ANNOTATIONS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.categories = [{"id": 0, "name": "trash", "supercategory": "none"}]
        
        def _build_list_dict(lst):
            return [{"id": i, "name": name} for i, name in enumerate(lst)]
            
        self.materials = _build_list_dict(MATERIAL_CLASSES)
        self.objects = _build_list_dict(OBJECT_CLASSES)
        self.contaminations = _build_list_dict(CONTAMINATION_CLASSES)

    def _resolve_id(self, lst: list, name: str) -> int:
        try:
            return lst.index(name)
        except ValueError:
            return 0  # default to unknown/first element

    def _extract_attributes(self, class_name: str) -> Dict:
        # Fallback heuristic if no custom attributes are provided
        if not class_name.startswith("m-"):
            return {}

        material = class_name[2:]
        return {
            "material_id": self._resolve_id(MATERIAL_CLASSES, material),
            "object_id": self._resolve_id(OBJECT_CLASSES, "unknown"),
            "contamination_id": self._resolve_id(CONTAMINATION_CLASSES, "null")
        }

    def export_batch(self, items: List[Dict], filename: str = "custom_coco_annotations.json", yolo_target: str = "material") -> str:
        coco = {
            "info": {
                "description": "MTA-COCO v2 Export",
                "version": "2.0",
                "year": datetime.datetime.now().year,
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": self.categories,
            "materials": self.materials,
            "objects": self.objects,
            "contaminations": self.contaminations
        }

        ann_id = 0
        img_id = 0

        for item in items:
            img_path = Path(item["image_path"])
            h = item.get("image_height", 640)
            w = item.get("image_width", 640)

            coco["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "height": h,
                "width": w
            })

            yolo_lines = []

            for poly_info in item.get("polygons", []):
                polygon = poly_info.get("polygon", [])
                class_name = poly_info.get("class_name", "unknown")

                if len(polygon) < 3:
                    continue

                # bounding box
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bw, bh = xmax - xmin, ymax - ymin

                # Shoelace formula for area
                area = 0.0
                for j in range(len(polygon)):
                    j1 = (j + 1) % len(polygon)
                    area += polygon[j][0] * polygon[j1][1]
                    area -= polygon[j1][0] * polygon[j][1]
                area = abs(area) / 2.0

                flat_poly = []
                norm_poly = []
                for x, y in polygon:
                    flat_poly.extend([float(x), float(y)])
                    norm_poly.extend([f"{float(x)/w:.6f}", f"{float(y)/h:.6f}"])

                if "custom_attributes" in poly_info and poly_info["custom_attributes"] is not None:
                    ca = poly_info["custom_attributes"]
                    attributes = {
                        "material_id": self._resolve_id(MATERIAL_CLASSES, ca.get("material", "unknown")),
                        "object_id": self._resolve_id(OBJECT_CLASSES, ca.get("object", "unknown")),
                        "contamination_id": self._resolve_id(CONTAMINATION_CLASSES, str(ca.get("contamination", "null")))
                    }
                else:
                    attributes = self._extract_attributes(class_name)

                # Determine YOLO class ID
                if yolo_target.lower() == "object":
                    yolo_cls_id = attributes["object_id"]
                elif yolo_target.lower() == "contamination":
                    yolo_cls_id = attributes["contamination_id"]
                else:
                    yolo_cls_id = attributes["material_id"]

                yolo_lines.append(f"{yolo_cls_id} " + " ".join(norm_poly))

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,  # MTA-COCO v2: Everything is just trash
                    "segmentation": [flat_poly],
                    "area": area,
                    "bbox": [xmin, ymin, bw, bh],
                    "iscrowd": 0,
                    "attributes": attributes
                })
                ann_id += 1

            # Write YOLO TXT file
            yolo_txt_path = self.output_dir / (img_path.stem + ".txt")
            with open(yolo_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")

            img_id += 1

        json_path = self.output_dir / filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

        print(f"[Custom COCO] Exported {len(coco['images'])} images, {ann_id} annotations to {json_path}")
        return str(json_path)

    def export_annotation(self, image_path: str, polygons_with_classes: List[Dict], image_height: int, image_width: int, yolo_target: str = "material") -> str:
        # Wrapper for single image export
        item = {
            "image_path": image_path,
            "image_height": image_height,
            "image_width": image_width,
            "polygons": polygons_with_classes
        }
        filename = Path(image_path).stem + "_coco.json"
        return self.export_batch([item], filename=filename, yolo_target=yolo_target)
