"""
MTA Active Learning Pipeline — Configuration
=============================================
Central configuration for all pipeline components.
Edit this file to set your API keys, model paths, and thresholds.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ANNOTATIONS_DIR = SCRIPT_DIR / "annotations"
CORRECTIONS_DIR = SCRIPT_DIR / "corrections"
DB_PATH = SCRIPT_DIR / "active_learning.db"

# Ensure output dirs exist
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
SAM2_MODEL = "sam2.1_t.pt"          # Default: tiny (fastest)
SAM2_AVAILABLE_MODELS = [           # For UI dropdown switcher
    "sam2.1_t.pt",                   # Tiny  — fastest, ~150MB
    "sam2.1_s.pt",                   # Small — balanced, ~180MB
    "sam2.1_b.pt",                   # Base  — ~375MB
    "sam2.1_l.pt",                   # Large — best quality, ~2.4GB
]
MTA_MODEL_PATH = PROJECT_DIR / "float32-small-x1.pt"  # YOLOv11s segmentation model
MTA_ATTENTION_PATH = SCRIPT_DIR / "mta_attention.pt"  # YOLO attention model

# Fallback to other models if primary not found
if not MTA_MODEL_PATH.exists():
    for fallback in ["best.pt", "float32-nano-x5.pt", "last.pt"]:
        alt = PROJECT_DIR / fallback
        if alt.exists():
            MTA_MODEL_PATH = alt
            break

# ONNX model for X-AnyLabeling
MTA_ONNX_PATH = PROJECT_DIR / "best.onnx"

# ---------------------------------------------------------------------------
# MTA Class Names (must match data.yaml order)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "trash",
    "m-composite",
    "m-glass",
    "m-metal",
    "m-paper",
    "m-plastic-film",
    "m-plastic-rigid",
    "s-cigarette-butt",
    "s-e-waste",
    "s-hazardous",
    "s-ikat-tepi",
    "s-litter",
    "s-organic",
    "s-other",
    "s-textile",
]

# Color palette — one distinct colour per class (RGB)
CLASS_COLORS = [
    (230,  25,  75),   # trash
    (255, 225,  25),   # m-composite
    (  0, 130, 200),   # m-glass
    (245, 130,  48),   # m-metal
    ( 60, 180,  75),   # m-paper
    (145,  30, 180),   # m-plastic-film
    ( 70, 240, 240),   # m-plastic-rigid
    (240,  50, 230),   # s-cigarette-butt
    (210, 245,  60),   # s-e-waste
    (250, 190, 212),   # s-hazardous
    (  0, 128, 128),   # s-ikat-tepi
    (220, 190, 255),   # s-litter
    (170, 110,  40),   # s-organic
    (255, 250, 200),   # s-other
    (128,   0,   0),   # s-textile
]

# ---------------------------------------------------------------------------
# Roboflow Configuration
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_PROJECT = os.environ.get("ROBOFLOW_PROJECT", "")

# ---------------------------------------------------------------------------
# Active Learning Thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.50       # Below this → uncertain sample (needs review)
MARGIN_THRESHOLD = 0.15           # top1 - top2 < this → model is confused
AUTO_APPROVE_THRESHOLD = 0.85     # Above this → can auto-approve
RETRAIN_TRIGGER_COUNT = 50        # Retrain after N corrections accumulated
# ---------------------------------------------------------------------------
# SAM2 Settings
# ---------------------------------------------------------------------------
SAM2_POINTS_PER_SIDE = 12         # Density of point grid (12x12 = 144 points). Lower = faster.
SAM2_PRED_IOU_THRESH = 0.86       # Predicted IoU threshold for mask filtering
SAM2_STABILITY_SCORE = 0.92       # Stability score threshold
SAM2_MIN_MASK_AREA = 100          # Minimum mask area in pixels (filter tiny masks)
POLYGON_SIMPLIFY_EPSILON = 2.0    # cv2.approxPolyDP epsilon for polygon simplification

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = 640                       # Input image size for YOLO inference

# ---------------------------------------------------------------------------
# Scoring & Pipeline Thresholds
# ---------------------------------------------------------------------------
REWARD_POLYGON_CORRECT = 1
REWARD_POLYGON_INCORRECT = -1
REWARD_CLASS_CORRECT = 1
REWARD_CLASS_INCORRECT = -1
REWARD_AUTO_APPROVE = 2

IOU_THRESHOLD_ENSEMBLE = 0.3      # Overlap required to merge in Ensemble mode
IOU_THRESHOLD_INTERSECT = 0.1     # Overlap required to intersect in Intersect mode
MIN_INTERSECT_AREA = 50           # Minimum pixel area for a valid mask intersection

MTA_CONFIDENCE_THRESHOLD = 0.05   # Base threshold for candidate generation
