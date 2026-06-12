"""
Active Learning Module
======================
Reward/feedback tracking system for the MTA pipeline.
Uses SQLite to persist prediction history, rewards, and corrections.

Implements:
  - Reward tracking (+1 correct, -1 incorrect) for polygons AND classes
  - Confidence-based uncertainty sampling
  - Margin-based confusion detection
  - Correction export for model retraining
  - Per-class accuracy metrics and dashboards
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from config import (
    DB_PATH, CLASS_NAMES, CORRECTIONS_DIR,
    CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD,
    RETRAIN_TRIGGER_COUNT, REWARD_POLYGON_CORRECT, REWARD_POLYGON_INCORRECT,
    REWARD_CLASS_CORRECT, REWARD_CLASS_INCORRECT
)


class ActiveLearningManager:
    """
    Manages the active learning loop with reward/feedback tracking.

    Database schema:
      - predictions: all model predictions with feedback
      - rewards: cumulative reward scores
      - model_versions: training history
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the active learning manager.

        Args:
            db_path: Path to SQLite database. Defaults to config value.
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._init_db()

    def _init_db(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                segment_id INTEGER,
                polygon_json TEXT,
                predicted_class TEXT,
                predicted_class_id INTEGER,
                confidence REAL,
                margin REAL,
                actual_class TEXT,
                actual_class_id INTEGER,
                polygon_correct INTEGER DEFAULT NULL,
                class_correct INTEGER DEFAULT NULL,
                polygon_reward INTEGER DEFAULT 0,
                class_reward INTEGER DEFAULT 0,
                reviewed INTEGER DEFAULT 0,
                model_version TEXT DEFAULT 'v1.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS reward_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                reward_type TEXT,
                reward_value INTEGER,
                class_name TEXT,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE,
                model_path TEXT,
                training_data_count INTEGER,
                corrections_count INTEGER,
                map50 REAL,
                map50_95 REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS session_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                total_predictions INTEGER DEFAULT 0,
                polygons_correct INTEGER DEFAULT 0,
                polygons_incorrect INTEGER DEFAULT 0,
                classes_correct INTEGER DEFAULT 0,
                classes_incorrect INTEGER DEFAULT 0,
                total_reward INTEGER DEFAULT 0,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Recording predictions
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        image_path: str,
        segment_id: int,
        polygon: List[Tuple[int, int]],
        predicted_class: str,
        predicted_class_id: int,
        confidence: float,
        margin: float = 0.0,
        model_version: str = "v1.0",
    ) -> int:
        """
        Record a new model prediction.

        Args:
            image_path: Path to the source image.
            segment_id: Index of the segment within the image.
            polygon: List of (x, y) polygon points.
            predicted_class: Predicted class name.
            predicted_class_id: Predicted class index.
            confidence: Prediction confidence.
            margin: Confidence margin (top1 - top2).
            model_version: Current model version string.

        Returns:
            The prediction ID in the database.
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            INSERT INTO predictions
                (image_path, segment_id, polygon_json, predicted_class,
                 predicted_class_id, confidence, margin, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(image_path), segment_id, json.dumps(polygon),
            predicted_class, predicted_class_id, confidence,
            margin, model_version
        ))

        pred_id = c.lastrowid
        conn.commit()
        conn.close()
        return pred_id

    # ------------------------------------------------------------------
    # Recording feedback
    # ------------------------------------------------------------------

    def record_polygon_feedback(
        self, prediction_id: int, is_correct: bool
    ) -> int:
        """
        Record whether the polygon segmentation was correct.

        Args:
            prediction_id: ID of the prediction record.
            is_correct: True if polygon is acceptable, False if not.

        Returns:
            Reward value (+1 or -1).
        """
        reward = REWARD_POLYGON_CORRECT if is_correct else REWARD_POLYGON_INCORRECT

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            UPDATE predictions
            SET polygon_correct = ?, polygon_reward = ?, reviewed_at = ?
            WHERE id = ?
        """, (int(is_correct), reward, datetime.now().isoformat(), prediction_id))

        c.execute("""
            INSERT INTO reward_history
                (prediction_id, reward_type, reward_value, class_name, model_version)
            SELECT ?, 'polygon', ?, predicted_class, model_version
            FROM predictions WHERE id = ?
        """, (prediction_id, reward, prediction_id))

        conn.commit()
        conn.close()

        feedback_msg = "✅ CORRECT (+1 reward)" if is_correct else "❌ INCORRECT (-1 penalty)"
        print(f"[Active Learning] Polygon #{prediction_id}: {feedback_msg}")
        return reward

    def record_class_feedback(
        self,
        prediction_id: int,
        is_correct: bool,
        corrected_class: Optional[str] = None,
        corrected_class_id: Optional[int] = None,
    ) -> int:
        """
        Record whether the class prediction was correct.
        If incorrect, store the corrected class for retraining.

        Args:
            prediction_id: ID of the prediction record.
            is_correct: True if class is correct, False if not.
            corrected_class: The correct class name (if is_correct is False).
            corrected_class_id: The correct class index.

        Returns:
            Reward value (+1 or -1).
        """
        reward = REWARD_CLASS_CORRECT if is_correct else REWARD_CLASS_INCORRECT

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        if is_correct:
            # Class is correct — actual = predicted
            c.execute("""
                UPDATE predictions
                SET class_correct = 1, class_reward = ?,
                    actual_class = predicted_class,
                    actual_class_id = predicted_class_id,
                    reviewed = 1, reviewed_at = ?
                WHERE id = ?
            """, (reward, datetime.now().isoformat(), prediction_id))
        else:
            # Class is wrong — store correction
            c.execute("""
                UPDATE predictions
                SET class_correct = 0, class_reward = ?,
                    actual_class = ?, actual_class_id = ?,
                    reviewed = 1, reviewed_at = ?
                WHERE id = ?
            """, (
                reward, corrected_class, corrected_class_id,
                datetime.now().isoformat(), prediction_id
            ))

        c.execute("""
            INSERT INTO reward_history
                (prediction_id, reward_type, reward_value, class_name, model_version)
            SELECT ?, 'class', ?, predicted_class, model_version
            FROM predictions WHERE id = ?
        """, (prediction_id, reward, prediction_id))

        conn.commit()
        conn.close()

        if is_correct:
            print(f"[Active Learning] Class #{prediction_id}: ✅ CORRECT (+1 reward)")
        else:
            print(f"[Active Learning] Class #{prediction_id}: ❌ INCORRECT → "
                  f"corrected to '{corrected_class}' (-1 penalty)")
        return reward

    # ------------------------------------------------------------------
    # Metrics & summaries
    # ------------------------------------------------------------------

    def get_reward_summary(self) -> Dict:
        """
        Get overall reward summary and accuracy metrics.

        Returns:
            Dict with total_predictions, accuracy, rewards, per-class stats.
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # Overall stats
        c.execute("SELECT COUNT(*) FROM predictions")
        total = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM predictions WHERE reviewed = 1")
        reviewed = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM predictions WHERE polygon_correct = 1")
        poly_correct = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM predictions WHERE polygon_correct = 0")
        poly_incorrect = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM predictions WHERE class_correct = 1")
        cls_correct = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM predictions WHERE class_correct = 0")
        cls_incorrect = c.fetchone()[0]

        c.execute("SELECT COALESCE(SUM(polygon_reward + class_reward), 0) FROM predictions")
        total_reward = c.fetchone()[0]

        # Per-class accuracy
        c.execute("""
            SELECT predicted_class,
                   COUNT(*) as total,
                   SUM(CASE WHEN class_correct = 1 THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN class_correct = 0 THEN 1 ELSE 0 END) as incorrect,
                   AVG(confidence) as avg_conf
            FROM predictions
            WHERE reviewed = 1
            GROUP BY predicted_class
            ORDER BY total DESC
        """)
        per_class = []
        for row in c.fetchall():
            class_total = row[1]
            class_correct_count = row[2] or 0
            per_class.append({
                "class_name": row[0],
                "total": class_total,
                "correct": class_correct_count,
                "incorrect": row[3] or 0,
                "accuracy": class_correct_count / max(class_total, 1),
                "avg_confidence": row[4] or 0.0,
            })

        conn.close()

        poly_reviewed = poly_correct + poly_incorrect
        cls_reviewed = cls_correct + cls_incorrect

        return {
            "total_predictions": total,
            "total_reviewed": reviewed,
            "polygon_accuracy": poly_correct / max(poly_reviewed, 1),
            "polygon_correct": poly_correct,
            "polygon_incorrect": poly_incorrect,
            "class_accuracy": cls_correct / max(cls_reviewed, 1),
            "class_correct": cls_correct,
            "class_incorrect": cls_incorrect,
            "total_reward": total_reward,
            "per_class": per_class,
        }

    def get_session_stats(self, last_n: int = 10) -> List[Dict]:
        """Get recent reward history entries."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT rh.id, rh.reward_type, rh.reward_value, rh.class_name,
                   rh.model_version, rh.created_at,
                   p.image_path, p.predicted_class, p.confidence
            FROM reward_history rh
            JOIN predictions p ON rh.prediction_id = p.id
            ORDER BY rh.created_at DESC
            LIMIT ?
        """, (last_n,))

        history = []
        for row in c.fetchall():
            history.append({
                "id": row[0],
                "type": row[1],
                "reward": row[2],
                "class": row[3],
                "model_version": row[4],
                "timestamp": row[5],
                "image": row[6],
                "predicted_class": row[7],
                "confidence": row[8],
            })

        conn.close()
        return history

    def get_confusion_matrix(self) -> Dict:
        """
        Build a confusion matrix from predictions vs actual classes.

        Returns:
            Dict with 'matrix' (2D list), 'labels' (class names).
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT predicted_class, actual_class, COUNT(*)
            FROM predictions
            WHERE reviewed = 1 AND actual_class IS NOT NULL
            GROUP BY predicted_class, actual_class
        """)

        matrix = {}
        for row in c.fetchall():
            pred, actual, count = row
            if pred not in matrix:
                matrix[pred] = {}
            matrix[pred][actual] = count

        conn.close()
        return {"matrix": matrix, "labels": CLASS_NAMES}

    # ------------------------------------------------------------------
    # Active learning sampling
    # ------------------------------------------------------------------

    def get_hard_samples(self, n: int = 50) -> List[Dict]:
        """
        Get the most uncertain/hard samples for priority review.
        Uses confidence-based and margin-based sampling.

        Args:
            n: Number of samples to return.

        Returns:
            List of prediction dicts sorted by uncertainty.
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT id, image_path, segment_id, predicted_class,
                   confidence, margin, polygon_json
            FROM predictions
            WHERE reviewed = 0
            ORDER BY confidence ASC, margin ASC
            LIMIT ?
        """, (n,))

        samples = []
        for row in c.fetchall():
            samples.append({
                "prediction_id": row[0],
                "image_path": row[1],
                "segment_id": row[2],
                "predicted_class": row[3],
                "confidence": row[4],
                "margin": row[5],
                "polygon": json.loads(row[6]) if row[6] else [],
            })

        conn.close()
        return samples

    def get_unreviewed_count(self) -> int:
        """Get the number of predictions awaiting review."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions WHERE reviewed = 0")
        count = c.fetchone()[0]
        conn.close()
        return count

    # ------------------------------------------------------------------
    # Retraining support
    # ------------------------------------------------------------------

    def should_retrain(self) -> bool:
        """
        Check if enough corrections have accumulated to trigger retraining.

        Returns:
            True if corrections count >= RETRAIN_TRIGGER_COUNT.
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE class_correct = 0 AND reviewed = 1
        """)
        corrections = c.fetchone()[0]

        conn.close()
        return corrections >= RETRAIN_TRIGGER_COUNT

    def export_corrections_for_training(
        self, output_dir: Optional[str] = None
    ) -> str:
        """
        Export all corrections in YOLO format for fine-tuning.

        Creates:
          - images/ folder with source images
          - labels/ folder with corrected YOLO txt labels

        Args:
            output_dir: Output directory. Defaults to corrections/ folder.

        Returns:
            Path to the output directory.
        """
        out = Path(output_dir) if output_dir else CORRECTIONS_DIR
        images_dir = out / "images"
        labels_dir = out / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # Get all reviewed predictions (correct ones too for complete labels)
        c.execute("""
            SELECT image_path, segment_id, polygon_json,
                   actual_class_id, actual_class
            FROM predictions
            WHERE reviewed = 1 AND actual_class IS NOT NULL
            ORDER BY image_path, segment_id
        """)

        # Group by image
        image_labels = {}
        for row in c.fetchall():
            img_path, seg_id, poly_json, cls_id, cls_name = row
            if img_path not in image_labels:
                image_labels[img_path] = []
            polygon = json.loads(poly_json) if poly_json else []
            image_labels[img_path].append({
                "class_id": cls_id,
                "polygon": polygon,
            })

        conn.close()

        # Write YOLO format labels
        import cv2
        import shutil

        exported_count = 0
        for img_path, labels in image_labels.items():
            img_path = Path(img_path)
            if not img_path.exists():
                continue

            # Copy image
            dst_img = images_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(str(img_path), str(dst_img))

            # Read image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Write YOLO polygon label
            label_path = labels_dir / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                for label in labels:
                    cls_id = label["class_id"]
                    polygon = label["polygon"]
                    if not polygon or cls_id is None:
                        continue
                    # Normalize polygon coordinates
                    normalized = []
                    for px, py in polygon:
                        normalized.append(f"{px / w:.6f}")
                        normalized.append(f"{py / h:.6f}")
                    f.write(f"{cls_id} {' '.join(normalized)}\n")
                    exported_count += 1

        print(f"[Active Learning] Exported {exported_count} labels "
              f"from {len(image_labels)} images to {out}")
        return str(out)

    def get_corrections_count(self) -> int:
        """Get the number of class corrections awaiting export."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE class_correct = 0 AND reviewed = 1
        """)
        count = c.fetchone()[0]
        conn.close()
        return count

    def record_model_version(
        self,
        version: str,
        model_path: str,
        training_data_count: int = 0,
        corrections_count: int = 0,
        map50: float = 0.0,
        map50_95: float = 0.0,
        notes: str = "",
    ):
        """Record a new model version after retraining."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO model_versions
                (version, model_path, training_data_count, corrections_count,
                 map50, map50_95, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (version, model_path, training_data_count,
              corrections_count, map50, map50_95, notes))
        conn.commit()
        conn.close()
