"""
MTA Active Learning Pipeline — CLI Entry Point
===============================================
Main entry point for the MTA annotation pipeline.

Usage:
    python run.py                     # Launch review UI (default)
    python run.py --mode ui           # Launch review UI
    python run.py --mode batch        # Batch process folder
    python run.py --mode export       # Export corrections for retraining
    python run.py --mode dashboard    # Print active learning metrics
    python run.py --mode config       # Generate X-AnyLabeling model config
"""

import argparse
import sys
import io
import os
from pathlib import Path

# Ensure script directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fix Windows console encoding for emoji output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser(
        description="MTA Active Learning Pipeline — SAM2 + YOLO + Roboflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                              Launch review UI
  python run.py --mode batch --input gallery  Batch process gallery folder
  python run.py --mode export                Export corrections for retraining
  python run.py --mode dashboard             Show active learning metrics
  python run.py --mode config                Generate X-AnyLabeling config
        """
    )

    parser.add_argument(
        "--mode", type=str, default="ui",
        choices=["ui", "batch", "export", "dashboard", "config"],
        help="Pipeline mode (default: ui)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input image file or directory (for batch mode)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for annotations"
    )
    parser.add_argument(
        "--no-sam2", action="store_true",
        help="Disable SAM2, use MTA model masks only"
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="Auto-approve high-confidence predictions in batch mode"
    )
    parser.add_argument(
        "--upload-roboflow", action="store_true",
        help="Upload approved annotations to Roboflow"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override confidence threshold"
    )

    args = parser.parse_args()

    if args.mode == "ui":
        _run_ui()
    elif args.mode == "batch":
        _run_batch(args)
    elif args.mode == "export":
        _run_export(args)
    elif args.mode == "dashboard":
        _run_dashboard()
    elif args.mode == "config":
        _run_config(args)


def _run_ui():
    """Launch the interactive review UI."""
    print("=" * 60)
    print("  MTA Active Learning Pipeline — Review UI")
    print("=" * 60)
    print()

    from review_ui import ReviewApp
    app = ReviewApp()
    app.mainloop()


def _run_batch(args):
    """Process a folder of images in batch mode."""
    from pipeline import MTAPipeline

    input_path = args.input
    if not input_path:
        # Default to gallery folder
        input_path = str(Path(__file__).resolve().parent.parent / "gallery")

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ Input path not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("  MTA Active Learning Pipeline — Batch Mode")
    print("=" * 60)
    print(f"  Input: {input_path}")
    print(f"  SAM2: {'Enabled' if not args.no_sam2 else 'Disabled'}")
    print(f"  Auto-approve: {'Yes' if args.auto_approve else 'No'}")
    print("=" * 60)
    print()

    pipe = MTAPipeline(use_sam2=not args.no_sam2)

    if input_path.is_file():
        # Single image
        result = pipe.process_image(str(input_path))
        n = len(result.get("segments", []))
        t = result.get("processing_time", 0.0)
        print(f"\n✅ Processed {input_path.name}: {n} segments found in {t:.2f}s")

        if args.auto_approve:
            auto = pipe.auto_approve_high_confidence(result["segments"])
            print(f"  ⚡ Auto-approved: {auto['auto_approved']}, "
                  f"Pending: {auto['pending']}")

        # Export
        pipe.export_approved(
            str(input_path), result["segments"],
            to_roboflow=args.upload_roboflow,
            to_xanylabeling=True,
        )
    else:
        # Directory
        results = pipe.process_batch(
            str(input_path), auto_approve=args.auto_approve
        )
        total_segments = sum(len(r.get("segments", [])) for r in results)
        total_time = sum(r.get("processing_time", 0.0) for r in results)
        print(f"\n✅ Batch complete: {len(results)} images, "
              f"{total_segments} total segments in {total_time:.2f}s")

    # Print summary
    summary = pipe.active_learning.get_reward_summary()
    print(f"\n📊 Active Learning Summary:")
    print(f"  Total predictions: {summary['total_predictions']}")
    print(f"  Total reward: {summary['total_reward']:+d}")


def _run_export(args):
    """Export corrections for model retraining."""
    from active_learning import ActiveLearningManager

    print("=" * 60)
    print("  MTA Active Learning — Export Corrections")
    print("=" * 60)

    al = ActiveLearningManager()
    corrections = al.get_corrections_count()
    print(f"\n  Corrections available: {corrections}")

    if corrections == 0:
        print("  No corrections to export. Review some predictions first.")
        return

    output = args.output or None
    out_path = al.export_corrections_for_training(output)
    print(f"\n✅ Corrections exported to: {out_path}")
    print(f"  You can now fine-tune your model using this data.")
    print(f"\n  Example training command:")
    print(f"    yolo segment train data={out_path}/data.yaml "
          f"model=float32-small-x1.pt epochs=20")


def _run_dashboard():
    """Print active learning metrics to console."""
    from active_learning import ActiveLearningManager

    al = ActiveLearningManager()
    summary = al.get_reward_summary()

    print("=" * 60)
    print("  📊 MTA Active Learning Dashboard")
    print("=" * 60)
    print()
    print(f"  Total predictions   : {summary['total_predictions']}")
    print(f"  Total reviewed      : {summary['total_reviewed']}")
    print(f"  ─────────────────────────")
    print(f"  Polygon accuracy    : {summary['polygon_accuracy']:.1%} "
          f"({summary['polygon_correct']}✅ / {summary['polygon_incorrect']}❌)")
    print(f"  Class accuracy      : {summary['class_accuracy']:.1%} "
          f"({summary['class_correct']}✅ / {summary['class_incorrect']}❌)")
    print(f"  ─────────────────────────")
    print(f"  Total reward        : {summary['total_reward']:+d}")
    print(f"  Corrections pending : {al.get_corrections_count()}")
    print(f"  Should retrain?     : {'🔴 YES' if al.should_retrain() else '🟢 No'}")
    print()

    if summary.get("per_class"):
        print("  Per-Class Accuracy:")
        print(f"  {'Class':<20} {'Accuracy':<10} {'Correct':<10} {'Total':<8}")
        print(f"  {'─' * 48}")
        for pc in summary["per_class"]:
            print(f"  {pc['class_name']:<20} {pc['accuracy']:<10.1%} "
                  f"{pc['correct']:<10} {pc['total']:<8}")

    print()
    print("=" * 60)

    # Recent history
    history = al.get_session_stats(10)
    if history:
        print("\n  📜 Recent Activity:")
        for h in history:
            emoji = "✅" if h["reward"] > 0 else "❌"
            print(f"    {emoji} {h['type']}: {h['class']} ({h['reward']:+d}) "
                  f"- {h.get('timestamp', '')}")


def _run_config(args):
    """Generate X-AnyLabeling model configuration."""
    from xanylabeling_export import XAnyLabelingExporter

    print("=" * 60)
    print("  MTA — X-AnyLabeling Configuration")
    print("=" * 60)

    output = args.output or None
    config_path = XAnyLabelingExporter.generate_model_config(
        output_path=output
    )
    print(f"\n✅ Configuration saved: {config_path}")
    print(f"\n  To use in X-AnyLabeling:")
    print(f"  1. Open X-AnyLabeling")
    print(f"  2. Click the AI icon (or Ctrl+A)")
    print(f"  3. Click 'Load Custom Model'")
    print(f"  4. Select: {config_path}")


if __name__ == "__main__":
    main()
