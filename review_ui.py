"""
MTA Active Learning Review UI
==============================
CustomTkinter desktop application for human-in-the-loop review of
SAM2 polygons and MTA class predictions.

Features:
  - Image canvas with polygon overlays (click to select)
  - Polygon review: ✅ Accept / ❌ Reject
  - Class review: ✅ Accept / ✏️ Edit (dropdown correction)
  - Reward tracking display
  - Active learning dashboard
  - Roboflow upload integration
  - X-AnyLabeling export
"""

import os
import sys
import json
import threading
import time
from pathlib import Path
from tkinter import filedialog, Canvas

import cv2
import numpy as np
import torch
import customtkinter as ctk
from PIL import Image, ImageTk
from typing import Dict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CLASS_NAMES, CLASS_COLORS, DEVICE, ANNOTATIONS_DIR,
    AUTO_APPROVE_THRESHOLD, CONFIDENCE_THRESHOLD,
    SAM2_MODEL, SAM2_AVAILABLE_MODELS,
    REWARD_POLYGON_CORRECT, REWARD_POLYGON_INCORRECT,
    REWARD_CLASS_CORRECT, REWARD_CLASS_INCORRECT, REWARD_AUTO_APPROVE,
    MATERIAL_CLASSES, OBJECT_CLASSES, CONTAMINATION_CLASSES
)


# ============================================================================
# GPU Diagnostics
# ============================================================================
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"


class ReviewApp(ctk.CTk):
    """Main active learning review application."""

    VID_W, VID_H = 820, 600
    SIDE_W = 400

    def __init__(self, experimental_coco_ui=False):
        super().__init__()
        self.experimental_coco_ui = experimental_coco_ui
        self.title("MTA Active Learning Pipeline — Review UI")
        self.geometry("1320x800")
        self.minsize(1200, 700)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State
        self.pipeline = None
        self.current_result = None    # Pipeline result dict
        self.selected_segment = None  # Index into segments
        self.image_queue = []         # List of image paths to process
        self.queue_index = 0
        # Session state
        self.session_rewards = 0
        self.session_reviewed = 0
        self.session_correct = 0

        # Interactive Segmentation State
        self.active_points = []
        self.active_labels = []
        self.active_preview_seg = None
        self.editing_segment_index = None

        # Zoom/pan state
        self._zoom_factor = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._is_dragging = False
        self._scale = 1.0
        self._ox = self._oy = 0

        self._build_ui()
        self._load_pipeline_async()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Layout ──────────────────────────────────────────────────
    def _build_ui(self):
        # ─── Top bar ───
        top = ctk.CTkFrame(self, height=56, corner_radius=0, fg_color="#0d1117")
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(
            top, text="🔬  MTA Active Learning Pipeline",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#58a6ff"
        ).pack(side="left", padx=18)

        self.reward_display = ctk.CTkLabel(
            top, text="🏆 Rewards: 0  |  Reviewed: 0/0",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#7ee787"
        )
        self.reward_display.pack(side="right", padx=18)

        self.status = ctk.CTkLabel(
            top, text="Loading pipeline…",
            font=ctk.CTkFont(size=12), text_color="#8b949e"
        )
        self.status.pack(side="right", padx=18)

        # ─── Body ───
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        # ─── Canvas (left) ───
        vf = ctk.CTkFrame(body, corner_radius=12, fg_color="#161b22")
        vf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.canvas = Canvas(vf, bg="#161b22", highlightthickness=0, cursor="hand2")
        self.canvas.pack(expand=True, fill="both", padx=4, pady=4)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Keyboard bindings for interactive segmentation
        self.bind("<Return>", self._commit_polygon)
        self.bind("<space>", self._commit_polygon)
        self.bind("<Escape>", self._cancel_polygon)

        # ─── Sidebar (right) ───
        sb = ctk.CTkFrame(body, width=self.SIDE_W, corner_radius=12, fg_color="#0d1117")
        sb.grid(row=0, column=1, sticky="nsew")
        sb.pack_propagate(False)

        # Sidebar header
        ctk.CTkLabel(
            sb, text="🏷️  Segment Review",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#58a6ff"
        ).pack(pady=(12, 4))

        # Scrollable review panel
        self.review_scroll = ctk.CTkScrollableFrame(
            sb, fg_color="transparent",
            scrollbar_button_color="#21262d"
        )
        self.review_scroll.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.hint_label = ctk.CTkLabel(
            self.review_scroll,
            text="Load an image to begin reviewing.\n\n"
                 "The pipeline will:\n"
                 "1. Run SAM2 for polygon segmentation\n"
                 "2. Run MTA model for class prediction\n"
                 "3. Present results for your review\n\n"
                 "Click any polygon to review it.",
            font=ctk.CTkFont(size=12), text_color="#484f58",
            wraplength=360, justify="center"
        )
        self.hint_label.pack(pady=40)

        # ─── Control bar (bottom) ───
        ctrl = ctk.CTkFrame(self, height=96, corner_radius=0, fg_color="#161b22")
        ctrl.pack(fill="x")
        ctrl.pack_propagate(False)

        # Top row: Data & Navigation
        row1 = ctk.CTkFrame(ctrl, fg_color="transparent")
        row1.pack(fill="x", padx=4, pady=(8, 4))
        
        # Bottom row: Pipeline Settings & Exports
        row2 = ctk.CTkFrame(ctrl, fg_color="transparent")
        row2.pack(fill="x", padx=4, pady=(0, 8))

        # ================= ROW 1 =================
        # Load buttons
        ctk.CTkButton(
            row1, text="📂 Load Image", width=120,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#238636", hover_color="#2ea043",
            command=self._load_image
        ).pack(side="left", padx=(8, 4))

        ctk.CTkButton(
            row1, text="📁 Load Folder", width=120,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1f6feb", hover_color="#388bfd",
            command=self._load_folder
        ).pack(side="left", padx=4)

        # Navigation
        self.btn_prev = ctk.CTkButton(
            row1, text="◀ Prev", width=80,
            font=ctk.CTkFont(size=12), fg_color="#21262d",
            hover_color="#30363d", command=self._prev_image,
            state="disabled"
        )
        self.btn_prev.pack(side="left", padx=4)

        self.btn_next = ctk.CTkButton(
            row1, text="Next ▶", width=80,
            font=ctk.CTkFont(size=12), fg_color="#21262d",
            hover_color="#30363d", command=self._next_image,
            state="disabled"
        )
        self.btn_next.pack(side="left", padx=4)

        self.queue_label = ctk.CTkLabel(
            row1, text="", font=ctk.CTkFont(size=11), text_color="#8b949e"
        )
        self.queue_label.pack(side="left", padx=8)

        # Export buttons (right side of row 1)
        ctk.CTkButton(
            row1, text="📊 Dashboard", width=100,
            font=ctk.CTkFont(size=12), fg_color="#8957e5",
            hover_color="#a371f7", command=self._show_dashboard
        ).pack(side="right", padx=4)

        ctk.CTkButton(
            row1, text="📤 Upload Roboflow", width=130,
            font=ctk.CTkFont(size=12), fg_color="#da3633",
            hover_color="#f85149", command=self._upload_to_roboflow
        ).pack(side="right", padx=4)

        ctk.CTkButton(
            row1, text="💾 Export Labels", width=110,
            font=ctk.CTkFont(size=12), fg_color="#0d6d6e",
            hover_color="#1a8d8e", command=self._export_xanylabeling
        ).pack(side="right", padx=4)

        if self.experimental_coco_ui:
            ctk.CTkButton(
                row1, text="🧪 Export COCO", width=110,
                font=ctk.CTkFont(size=12), fg_color="#e5a00d",
                hover_color="#e6b445", command=self._export_custom_coco
            ).pack(side="right", padx=4)

            self.yolo_target_var = ctk.StringVar(value="Material")
            ctk.CTkOptionMenu(
                row1, variable=self.yolo_target_var,
                values=["Material", "Object", "Contamination"],
                width=110, font=ctk.CTkFont(size=11),
                fg_color="#e5a00d", button_color="#cc8f0c"
            ).pack(side="right", padx=4)

            ctk.CTkLabel(row1, text="YOLO Target:", font=ctk.CTkFont(size=11), text_color="#8b949e").pack(side="right", padx=(12, 2))

        # ================= ROW 2 =================
        # Auto-approve button
        ctk.CTkButton(
            row2, text="⚡ Auto-Approve", width=120,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#9b6a00", hover_color="#bb8009",
            command=self._auto_approve
        ).pack(side="left", padx=(8, 4))

        # SAM2 toggle + model selector
        self.use_sam2_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            row2, text="SAM2", variable=self.use_sam2_var,
            font=ctk.CTkFont(size=11), text_color="#8b949e",
            onvalue=True, offvalue=False,
            command=self._process_current
        ).pack(side="left", padx=(12, 2))

        self.sam2_model_var = ctk.StringVar(value=SAM2_MODEL)
        self.sam2_dropdown = ctk.CTkOptionMenu(
            row2, variable=self.sam2_model_var,
            values=SAM2_AVAILABLE_MODELS, width=110,
            font=ctk.CTkFont(size=11),
            fg_color="#21262d", button_color="#30363d",
            command=self._on_sam2_model_change
        )
        self.sam2_dropdown.pack(side="left", padx=(0, 4))

        # Pipeline Mode dropdown
        self.pipeline_mode_var = ctk.StringVar(value="Attention First")
        ctk.CTkOptionMenu(
            row2, variable=self.pipeline_mode_var,
            values=["Attention First", "Classifier Only", "Ensemble (U then N)", "Boolean Mask Fusion / Consensus Segmentation"],
            width=160,
            font=ctk.CTkFont(size=11),
            fg_color="#21262d", button_color="#30363d",
            command=lambda _: self._process_current()
        ).pack(side="left", padx=(12, 4))

        # Precision Slider
        import config
        self.epsilon_var = ctk.DoubleVar(value=config.POLYGON_SIMPLIFY_EPSILON)
        ctk.CTkLabel(row2, text="Smoothing:", font=ctk.CTkFont(size=11), text_color="#8b949e").pack(side="left", padx=(12, 2))
        self.epsilon_slider = ctk.CTkSlider(
            row2, from_=0.1, to=10.0, variable=self.epsilon_var,
            width=80, height=12,
            command=self._on_epsilon_change
        )
        self.epsilon_slider.pack(side="left", padx=(0, 4))

        # Auto Upload Toggle
        self.auto_upload_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            row2, text="Auto Upload", variable=self.auto_upload_var,
            font=ctk.CTkFont(size=11), text_color="#8b949e",
            onvalue=True, offvalue=False
        ).pack(side="left", padx=(12, 4))

    # ── Pipeline Loading ───────────────────────────────────────────
    def _load_pipeline_async(self):
        def _load():
            try:
                from pipeline import MTAPipeline
                self.pipeline = MTAPipeline(use_sam2=True)
                # Pre-load MTA model
                _ = self.pipeline.mta
                msg = f"✅ Pipeline ready ({GPU_NAME})"
                self.after(0, lambda: self.status.configure(
                    text=msg, text_color="#7ee787"))
            except Exception as e:
                self.after(0, lambda: self.status.configure(
                    text=f"❌ Pipeline error: {e}", text_color="#f85149"))
        threading.Thread(target=_load, daemon=True).start()

    def _on_sam2_model_change(self, model_name: str):
        """Switch SAM2 model. Clears cached instance so next run uses the new one."""
        if self.pipeline is None:
            return

        self.pipeline._sam2_model_name = model_name
        self.pipeline._sam2 = None  # Clear cached SAM2 so it reloads

        self.status.configure(
            text=f"🔄 SAM2 switched to {model_name} — will load on next image",
            text_color="#d29922"
        )

    # ── Image Loading ──────────────────────────────────────────────
    def _auto_upload_current(self):
        """Upload current result to Roboflow if auto-upload is enabled."""
        if not self.auto_upload_var.get() or self.current_result is None:
            return
        
        segments = self.current_result.get("segments", [])
        if not segments:
            return

        from roboflow_uploader import RoboflowUploader
        if not RoboflowUploader.is_configured():
            return

        # Upload in background so UI doesn't stutter during navigation
        res_copy = {**self.current_result}
        def _bg_upload():
            try:
                uploader = RoboflowUploader()
                uploader.upload_annotation(
                    res_copy["image_path"],
                    res_copy["segments"],
                    is_prediction=False,
                )
            except Exception as e:
                print(f"[Auto Upload] Failed: {e}")

        threading.Thread(target=_bg_upload, daemon=True).start()

    def _load_image(self):
        if self.pipeline is None:
            self.status.configure(text="⏳ Pipeline loading…", text_color="#d29922")
            return

        self._auto_upload_current()

        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return

        self.image_queue = [path]
        self.queue_index = 0
        self._process_current()

    def _load_folder(self):
        if self.pipeline is None:
            self.status.configure(text="⏳ Pipeline loading…", text_color="#d29922")
            return

        self._auto_upload_current()

        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return

        folder = Path(folder)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_queue = sorted([
            str(f) for f in folder.iterdir() if f.suffix.lower() in exts
        ])

        if not self.image_queue:
            self.status.configure(text="❌ No images found", text_color="#f85149")
            return

        self.queue_index = 0
        self._update_nav_buttons()
        self._process_current()

    # ── Navigation ─────────────────────────────────────────────────
    def _prev_image(self):
        if self.queue_index > 0:
            self._auto_upload_current()
            self.queue_index -= 1
            self._update_nav_buttons()
            self._process_current()

    def _next_image(self):
        if self.queue_index < len(self.image_queue) - 1:
            self._auto_upload_current()
            self.queue_index += 1
            self._update_nav_buttons()
            self._process_current()

    def _update_nav_buttons(self):
        self.btn_prev.configure(
            state="normal" if self.queue_index > 0 else "disabled"
        )
        self.btn_next.configure(
            state="normal" if self.queue_index < len(self.image_queue) - 1 else "disabled"
        )
        if self.image_queue:
            self.queue_label.configure(
                text=f"Image {self.queue_index + 1}/{len(self.image_queue)}"
            )

    # ── Processing ─────────────────────────────────────────────────
    def _process_current(self):
        if not self.image_queue:
            return

        image_path = self.image_queue[self.queue_index]
        name = Path(image_path).name
        self.status.configure(text=f"⏳ Processing {name}…", text_color="#d29922")

        def _do():
            try:
                use_sam2 = self.use_sam2_var.get()
                pipeline_mode = self.pipeline_mode_var.get()
                result = self.pipeline.process_frame(
                    cv2.imread(image_path),
                    image_path=image_path,
                    use_sam2_override=use_sam2,
                    pipeline_mode=pipeline_mode,
                )
                self.after(0, lambda: self._on_processed(result))
            except Exception as e:
                self.after(0, lambda: self.status.configure(
                    text=f"❌ Error: {e}", text_color="#f85149"))

        threading.Thread(target=_do, daemon=True).start()

    def _on_processed(self, result):
        self.current_result = result
        self.selected_segment = None

        n = len(result.get("segments", []))
        name = Path(result["image_path"]).name
        t = result.get("processing_time", 0.0)
        self.status.configure(
            text=f"🖼 {name} — {n} segments ({t:.2f}s)", text_color="#7ee787"
        )

        # Reset zoom
        self._zoom_factor = 1.0
        self._pan_x = 0
        self._pan_y = 0

        self._render_canvas()
        self._show_segments_list()
        self._update_nav_buttons()

    # ── Canvas Rendering ───────────────────────────────────────────
    def _render_canvas(self):
        if self.current_result is None:
            return

        annotated = self.current_result["annotated_image"]
        if annotated is None:
            return

        # Highlight selected segment
        if self.selected_segment is not None:
            annotated = annotated.copy()
            seg = self.current_result["segments"][self.selected_segment]
            polygon = seg.get("polygon", [])
            if polygon:
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)

        # Draw interactive SAM2 preview polygon
        if self.active_preview_seg and "polygon" in self.active_preview_seg:
            annotated = annotated.copy()
            pts = np.array(self.active_preview_seg["polygon"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], True, (255, 255, 0), 2)  # Cyan (BGR)

        # Draw interactive points
        if self.active_points:
            if annotated is self.current_result["annotated_image"]:
                annotated = annotated.copy()
            for pt, lbl in zip(self.active_points, self.active_labels):
                color = (0, 255, 0) if lbl == 1 else (0, 0, 255) # Green (+), Red (-)
                cv2.circle(annotated, (pt[0], pt[1]), 5, color, -1)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        self.canvas.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        base_s = min(cw / w, ch / h)
        s = base_s * self._zoom_factor
        nw, nh = int(w * s), int(h * s)

        self._scale = s
        self._ox = (cw - nw) // 2 + self._pan_x
        self._oy = (ch - nh) // 2 + self._pan_y

        img = Image.fromarray(cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self._ox, self._oy, anchor="nw", image=self._photo)

    # ── Sidebar: Segments List ─────────────────────────────────────
    def _show_segments_list(self):
        for w in self.review_scroll.winfo_children():
            w.destroy()

        if self.current_result is None:
            return

        segments = self.current_result.get("segments", [])
        if not segments:
            ctk.CTkLabel(
                self.review_scroll,
                text="No segments detected.\nTry lowering confidence or\nloading a different image.",
                font=ctk.CTkFont(size=12), text_color="#484f58",
                wraplength=340, justify="center"
            ).pack(pady=40)
            return

        # Summary card
        summary = ctk.CTkFrame(self.review_scroll, corner_radius=10, fg_color="#21262d")
        summary.pack(fill="x", padx=4, pady=(4, 8))

        n_uncertain = sum(1 for s in segments if s.get("is_uncertain", False))
        ctk.CTkLabel(
            summary,
            text=f"📊  {len(segments)} segments  |  ⚠️ {n_uncertain} uncertain",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#c9d1d9"
        ).pack(padx=12, pady=8)

        # Segment rows
        for i, seg in enumerate(segments):
            self._create_segment_row(i, seg)

    def _create_segment_row(self, index: int, seg: Dict):
        """Create a clickable segment row in the sidebar."""
        is_uncertain = seg.get("is_uncertain", False)
        confidence = seg.get("confidence", 0.0)
        class_name = seg.get("predicted_class", "?")
        color = seg.get("color", (128, 128, 128))
        r, g, b = color
        hex_col = f"#{r:02x}{g:02x}{b:02x}"

        bg = "#2d1b1b" if is_uncertain else "#1c2128"
        row = ctk.CTkFrame(
            self.review_scroll, corner_radius=8,
            fg_color=bg, height=50, cursor="hand2"
        )
        row.pack(fill="x", pady=2, padx=4)
        row.pack_propagate(False)
        row.bind("<Button-1>", lambda e, i=index: self._select_segment(i))

        # Color swatch
        swatch = ctk.CTkFrame(row, width=8, corner_radius=4, fg_color=hex_col)
        swatch.pack(side="left", fill="y", padx=(6, 8), pady=6)

        # Segment info
        info_frame = ctk.CTkFrame(row, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, pady=4)
        info_frame.bind("<Button-1>", lambda e, i=index: self._select_segment(i))

        lbl = ctk.CTkLabel(
            info_frame, text=f"#{index + 1}  {class_name}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#c9d1d9"
        )
        lbl.pack(anchor="w")
        lbl.bind("<Button-1>", lambda e, i=index: self._select_segment(i))

        conf_text = f"{'⚠️ ' if is_uncertain else ''}{confidence:.1%}"
        conf_color = "#f85149" if is_uncertain else "#7ee787"
        sub = ctk.CTkLabel(
            info_frame, text=conf_text,
            font=ctk.CTkFont(size=11), text_color=conf_color
        )
        sub.pack(anchor="w")
        sub.bind("<Button-1>", lambda e, i=index: self._select_segment(i))

        # Quick action buttons
        btn_frame = ctk.CTkFrame(row, fg_color="transparent")
        btn_frame.pack(side="right", padx=6, pady=4)

        ctk.CTkButton(
            btn_frame, text="✅", width=32, height=32,
            font=ctk.CTkFont(size=14), fg_color="#238636",
            hover_color="#2ea043",
            command=lambda i=index: self._quick_approve(i)
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            btn_frame, text="❌", width=32, height=32,
            font=ctk.CTkFont(size=14), fg_color="#da3633",
            hover_color="#f85149",
            command=lambda i=index: self._quick_reject(i)
        ).pack(side="left", padx=2)

    # ── Sidebar: Segment Detail ────────────────────────────────────
    def _select_segment(self, index: int):
        """Show detailed review panel for a segment."""
        self.selected_segment = index
        self._render_canvas()

        for w in self.review_scroll.winfo_children():
            w.destroy()

        if self.current_result is None:
            return

        seg = self.current_result["segments"][index]
        confidence = seg.get("confidence", 0.0)
        class_name = seg.get("predicted_class", "?")
        color = seg.get("color", (128, 128, 128))
        r, g, b = color
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        pred_id = seg.get("prediction_id")

        # Back button
        ctk.CTkButton(
            self.review_scroll, text="← Back to all segments",
            font=ctk.CTkFont(size=12), height=30,
            fg_color="#21262d", hover_color="#30363d",
            command=self._deselect_segment
        ).pack(fill="x", padx=4, pady=(4, 8))

        # ─── Header card ───
        hdr = ctk.CTkFrame(self.review_scroll, corner_radius=10, fg_color="#21262d")
        hdr.pack(fill="x", padx=4, pady=(0, 6))

        ctk.CTkLabel(
            hdr, text=f"🔍  Segment #{index + 1}",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#58a6ff"
        ).pack(anchor="w", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            hdr, text=f"Predicted: {class_name}  •  Conf: {confidence:.1%}",
            font=ctk.CTkFont(size=13), text_color=hex_col
        ).pack(anchor="w", padx=12, pady=(0, 8))

        # ─── Polygon Review ───
        sep1 = ctk.CTkFrame(self.review_scroll, height=2, fg_color="#21262d")
        sep1.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(
            self.review_scroll, text="📐  Polygon Quality",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#c9d1d9"
        ).pack(anchor="w", padx=10, pady=(6, 4))

        ctk.CTkLabel(
            self.review_scroll,
            text="Is the polygon boundary correctly tracing the object?",
            font=ctk.CTkFont(size=11), text_color="#8b949e",
            wraplength=360, justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 6))

        poly_btns = ctk.CTkFrame(self.review_scroll, fg_color="transparent")
        poly_btns.pack(fill="x", padx=10, pady=(0, 8))

        self.poly_correct_btn = ctk.CTkButton(
            poly_btns, text=f"✅ Correct ({REWARD_POLYGON_CORRECT:+d})", width=110, height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#238636", hover_color="#2ea043",
            command=lambda: self._review_polygon(True)
        )
        self.poly_correct_btn.pack(side="left", padx=(0, 4))

        self.poly_edit_btn = ctk.CTkButton(
            poly_btns, text=f"✏️ Edit ({REWARD_POLYGON_INCORRECT:+d})", width=105, height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#9b6a00", hover_color="#bb8009",
            command=self._start_polygon_edit
        )
        self.poly_edit_btn.pack(side="left", padx=(0, 4))

        self.poly_reject_btn = ctk.CTkButton(
            poly_btns, text=f"❌ Wrong ({REWARD_POLYGON_INCORRECT:+d})", width=110, height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#da3633", hover_color="#f85149",
            command=lambda: self._review_polygon(False)
        )
        self.poly_reject_btn.pack(side="left")

        # ─── Class Review ───
        sep2 = ctk.CTkFrame(self.review_scroll, height=2, fg_color="#21262d")
        sep2.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(
            self.review_scroll, text="🏷️  Class Prediction",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#c9d1d9"
        ).pack(anchor="w", padx=10, pady=(6, 4))

        # Show predicted class output
        pred_card = ctk.CTkFrame(self.review_scroll, corner_radius=8, fg_color="#161b22")
        pred_card.pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(
            pred_card, text=f"Model predicts: {class_name}",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=hex_col
        ).pack(anchor="w", padx=12, pady=(8, 4))

        ctk.CTkLabel(
            pred_card, text=f"Confidence: {confidence:.1%}",
            font=ctk.CTkFont(size=12), text_color="#8b949e"
        ).pack(anchor="w", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            self.review_scroll,
            text="Is this class prediction correct?",
            font=ctk.CTkFont(size=11), text_color="#8b949e",
            wraplength=360
        ).pack(anchor="w", padx=14, pady=(4, 6))

        cls_btns = ctk.CTkFrame(self.review_scroll, fg_color="transparent")
        cls_btns.pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkButton(
            cls_btns, text=f"✅ Class Correct ({REWARD_CLASS_CORRECT:+d})", width=170, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#238636", hover_color="#2ea043",
            command=lambda: self._review_class(True)
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            cls_btns, text=f"✏️ Edit Class ({REWARD_CLASS_INCORRECT:+d})", width=170, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#9b6a00", hover_color="#bb8009",
            command=self._show_class_editor
        ).pack(side="left")

        # ─── Experimental COCO Attributes ───
        if self.experimental_coco_ui and class_name.startswith("m-"):
            sep_exp = ctk.CTkFrame(self.review_scroll, height=2, fg_color="#21262d")
            sep_exp.pack(fill="x", padx=6, pady=4)

            ctk.CTkLabel(
                self.review_scroll, text="🧪  Custom COCO Attributes",
                font=ctk.CTkFont(size=14, weight="bold"), text_color="#e5a00d"
            ).pack(anchor="w", padx=10, pady=(6, 4))

            attr_frame = ctk.CTkFrame(self.review_scroll, fg_color="transparent")
            attr_frame.pack(fill="x", padx=14, pady=4)

            if "custom_attributes" not in seg:
                seg["custom_attributes"] = {
                    "material": class_name[2:],
                    "object": "unknown",
                    "contamination": "null"
                }

            attrs = seg["custom_attributes"]

            # Material Dropdown
            mat_frame = ctk.CTkFrame(attr_frame, fg_color="transparent")
            mat_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(mat_frame, text="Material:", width=80, anchor="w").pack(side="left")
            mat_var = ctk.StringVar(value=attrs["material"])
            def _on_mat_change(val, s=seg): s["custom_attributes"]["material"] = val
            ctk.CTkOptionMenu(mat_frame, variable=mat_var, values=MATERIAL_CLASSES, command=_on_mat_change).pack(side="left", fill="x", expand=True)

            # Object Dropdown
            obj_frame = ctk.CTkFrame(attr_frame, fg_color="transparent")
            obj_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(obj_frame, text="Object:", width=80, anchor="w").pack(side="left")
            obj_var = ctk.StringVar(value=attrs["object"])
            def _on_obj_change(val, s=seg): s["custom_attributes"]["object"] = val
            ctk.CTkOptionMenu(obj_frame, variable=obj_var, values=OBJECT_CLASSES, command=_on_obj_change).pack(side="left", fill="x", expand=True)

            # Contamination Dropdown
            cont_frame = ctk.CTkFrame(attr_frame, fg_color="transparent")
            cont_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(cont_frame, text="Contamination:", width=90, anchor="w").pack(side="left")
            cont_var = ctk.StringVar(value=str(attrs["contamination"]))
            def _on_cont_change(val, s=seg): s["custom_attributes"]["contamination"] = val
            ctk.CTkOptionMenu(cont_frame, variable=cont_var, values=CONTAMINATION_CLASSES, command=_on_cont_change).pack(side="left", fill="x", expand=True)

        # ─── Top-5 Alternative Classes ───
        sep3 = ctk.CTkFrame(self.review_scroll, height=2, fg_color="#21262d")
        sep3.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(
            self.review_scroll, text="📊  All Class Scores",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#8b949e"
        ).pack(anchor="w", padx=10, pady=(6, 4))

        all_scores = seg.get("all_scores", [])
        for score in all_scores[:10]:
            sc = score["confidence"]
            if sc <= 0:
                continue
            sname = score["class_name"]
            cid = score["class_id"]
            sc_color = CLASS_COLORS[cid % len(CLASS_COLORS)]
            sr, sg_, sb = sc_color
            shex = f"#{sr:02x}{sg_:02x}{sb:02x}"

            score_row = ctk.CTkFrame(
                self.review_scroll, corner_radius=6,
                fg_color="#161b22", height=28
            )
            score_row.pack(fill="x", padx=10, pady=1)
            score_row.pack_propagate(False)

            ctk.CTkFrame(
                score_row, width=4, corner_radius=2, fg_color=shex
            ).pack(side="left", fill="y", padx=(4, 6), pady=4)

            ctk.CTkLabel(
                score_row, text=sname,
                font=ctk.CTkFont(size=11), text_color="#c9d1d9"
            ).pack(side="left", padx=(0, 4))

            ctk.CTkLabel(
                score_row, text=f"{sc:.1%}",
                font=ctk.CTkFont(size=11, weight="bold"), text_color=shex,
                width=40
            ).pack(side="right", padx=6)

    def _deselect_segment(self):
        self.selected_segment = None
        self._render_canvas()
        self._show_segments_list()

    # ── Review Actions ─────────────────────────────────────────────
    def _review_polygon(self, is_correct: bool):
        if self.selected_segment is None or self.current_result is None:
            return

        seg = self.current_result["segments"][self.selected_segment]
        pred_id = seg.get("prediction_id")
        if pred_id is None:
            return

        reward = self.pipeline.active_learning.record_polygon_feedback(
            pred_id, is_correct
        )
        self.session_rewards += reward

        if is_correct:
            # Polygon accepted
            self.status.configure(
                text=f"✅ Polygon accepted! ({REWARD_POLYGON_CORRECT:+d})", text_color="#7ee787"
            )
            # Disable polygon buttons
            self.poly_correct_btn.configure(state="disabled")
            self.poly_edit_btn.configure(state="disabled")
            self.poly_reject_btn.configure(state="disabled")
        else:
            # Polygon rejected — remove it from the list and re-render
            removed_index = self.selected_segment
            self.current_result["segments"].pop(removed_index)
            self.selected_segment = None

            # Re-draw the annotated image without the removed polygon
            self.current_result["annotated_image"] = self.pipeline._draw_annotations(
                self.current_result["image"],
                self.current_result["segments"],
            )

            n = len(self.current_result["segments"])
            self.status.configure(
                text=f"❌ Polygon removed ({REWARD_POLYGON_INCORRECT:+d}). {n} segments remaining.",
                text_color="#f85149"
            )

            # Refresh canvas and sidebar
            self._render_canvas()
            self._show_segments_list()

        self._update_reward_display()

    def _start_polygon_edit(self):
        if self.selected_segment is None or self.current_result is None:
            return
            
        seg = self.current_result["segments"][self.selected_segment]
        pred_id = seg.get("prediction_id")
        if pred_id is not None:
            reward = self.pipeline.active_learning.record_polygon_feedback(pred_id, False)
            self.session_rewards += reward
            self._update_reward_display()
            
        self.editing_segment_index = self.selected_segment
        self.editing_bbox = seg.get("bbox")
        self.active_points = []
        self.active_labels = []
        self.active_preview_seg = seg
        self._render_canvas()
        
        self.status.configure(
            text="✏️ Edit Mode: Click on the object to draw a new polygon. Press Enter when done.",
            text_color="#58a6ff"
        )
        
        # Disable buttons while editing
        if hasattr(self, 'poly_correct_btn') and self.poly_correct_btn.winfo_exists():
            self.poly_correct_btn.configure(state="disabled")
            self.poly_edit_btn.configure(state="disabled")
            self.poly_reject_btn.configure(state="disabled")

    def _review_class(self, is_correct: bool, corrected: str = None, corrected_id: int = None):
        if self.selected_segment is None or self.current_result is None:
            return

        seg = self.current_result["segments"][self.selected_segment]
        pred_id = seg.get("prediction_id")
        if pred_id is None:
            return

        reward = self.pipeline.active_learning.record_class_feedback(
            pred_id, is_correct,
            corrected_class=corrected,
            corrected_class_id=corrected_id,
        )
        self.session_rewards += reward
        self.session_reviewed += 1
        if is_correct:
            self.session_correct += 1
            msg = f"✅ Class confirmed! ({REWARD_CLASS_CORRECT:+d})"
            color = "#7ee787"
        else:
            msg = f"✏️ Corrected to '{corrected}' ({REWARD_CLASS_INCORRECT:+d})"
            color = "#d29922"
            
            # Update the segment data in memory!
            seg["predicted_class"] = corrected
            seg["predicted_class_id"] = corrected_id
            seg["color"] = CLASS_COLORS[corrected_id % len(CLASS_COLORS)]
            
            # Re-draw the annotated image to show new class and color
            self.current_result["annotated_image"] = self.pipeline._draw_annotations(
                self.current_result["image"],
                self.current_result["segments"],
            )
            # Refresh view
            self._render_canvas()
            self._show_segments_list()

        self.status.configure(text=msg, text_color=color)
        self._update_reward_display()

        # Check retrain
        if self.pipeline.active_learning.should_retrain():
            self.status.configure(
                text="🔄 Retrain threshold reached! Consider retraining.",
                text_color="#d29922"
            )

    def _show_class_editor(self):
        """Show dropdown to select correct class."""
        if self.selected_segment is None:
            return

        # Create correction popup
        popup = ctk.CTkToplevel(self)
        popup.title("Edit Class")
        popup.geometry("350x200")
        popup.attributes("-topmost", True)
        popup.configure(fg_color="#161b22")

        ctk.CTkLabel(
            popup, text="Select the correct class:",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#c9d1d9"
        ).pack(pady=(16, 8))

        class_var = ctk.StringVar(value=CLASS_NAMES[0])
        dropdown = ctk.CTkOptionMenu(
            popup, variable=class_var, values=CLASS_NAMES,
            width=280, fg_color="#21262d"
        )
        dropdown.pack(pady=8)

        def _confirm():
            selected = class_var.get()
            selected_id = CLASS_NAMES.index(selected) if selected in CLASS_NAMES else 0
            self._review_class(False, corrected=selected, corrected_id=selected_id)
            popup.destroy()

        ctk.CTkButton(
            popup, text="✅ Confirm Correction", width=200, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#238636", hover_color="#2ea043",
            command=_confirm
        ).pack(pady=12)

    def _quick_approve(self, index: int):
        """Quick approve both polygon and class for a segment."""
        seg = self.current_result["segments"][index]
        pred_id = seg.get("prediction_id")
        if pred_id is None:
            return

        result = self.pipeline.approve_segment(
            pred_id, polygon_correct=True, class_correct=True
        )
        self.session_rewards += result["total_reward"]
        self.session_reviewed += 1
        self.session_correct += 1

        self.status.configure(
            text=f"✅ Segment #{index + 1} approved ({result['total_reward']:+d})",
            text_color="#7ee787"
        )
        self._update_reward_display()

    def _quick_reject(self, index: int):
        """Quick reject polygon — remove it from the list."""
        if index >= len(self.current_result["segments"]):
            return
        seg = self.current_result["segments"][index]
        pred_id = seg.get("prediction_id")
        if pred_id is not None:
            self.pipeline.active_learning.record_polygon_feedback(pred_id, False)
        self.session_rewards -= 1

        # Remove from segments list
        self.current_result["segments"].pop(index)

        # Re-draw annotated image without the removed polygon
        self.current_result["annotated_image"] = self.pipeline._draw_annotations(
            self.current_result["image"],
            self.current_result["segments"],
        )

        # Clear selection if it was the removed one
        self.selected_segment = None

        n = len(self.current_result["segments"])
        self.status.configure(
            text=f"❌ Segment removed ({REWARD_POLYGON_INCORRECT:+d}). {n} remaining.",
            text_color="#f85149"
        )

        # Refresh view
        self._render_canvas()
        self._show_segments_list()
        self._update_reward_display()

    def _auto_approve(self):
        """Auto-approve high-confidence segments."""
        if self.current_result is None:
            return

        result = self.pipeline.auto_approve_high_confidence(
            self.current_result["segments"], AUTO_APPROVE_THRESHOLD
        )
        approved = result["auto_approved"]
        pending = result["pending"]

        self.session_rewards += approved * REWARD_AUTO_APPROVE
        self.session_reviewed += approved
        self.session_correct += approved

        self.status.configure(
            text=f"⚡ Auto-approved {approved} segments (≥{AUTO_APPROVE_THRESHOLD:.0%}), "
                 f"{pending} need manual review",
            text_color="#d29922"
        )
        self._update_reward_display()

    # ── Reward Display ─────────────────────────────────────────────
    def _update_reward_display(self):
        total_segs = len(self.current_result["segments"]) if self.current_result else 0
        acc = (self.session_correct / max(self.session_reviewed, 1)) * 100

        self.reward_display.configure(
            text=f"🏆 Rewards: {self.session_rewards:+d}  |  "
                 f"Reviewed: {self.session_reviewed}  |  "
                 f"Accuracy: {acc:.0f}%"
        )

    # ── Export Actions ─────────────────────────────────────────────
    def _export_xanylabeling(self):
        if self.current_result is None:
            return

        result = self.current_result
        self.pipeline.export_approved(
            image_path=result["image_path"],
            segments=result["segments"],
            to_roboflow=False,
            to_xanylabeling=True,
        )
        self.status.configure(
            text=f"💾 Exported to X-AnyLabeling: {ANNOTATIONS_DIR}",
            text_color="#7ee787"
        )

    def _export_custom_coco(self):
        if self.current_result is None:
            return

        result = self.current_result
        yolo_target = getattr(self, 'yolo_target_var', None)
        target_val = yolo_target.get() if yolo_target else "Material"
        
        self.pipeline.export_approved(
            image_path=result["image_path"],
            segments=result["segments"],
            to_roboflow=False,
            to_xanylabeling=False,
            to_custom_coco=True,
            yolo_target=target_val
        )
        self.status.configure(
            text=f"🧪 Exported Custom COCO & YOLO: {ANNOTATIONS_DIR}",
            text_color="#7ee787"
        )

    def _upload_to_roboflow(self):
        if self.current_result is None:
            return

        from roboflow_uploader import RoboflowUploader
        if not RoboflowUploader.is_configured():
            self.status.configure(
                text="❌ Roboflow not configured. Set ROBOFLOW_API_KEY env var.",
                text_color="#f85149"
            )
            return

        def _do():
            result = self.pipeline.export_approved(
                image_path=self.current_result["image_path"],
                segments=self.current_result["segments"],
                to_roboflow=True,
                to_xanylabeling=False,
            )
            ok = result.get("roboflow_uploaded", False)
            msg = "✅ Uploaded to Roboflow!" if ok else "❌ Roboflow upload failed"
            color = "#7ee787" if ok else "#f85149"
            self.after(0, lambda: self.status.configure(text=msg, text_color=color))

        threading.Thread(target=_do, daemon=True).start()
        self.status.configure(text="📤 Uploading to Roboflow…", text_color="#d29922")

    # ── Dashboard ──────────────────────────────────────────────────
    def _show_dashboard(self):
        """Show active learning dashboard in a popup window."""
        summary = self.pipeline.active_learning.get_reward_summary()
        history = self.pipeline.active_learning.get_session_stats(20)

        dash = ctk.CTkToplevel(self)
        dash.title("📊 Active Learning Dashboard")
        dash.geometry("700x600")
        dash.configure(fg_color="#0d1117")
        dash.attributes("-topmost", True)

        # Title
        ctk.CTkLabel(
            dash, text="📊  Active Learning Dashboard",
            font=ctk.CTkFont(size=20, weight="bold"), text_color="#58a6ff"
        ).pack(pady=(16, 8))

        scroll = ctk.CTkScrollableFrame(dash, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # ─── Overall metrics ───
        metrics = ctk.CTkFrame(scroll, corner_radius=10, fg_color="#161b22")
        metrics.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            metrics, text="📈  Overall Metrics",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#c9d1d9"
        ).pack(anchor="w", padx=12, pady=(10, 4))

        stats_text = (
            f"Total predictions: {summary['total_predictions']}\n"
            f"Total reviewed: {summary['total_reviewed']}\n"
            f"─────────────────────────\n"
            f"Polygon accuracy: {summary['polygon_accuracy']:.1%} "
            f"({summary['polygon_correct']}✅ / {summary['polygon_incorrect']}❌)\n"
            f"Class accuracy: {summary['class_accuracy']:.1%} "
            f"({summary['class_correct']}✅ / {summary['class_incorrect']}❌)\n"
            f"─────────────────────────\n"
            f"Total reward: {summary['total_reward']:+d}\n"
            f"Corrections pending: {self.pipeline.active_learning.get_corrections_count()}"
        )

        ctk.CTkLabel(
            metrics, text=stats_text,
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color="#8b949e", justify="left"
        ).pack(anchor="w", padx=16, pady=(0, 10))

        # Retrain indicator
        should_retrain = self.pipeline.active_learning.should_retrain()
        retrain_color = "#f85149" if should_retrain else "#7ee787"
        retrain_text = ("🔄 RETRAIN RECOMMENDED — Enough corrections accumulated"
                       if should_retrain else "✅ Model performance acceptable")
        ctk.CTkLabel(
            metrics, text=retrain_text,
            font=ctk.CTkFont(size=12, weight="bold"), text_color=retrain_color
        ).pack(anchor="w", padx=12, pady=(0, 10))

        # ─── Per-class accuracy ───
        if summary.get("per_class"):
            cls_frame = ctk.CTkFrame(scroll, corner_radius=10, fg_color="#161b22")
            cls_frame.pack(fill="x", pady=(0, 8))

            ctk.CTkLabel(
                cls_frame, text="🏷️  Per-Class Accuracy",
                font=ctk.CTkFont(size=15, weight="bold"), text_color="#c9d1d9"
            ).pack(anchor="w", padx=12, pady=(10, 6))

            for pc in summary["per_class"]:
                row = ctk.CTkFrame(cls_frame, corner_radius=6, fg_color="#21262d", height=30)
                row.pack(fill="x", padx=8, pady=1)
                row.pack_propagate(False)

                cid = CLASS_NAMES.index(pc["class_name"]) if pc["class_name"] in CLASS_NAMES else 0
                cr, cg, cb = CLASS_COLORS[cid % len(CLASS_COLORS)]
                chex = f"#{cr:02x}{cg:02x}{cb:02x}"

                ctk.CTkFrame(
                    row, width=4, corner_radius=2, fg_color=chex
                ).pack(side="left", fill="y", padx=(4, 6), pady=4)

                ctk.CTkLabel(
                    row, text=pc["class_name"],
                    font=ctk.CTkFont(size=11), text_color="#c9d1d9"
                ).pack(side="left", padx=(0, 4))

                acc_color = "#7ee787" if pc["accuracy"] >= 0.7 else "#f85149"
                ctk.CTkLabel(
                    row, text=f"{pc['accuracy']:.0%} ({pc['correct']}/{pc['total']})",
                    font=ctk.CTkFont(size=11, weight="bold"), text_color=acc_color,
                    width=80
                ).pack(side="right", padx=6)

        # ─── Recent history ───
        if history:
            hist_frame = ctk.CTkFrame(scroll, corner_radius=10, fg_color="#161b22")
            hist_frame.pack(fill="x", pady=(0, 8))

            ctk.CTkLabel(
                hist_frame, text="📜  Recent Activity",
                font=ctk.CTkFont(size=15, weight="bold"), text_color="#c9d1d9"
            ).pack(anchor="w", padx=12, pady=(10, 6))

            for h in history:
                emoji = "✅" if h["reward"] > 0 else "❌"
                text = f"{emoji} {h['type']}: {h['class']} ({h['reward']:+d})"
                ctk.CTkLabel(
                    hist_frame, text=text,
                    font=ctk.CTkFont(size=11), text_color="#8b949e"
                ).pack(anchor="w", padx=16, pady=1)

        # ─── Export corrections button ───
        ctk.CTkButton(
            scroll, text="📦 Export Corrections for Retraining",
            font=ctk.CTkFont(size=13, weight="bold"), height=38,
            fg_color="#8957e5", hover_color="#a371f7",
            command=self._export_corrections
        ).pack(fill="x", padx=4, pady=8)

    def _export_corrections(self):
        out = self.pipeline.active_learning.export_corrections_for_training()
        self.status.configure(
            text=f"📦 Corrections exported to: {out}",
            text_color="#7ee787"
        )

    # ── Canvas Interaction ─────────────────────────────────────────
    def _on_mousewheel(self, ev):
        old_zoom = self._zoom_factor
        if getattr(ev, 'delta', 0) > 0:
            self._zoom_factor *= 1.1
        elif getattr(ev, 'delta', 0) < 0:
            self._zoom_factor /= 1.1
        self._zoom_factor = max(1.0, min(self._zoom_factor, 10.0))
        if self._zoom_factor != old_zoom:
            self._render_canvas()

    def _on_press(self, ev):
        self._drag_start_x = ev.x
        self._drag_start_y = ev.y
        self._is_dragging = False

    def _on_drag(self, ev):
        dx = ev.x - self._drag_start_x
        dy = ev.y - self._drag_start_y
        if abs(dx) > 2 or abs(dy) > 2:
            self._is_dragging = True
            self._pan_x += dx
            self._pan_y += dy
            self._drag_start_x = ev.x
            self._drag_start_y = ev.y
            self._render_canvas()

    def _on_release(self, ev):
        if not self._is_dragging:
            self._on_click(ev)
        self._is_dragging = False

    def _on_click(self, ev):
        """Left click: Select segment OR Add Positive Point for interactive SAM2."""
        if self.current_result is None:
            return

        fx = (ev.x - self._ox) / self._scale
        fy = (ev.y - self._oy) / self._scale

        # If we are ALREADY in interactive mode OR editing a segment, any left click adds a positive point
        if self.active_points or getattr(self, 'editing_segment_index', None) is not None:
            self._add_interactive_point(fx, fy, 1)
            return

        segments = self.current_result.get("segments", [])
        
        # Find smallest bbox containing click
        best_idx, best_area = None, float("inf")
        for i, seg in enumerate(segments):
            bbox = seg.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_idx, best_area = i, area

        if best_idx is not None:
            self._select_segment(best_idx)
        else:
            # Clicked empty space — start interactive mode!
            if not self.use_sam2_var.get() or self.pipeline.sam2 is None:
                self.status.configure(
                    text="❌ Enable SAM2 to click-and-add missing objects.",
                    text_color="#d29922"
                )
                return
            self._add_interactive_point(fx, fy, 1)

    def _on_right_click(self, ev):
        """Right click: Add Negative Point for interactive SAM2."""
        if self.current_result is None:
            return
            
        if not self.active_points and getattr(self, 'editing_segment_index', None) is None:
            return
        
        fx = (ev.x - self._ox) / self._scale
        fy = (ev.y - self._oy) / self._scale
        self._add_interactive_point(fx, fy, 0)

    def _add_interactive_point(self, fx, fy, label):
        """Appends a point and runs SAM2 preview."""
        self.active_points.append([int(fx), int(fy)])
        self.active_labels.append(label)

        self.status.configure(text="⏳ Generating polygon preview...", text_color="#c9d1d9")
        
        def _preview():
            try:
                img = self.current_result["image"]
                bbox = getattr(self, 'editing_bbox', None)
                preview_seg = self.pipeline.preview_segment_by_points(
                    img, self.active_points, self.active_labels, bbox=bbox
                )
                
                if preview_seg:
                    self.active_preview_seg = preview_seg
                    self.after(0, lambda: self.status.configure(
                        text="✨ Preview ready! Press 'Enter' to confirm or click to refine.",
                        text_color="#58a6ff"
                    ))
                else:
                    # The new point broke the prediction! Revert it.
                    if self.active_points:
                        self.active_points.pop()
                        self.active_labels.pop()
                    self.after(0, lambda: self.status.configure(
                        text="⚠️ Point ignored: Caused SAM2 to lose the object.", text_color="#d29922"
                    ))
                self.after(0, self._render_canvas)
            except Exception as e:
                # Revert point on error
                if self.active_points:
                    self.active_points.pop()
                    self.active_labels.pop()
                err_msg = f"❌ Error updating preview: {e}"
                self.after(0, lambda msg=err_msg: self.status.configure(
                    text=msg, text_color="#f85149"
                ))

        threading.Thread(target=_preview, daemon=True).start()

    def _on_epsilon_change(self, val):
        import config
        config.POLYGON_SIMPLIFY_EPSILON = float(val)
        
        # Real-time update polygons if result exists
        if self.current_result and "segments" in self.current_result:
            from sam2_segmentor import SAM2Segmentor
            for seg in self.current_result["segments"]:
                if "mask" in seg:
                    new_poly = SAM2Segmentor.mask_to_polygon(seg["mask"])
                    if new_poly is not None:
                        seg["polygon"] = new_poly
            
            # Re-draw annotations
            if self.pipeline and hasattr(self.pipeline, '_draw_annotations'):
                self.current_result["annotated_image"] = self.pipeline._draw_annotations(
                    self.current_result["image"],
                    self.current_result["segments"],
                )
            
            self._render_canvas()

    def _commit_polygon(self, ev=None):
        """Space/Enter: Commits the active interactive polygon."""
        if not self.active_preview_seg or self.current_result is None:
            return

        self.status.configure(text="⏳ Committing polygon...", text_color="#c9d1d9")

        # Take a snapshot of the state
        preview_seg = self.active_preview_seg
        img = self.current_result["image"]
        img_path = self.current_result["image_path"]

        # Clear state immediately so UI feels responsive
        self.active_points = []
        self.active_labels = []
        self.active_preview_seg = None
        self.editing_bbox = None
        self._render_canvas()

        def _commit():
            try:
                new_item = self.pipeline.commit_preview_segment(img, img_path, preview_seg)
                
                if getattr(self, 'editing_segment_index', None) is not None:
                    idx = self.editing_segment_index
                    # Preserve properties of the old segment
                    old_seg = self.current_result["segments"][idx]
                    new_item["predicted_class"] = old_seg.get("predicted_class", "?")
                    new_item["predicted_class_id"] = old_seg.get("predicted_class_id", 0)
                    new_item["color"] = old_seg.get("color", (128,128,128))
                    new_item["prediction_id"] = old_seg.get("prediction_id") 
                    new_item["confidence"] = old_seg.get("confidence", new_item["confidence"])
                    
                    self.current_result["segments"][idx] = new_item
                    self.editing_segment_index = None
                    new_idx = idx
                else:
                    self.current_result["segments"].append(new_item)
                    new_idx = len(self.current_result["segments"]) - 1
                
                # Re-draw annotated image
                self.current_result["annotated_image"] = self.pipeline._draw_annotations(
                    img, self.current_result["segments"]
                )
                
                self.after(0, lambda: self._on_polygon_added(new_idx))
            except Exception as e:
                err_msg = f"❌ Error committing polygon: {e}"
                self.after(0, lambda msg=err_msg: self.status.configure(
                    text=msg, text_color="#f85149"
                ))

        threading.Thread(target=_commit, daemon=True).start()

    def _cancel_polygon(self, ev=None):
        """Escape: Cancels the active interactive polygon."""
        if self.active_points or getattr(self, 'editing_segment_index', None) is not None:
            self.active_points = []
            self.active_labels = []
            self.active_preview_seg = None
            self.editing_segment_index = None
            self.editing_bbox = None
            self._render_canvas()
            self.status.configure(text="🚫 Interactive mode cancelled.", text_color="#8b949e")

    def _on_polygon_added(self, index: int):
        self._render_canvas()
        self._show_segments_list()
        self._select_segment(index)
        n = len(self.current_result["segments"])
        self.status.configure(text=f"✨ Polygon added! {n} segments total.", text_color="#7ee787")

    # ── Cleanup ────────────────────────────────────────────────────
    def _on_close(self):
        self.destroy()


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    ReviewApp().mainloop()
